
# NIfTI 4D Visualizer for Streamlit (Patched)
# -------------------------------------------
# Features:
# - Upload .nii / .nii.gz and inspect header metadata
# - 2D orthogonal slice viewer with time slider + optional Cine GIF export
# - Montage
# - Intensity histogram
# - 3D iso-surface (marching cubes) + STL export
# - Volume rendering (robust coordinates) with downsampling
#
# Run:
#   pip install streamlit nibabel numpy plotly scikit-image imageio pandas
#   streamlit run nii_streamlit_app.py

import pandas as pd
# import plotly.graph_objects as go
import numpy as np
import streamlit as st
from PIL import Image
import io

st.set_page_config(page_title="NIfTI 4D Visualizer", layout="wide")

# --------- External libs (soft errors) ---------
try:
    import nibabel as nib
except Exception as e:
    st.error("need install nibabel：pip install nibabel")
    raise

# --------------------------
# Utility functions
# --------------------------

from pathlib import Path
import os

def locate_inr_script() -> Path:
    """
    Try several common locations to find inr_insert5_keep30.py.
    Order: same folder as this app, CWD, env var, /mnt/data (deploy).
    """
    candidates = [
        Path(__file__).with_name("inr_insert5_keep30.py"),
        Path.cwd() / "inr_insert5_keep30.py",
    ]
    env_path = os.environ.get("INR_SCRIPT")
    if env_path:
        candidates.insert(0, Path(env_path))  # env override first

    # deploy fallback
    candidates.append(Path("/mnt/data/inr_insert5_keep30.py"))

    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "Could not find inr_insert5_keep30.py. "
        "Tried: " + ", ".join(str(p) for p in candidates)
    )

def human_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024.0:
            return f"{n:3.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def load_nii(file_or_bytes):
    """
    Robust loader for Streamlit uploads across nibabel versions.
    Writes bytes to a temporary file on disk, then uses nib.load(path).
    Accepts: streamlit.UploadedFile, bytes/bytearray/memoryview, or path-like.
    """
    import tempfile, os

    # If already a path-like string, load directly
    if isinstance(file_or_bytes, str):
        img = nib.load(file_or_bytes)
        dataobj = img.get_fdata(dtype=np.float32)
        return img, dataobj, img.header, img.affine

    # Resolve to bytes and name hint
    file_bytes = None
    file_name_hint = "upload.nii.gz"
    if hasattr(file_or_bytes, "read"):  # Streamlit UploadedFile
        try:
            file_name_hint = getattr(file_or_bytes, "name", file_name_hint) or file_name_hint
        except Exception:
            pass
        file_bytes = file_or_bytes.read()
    elif isinstance(file_or_bytes, (bytes, bytearray, memoryview)):
        file_bytes = bytes(file_or_bytes)
    else:
        raise TypeError("load_nii need upload object or byte or path string")

    # choose suffix
    suffix = ".nii.gz"
    try:
        if file_name_hint.lower().endswith(".nii"):
            suffix = ".nii"
        elif file_name_hint.lower().endswith(".nii.gz"):
            suffix = ".nii.gz"
        else:
            # gzip magic check
            if file_bytes[:2] == b'\x1f\x8b':
                suffix = ".nii.gz"
            else:
                suffix = ".nii"
    except Exception:
        pass

    # write to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()

    img = nib.load(tmp.name)
    dataobj = img.get_fdata(dtype=np.float32)
    hdr = img.header
    affine = img.affine
    return img, dataobj, hdr, affine

def get_dims(data: np.ndarray):
    if data.ndim == 3:
        x, y, z = data.shape
        t = 1
    elif data.ndim == 4:
        x, y, z, t = data.shape
    else:
        raise ValueError(f"Unsupported data ndim={data.ndim}")
    return x, y, z, t

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    if vmax <= vmin + 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - vmin) / (vmax - vmin)
    return (scaled * 255).clip(0,255).astype(np.uint8)

def build_metadata_table(img, hdr, file_name: str) -> pd.DataFrame:
    shape = hdr.get_data_shape()
    zooms = hdr.get_zooms() if hasattr(hdr, 'get_zooms') else ()
    dtype = hdr.get_data_dtype()
    xyzt_units = getattr(hdr, 'get_xyzt_units', lambda: ('unknown','unknown'))()
    descrip = hdr.get('descrip', b'').tobytes().decode('utf-8','ignore').strip() if 'descrip' in hdr else ''
    bitpix = int(hdr['bitpix']) if 'bitpix' in hdr else None

    # Compatible with different nibabel versions
    intent_code = intent_name = intent_params = None
    if hasattr(hdr, "get_intent"):
        _intent = hdr.get_intent()
        try:
            if len(_intent) == 3:
                intent_code, intent_params, intent_name = _intent
            elif len(_intent) == 2:
                intent_code, intent_name = _intent
            elif len(_intent) == 1:
                intent_code = _intent[0]
        except Exception:
            pass

    voxels = int(np.prod(shape))
    bytes_per_voxel = np.dtype(dtype).itemsize
    mem_size = voxels * bytes_per_voxel
    rows = [
        ("file_name", file_name),
        ("shape", str(shape)),
        ("zooms", str(zooms)),
        ("xyzt_units", f"{xyzt_units[0]} / {xyzt_units[1]}"),
        ("dtype", str(dtype)),
        ("bitpix", bitpix),
        ("intent", f"{intent_code} / {intent_name}"),
        ("intent_params", str(intent_params) if intent_params is not None else "—"),
        ("descrip", descrip if descrip else "—"),
        ("voxels", f"{voxels:,}"),
        ("estimated_memory", human_bytes(mem_size)),
    ]
    return pd.DataFrame(rows, columns=["Key", "Value"])

# def plotly_mesh(vertices, faces):
#     x, y, z = vertices.T
#     i, j, k = faces.T
#     mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=1.0, flatshading=True)
#     fig = go.Figure(data=[mesh])
#     fig.update_layout(scene_aspectmode="data", margin=dict(l=0,r=0,b=0,t=30))
#     return fig

# def plotly_volume(volume, opacity=0.12):
#     # volume: 3D ndarray normalized to [0,1]
#     nx, ny, nz = volume.shape
#     # Build full coordinate grid so x,y,z align with values
#     xx, yy, zz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
#     vals = volume.flatten()
#     fig = go.Figure(
#         data=go.Volume(
#             x=xx.flatten(),
#             y=yy.flatten(),
#             z=zz.flatten(),
#             value=vals,
#             opacity=opacity,
#             surface_count=18,
#             isomin=float(vals.min()),
#             isomax=float(vals.max() if np.isfinite(vals.max()) else 1.0),
#             caps=dict(x_show=False, y_show=False, z_show=False),
#         )
#     )
#     fig.update_layout(scene_aspectmode="data", margin=dict(l=0,r=0,b=0,t=30))
#     return fig

# def export_gif(frames, duration_ms=80):
#     buf = io.BytesIO()
#     imageio.mimsave(buf, frames, format='GIF', duration=duration_ms/1000.0)
#     buf.seek(0)
#     return buf

def export_stl(vertices, faces):
    # Minimal ASCII STL writer
    import io as _io
    buf = _io.StringIO()
    buf.write("solid surface\n")
    v = vertices
    f = faces
    def facet(n, a, b, c):
        buf.write(f"  facet normal {n[0]} {n[1]} {n[2]}\n")
        buf.write("    outer loop\n")
        buf.write(f"      vertex {a[0]} {a[1]} {a[2]}\n")
        buf.write(f"      vertex {b[0]} {b[1]} {b[2]}\n")
        buf.write(f"      vertex {c[0]} {c[1]} {c[2]}\n")
        buf.write("    endloop\n")
        buf.write("  endfacet\n")
    for tri in f:
        a, b, c = v[tri[0]], v[tri[1]], v[tri[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n) + 1e-12
        n = n / n_norm
        facet(n, a, b, c)
    buf.write("endsolid surface\n")
    b = io.BytesIO(buf.getvalue().encode("utf-8"))
    b.seek(0)
    return b

# =========================
# Helpers (place near other helpers)
# =========================
def orient_slice(vol3d: np.ndarray, axis: str, idx: int) -> np.ndarray:
    """Return a 2D slice in viewer-friendly orientation."""
    if axis == "Axial (Z)":
        sl = vol3d[:, :, idx]
    elif axis == "Coronal (Y)":
        sl = vol3d[:, idx, :]
    else:  # "Sagittal (X)"
        sl = vol3d[idx, :, :]
    return np.rot90(sl)

def normalize_uint8(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Normalize to 0..255 uint8 using either provided lo/hi or robust percentiles."""
    arr = np.asarray(arr)
    if lo is None or hi is None:
        lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr)) if np.nanmax(arr) != np.nanmin(arr) else (0.0, 1.0)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)

# --------------------------
# Sidebar: Upload + controls
# --------------------------
st.sidebar.title("NIfTI 4D Visualizer")

uploaded = st.sidebar.file_uploader("upload .nii.gz", type=["nii.gz"])
if not uploaded:
    st.info("Please upload NIfTI file to continue")
    st.stop()


# Load
with st.spinner("loading..."):
    img, data, hdr, affine = load_nii(uploaded)

x, y, z, t = get_dims(data)
zooms = hdr.get_zooms() if hasattr(hdr, 'get_zooms') else ()
TR = zooms[3] if len(zooms) >= 4 else None
space_units, time_units = getattr(hdr, 'get_xyzt_units', lambda: ('unknown','unknown'))()

# --------------------------
# Main layout
# --------------------------
st.title("NIfTI 4D Visualizer")
cols = st.columns([1,1,1])
with cols[0]:
    st.metric("Dimensions", f"{x}×{y}×{z}" + (f"×{t}" if t>1 else ""))
with cols[1]:
    st.metric("Voxel Size", " × ".join([f"{zooms[i]:.2f}" for i in range(min(3,len(zooms)))] ) + f" {space_units}")
with cols[2]:
    st.metric("TR", f"{TR} {time_units}" if TR else "—")

with st.expander("Metadata", expanded=False):
    meta_df = build_metadata_table(img, hdr, uploaded.name)
    st.dataframe(meta_df, use_container_width=True)

tab_slices, tab_montage, tab_slice_loop, tab_upload, tab_ml = st.tabs([
    "2D slice",
    "Montage",
    "Slice Loop (index)",
    "ML upload",
    "ML slice"
])


# --------------------------
# 2D Slice Viewer
# --------------------------
with tab_slices:
    left, right = st.columns([1,2])
    with left:
        axis = st.selectbox("Axis", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], index=0)
        if axis.startswith("Axial"):
            max_idx = z-1
        elif axis.startswith("Coronal"):
            max_idx = y-1
        else:
            max_idx = x-1

        slicer = st.slider("Slice index", 0, max_idx, max_idx//2)
        t_idx = st.slider("Time frame t", 0, t-1, 0) if t>1 else 0
        use_equalize = st.checkbox("Normalize", value=False)
        # cine_gif = st.checkbox("Export Cine GIF")

        # if cine_gif and t <= 1:
        #     st.warning("current data is not 4D，cannot export Cine。")

    with right:
        # get 2D slice
        if axis.startswith("Axial"):
            img2d = data[:, :, slicer, t_idx] if t>1 else data[:, :, slicer]
        elif axis.startswith("Coronal"):
            img2d = data[:, slicer, :, t_idx] if t>1 else data[:, slicer, :]
            img2d = np.rot90(img2d)
        else:
            img2d = data[slicer, :, :, t_idx] if t>1 else data[slicer, :, :]
            img2d = np.rot90(img2d)

        disp = img2d.copy()
        if use_equalize:
            p1, p99 = np.percentile(disp, [1,99])
            if p99 - p1 < 1e-6:
                disp = normalize_to_uint8(disp)
            else:
                disp = np.clip((disp - p1) / max(p99 - p1, 1e-6), 0, 1)
                disp = (disp * 255).astype(np.uint8)
        else:
            disp = normalize_to_uint8(disp)

        st.image(disp, caption=f"{axis} | slice={slicer} | t={t_idx}", clamp=True, use_container_width=True)


# --------------------------
# Montage
# --------------------------
with tab_montage:
    m_col1, m_col2 = st.columns([1,2])
    with m_col1:
        axis_m = st.selectbox("Axis (Montage)", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], index=0, key="montage_axis")
        t_idx_m = st.slider("Time frame t", 0, t-1, 0, key="montage_t") if t>1 else 0
        step = st.slider("Step", 1, max(1, min(x,y,z)//8), 4)
    with m_col2:
        # Build montage grid
        if axis_m.startswith("Axial"):
            slices = data[:,:,::step,t_idx_m] if t>1 else data[:,:,::step]
        elif axis_m.startswith("Coronal"):
            tmp = data[:,::step,:,t_idx_m] if t>1 else data[:,::step,:]
            slices = np.rot90(tmp, axes=(0,2))
        else:
            tmp = data[::step,:,:,t_idx_m] if t>1 else data[::step,:,:]
            slices = np.rot90(tmp, axes=(1,2))

        # Arrange into grid
        n_slices = slices.shape[-1]
        cols_m = int(np.ceil(np.sqrt(n_slices)))
        rows_m = int(np.ceil(n_slices/cols_m))
        target_h = 128
        target_w = 128
        canvas = np.zeros((rows_m*target_h, cols_m*target_w), dtype=np.uint8)
        idx = 0
        for r in range(rows_m):
            for c in range(cols_m):
                if idx >= n_slices: break
                sl = slices[..., idx]
                sl = normalize_to_uint8(sl)
                sh, sw = sl.shape
                rh = target_h / sh
                rw = target_w / sw
                yy = (np.arange(target_h) / rh).astype(int).clip(0, sh-1)
                xx = (np.arange(target_w) / rw).astype(int).clip(0, sw-1)
                slr = sl[yy[:,None], xx[None,:]]
                canvas[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = slr
                idx += 1

        st.image(canvas, caption=f"Montage: {axis_m} @ t={t_idx_m} (step={step})", use_container_width=True)

#animation
with tab_slice_loop:
    is_4d = (data.ndim == 4)
    x, y, z = data.shape[:3]
    t = data.shape[3] if is_4d else 1

    ctl_top = st.columns([1,1,1,1])
    axis_sl = ctl_top[0].selectbox("Axis", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
                                   index=0, key="axis_slice_loop")
    fps_sl  = ctl_top[1].slider("FPS", 2, 30, 10, key="fps_slice_loop")
    # pingpong_time_sl = ctl_top[2].toggle("Ping-pong (time)", value=True, key="pingpong_time_slice_loop")
    ctl_top[3].markdown("&nbsp;")

    if not is_4d or t < 2:
        st.info("This dataset has no time dimension; per-slice cine needs 4D data.")
    else:
        n_slices_sl = {"Axial (Z)": z, "Coronal (Y)": y, "Sagittal (X)": x}[axis_sl]
        slice_idx_sl = st.slider("Slice index", 0, n_slices_sl - 1, 0, key="slice_idx_slice_loop")

        ctl2 = st.columns([1,1,1,1])
        time_step_sl = ctl2[0].slider("Time step (subsample)", 1, max(1, t // 2 if t > 2 else 1), 1,
                                      key="time_step_slice_loop")
        fit_mode_sl = ctl2[1].radio("Fit", ["Width", "Height"], index=1, key="fit_mode_slice_loop", horizontal=True)
        out_w_sl = ctl2[2].slider("Max width (px)", 256, 1024, 512, step=64, key="out_w_slice_loop")
        out_h_sl = ctl2[3].slider("Max height (px)", 256, 1024, 384, step=32, key="out_h_slice_loop")

        wl = st.session_state.get("slice_wl")
        lo_hi = (float(wl[0]), float(wl[1])) if (wl and len(wl) == 2) else (None, None)

        # Build time order
        time_order_sl = list(range(0, t, int(max(1, time_step_sl))))
        # if pingpong_time_sl and len(time_order_sl) > 2:
        #     time_order_sl = time_order_sl + time_order_sl[-2:0:-1]

        # Frames
        frames_sl = []
        for ti in time_order_sl:
            vol3d = data[..., ti]
            s2d = orient_slice(vol3d, axis_sl, slice_idx_sl)
            img = Image.fromarray(normalize_uint8(s2d, *lo_hi))
            # Resize by chosen constraint
            if fit_mode_sl == "Height":
                if img.height != out_h_sl:
                    target_h = int(out_h_sl)
                    target_w = int(round(img.width * (target_h / img.height)))
                    img = img.resize((target_w, target_h), Image.BILINEAR)
            else:  # Fit by Width
                if img.width != out_w_sl:
                    target_w = int(out_w_sl)
                    target_h = int(round(img.height * (target_w / img.width)))
                    img = img.resize((target_w, target_h), Image.BILINEAR)
            frames_sl.append(img.convert("P", palette=Image.ADAPTIVE))

        if not frames_sl:
            st.warning("No frames to render. Try reducing the time step.")
        else:
            buf_sl = io.BytesIO()
            frames_sl[0].save(
                buf_sl, format="GIF", save_all=True, append_images=frames_sl[1:],
                loop=0, duration=int(1000 / max(1, int(fps_sl))), optimize=True, disposal=2
            )
            buf_sl.seek(0)

            c_left, c_mid, c_right = st.columns([1, 3, 1])
            with c_mid:
                st.image(
                    buf_sl,
                    caption=f"{axis_sl} • slice {slice_idx_sl} • frames={len(time_order_sl)} • "
                            f"{'height' if fit_mode_sl == 'Height' else 'width'}-limited",
                    use_container_width=False,  # we already sized the frames
                )
            # st.download_button(
            #     "Download GIF",
            #     buf_sl,
            #     file_name=f"cine_single_slice_{axis_sl.split()[0].lower()}_{slice_idx_sl}_fit{fit_mode_sl[0]}_fps{fps_sl}.gif",
            #     mime="image/gif",
            #     key="download_slice_time_loop",
            # )

# nii_streamlit_app.py
# Minimal 4D NIfTI viewer + INR "insert frames" runner with side-by-side comparison
import os
import sys
from pathlib import Path
import tempfile

import numpy as np
import nibabel as nib
import streamlit as st

# ----------------------------
# Session defaults
# ----------------------------
for k, v in {
    "uploaded_nifti_path": None,
    "orig_arr": None,
    "orig_hdr": None,
    "ml_out_path": None,
    "ml_arr": None,
    "ml_hdr": None,
    "ml_err": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Helpers
# ----------------------------
def persist_uploaded_nifti(uploaded) -> str:
    """Safely save Streamlit UploadedFile to disk and return the path."""
    try:
        data = uploaded.getvalue()
    except AttributeError:
        # Already a path-like
        if isinstance(uploaded, str) and os.path.exists(uploaded):
            return uploaded
        raise

    if not data:
        # Last attempt: rewind and read
        try:
            uploaded.seek(0)
            data = uploaded.read()
        except Exception:
            pass
    if not data:
        raise ValueError("Uploaded file buffer is empty (size 0).")

    name = getattr(uploaded, "name", "upload.nii.gz")
    if name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif name.endswith(".nii"):
        suffix = ".nii"
    else:
        suffix = ".nii.gz"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    return path


def _ensure_4d(arr: np.ndarray) -> np.ndarray:
    """Guarantee shape (X,Y,Z,T)."""
    if arr.ndim == 3:
        return arr[..., np.newaxis]
    return arr


def _slice_2d(arr4d: np.ndarray, axis: str, idx: int, t_idx: int) -> np.ndarray:
    """Return a normalized 2D slice by axis/index/time."""
    x, y, z, t = arr4d.shape
    idx = int(np.clip(idx, 0, {"Sagittal (X)": x-1, "Coronal (Y)": y-1, "Axial (Z)": z-1}[axis]))
    t_idx = int(np.clip(t_idx, 0, t-1))
    if axis == "Axial (Z)":
        img2d = arr4d[:, :, idx, t_idx]
    elif axis == "Coronal (Y)":
        img2d = arr4d[:, idx, :, t_idx]
    else:  # Sagittal (X)
        img2d = arr4d[idx, :, :, t_idx]

    img2d = img2d.astype(np.float32)
    if np.isfinite(img2d).any():
        p1, p99 = np.percentile(img2d, (1, 99))
        if p99 > p1:
            img2d = (img2d - p1) / (p99 - p1)
        else:
            img2d = np.zeros_like(img2d)
    else:
        img2d = np.zeros_like(img2d)
    return img2d


def _centered_image(img2d: np.ndarray, clamp_height_px: int = 420):
    """Render a centered image with constrained height using matplotlib."""
    import matplotlib.pyplot as plt
    h = clamp_height_px / 96.0  # inches at ~96 DPI
    fig = plt.figure(figsize=(h, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img2d, cmap="gray", aspect="equal")
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)


def run_inr_insert(
    in_path: Path,
    out_dir: Path,
    *,
    k_phases: int = 3,
    steps: int = 3000,
    n_points: int = 16000,
    ed_idx: int = 0,
    sx: int = 8, sy: int = 8, sz: int = 24,
    z_heavy: float = 2.5,
    use_amp: bool = True,
):
    import subprocess

    # OLD: script = Path("/mnt/data/inr_insert5_keep30.py")
    script = locate_inr_script()  # <-- use the local script
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "inr.ckpt"

    cmd = [
        sys.executable, str(script),
        "--in_nifti", str(in_path),
        "--out_dir", str(out_dir),
        "--ed_idx", str(ed_idx),
        "--steps", str(steps),
        "--k_phases", str(k_phases),
        "--n_points", str(n_points),
        "--sx", str(sx), "--sy", str(sy), "--sz", str(sz),
        "--z_heavy", str(z_heavy),
        "--save_ckpt", str(ckpt),
    ]
    if use_amp:
        cmd.append("--amp")

    st.caption(f"Using INR script: {script}")
    st.caption("Running: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"INR script failed.\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")

    outs = list(out_dir.glob("*.nii*"))
    if not outs:
        raise FileNotFoundError("No output NIfTI found in INR out_dir.")
    return outs[0], proc.stderr or ""



# ----------------------------
# UI
# ----------------------------

# ===== Tab: Upload & Preview =====
# tab_upload, tab_ml = st.tabs(["Upload", "ML: Insert Frames"])

with tab_upload:
    uploaded = st.file_uploader("Upload NIfTI (.nii or .nii.gz)", type=["nii", "nii.gz"])

    if uploaded is not None:
        try:
            disk_path = persist_uploaded_nifti(uploaded)
            st.session_state.uploaded_nifti_path = disk_path

            # Load original
            img = nib.load(disk_path)
            arr = _ensure_4d(img.get_fdata())
            st.session_state.orig_arr = arr
            st.session_state.orig_hdr = img.header

            st.success(f"Uploaded: {uploaded.name}")
            st.caption(f"Saved to: {disk_path} • size: {os.path.getsize(disk_path)} bytes")
        except Exception as e:
            st.error("Failed to read the uploaded NIfTI.")
            st.code(str(e))
            st.stop()


# ===== Tab: ML: Insert Frames =====
with (tab_ml):
    st.subheader("Insert Frames with ML (INR) and Compare")

    if not st.session_state.get("uploaded_nifti_path"):
        st.info("Upload a 4D NIfTI first on the upload tab.")
        st.stop()

    # Settings
    with st.expander("ML settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        k_phases = c1.number_input("Frames to insert (K)", 1, 20, 5, 1)
        steps = c2.number_input("Training steps", 10, 100000, 3000, 500)
        n_points = c3.number_input("Batch points", 1000, 200000, 16000, 1000)
        # d1, d2, d3, d4 = st.columns(4)
        # ed_idx = d1.number_input("ED frame index", 0, 9999, 0, 1)
        # sx = d2.number_input("Grid sx", 1, 256, 8, 1)
        # sy = d3.number_input("Grid sy", 1, 256, 8, 1)
        # sz = d4.number_input("Grid sz", 1, 256, 24, 1)
        # e1, e2 = st.columns(2)
        # z_heavy = e1.number_input("Z heavy", 0.1, 10.0, 2.5, 0.1)
        # use_amp = e2.checkbox("Enable AMP (mixed precision)", value=True)

        # k_phases: int = 3,
        # steps: int = 3000,
        # n_points: int = 16000,
        ed_idx: int = 0
        sx = 8
        sy = 8
        sz= 24
        z_heavy = 2.5
        use_amp = True

    run_btn = st.button("Run ML with settings above", type="secondary")
    if run_btn:
        try:
            in_path = Path(st.session_state.uploaded_nifti_path)
            workdir = Path(tempfile.mkdtemp(prefix="inr_out_"))
            with st.spinner("Running INR with custom settings…"):
                out_path, stderr_txt = run_inr_insert(
                    in_path=in_path,
                    out_dir=workdir,
                    k_phases=int(k_phases),
                    steps=int(steps),
                    n_points=int(n_points),
                    ed_idx=int(ed_idx),
                    sx=int(sx), sy=int(sy), sz=int(sz),
                    z_heavy=float(z_heavy),
                    use_amp=bool(use_amp),
                )
                ml_img = nib.load(str(out_path))
                ml_arr = _ensure_4d(ml_img.get_fdata())
                st.session_state.ml_out_path = str(out_path)
                st.session_state.ml_hdr = ml_img.header
                st.session_state.ml_arr = ml_arr
                st.session_state.ml_err = stderr_txt
            st.success(f"Done. Output: {out_path.name}")
        except Exception as e:
            st.session_state.ml_out_path = None
            st.session_state.ml_arr = None
            st.session_state.ml_hdr = None
            st.session_state.ml_err = str(e)
            st.error("ML insertion failed.")
            st.code(str(e))

    # Comparison UI
    if st.session_state.orig_arr is not None and st.session_state.ml_arr is not None:
        st.divider()
        st.markdown("### Compare Original vs ML-Inserted")

        orig = st.session_state.orig_arr
        ml = st.session_state.ml_arr
        x1, y1, z1, t1 = orig.shape
        x2, y2, z2, t2 = ml.shape

        # axis = st.radio("Axis", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], horizontal=True, index=0)
        # --- inside the Compare Original vs ML-Inserted section ---

        axis = st.radio(
            "Axis",
            ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
            horizontal=True, index=0, key="axis_compare"
        )

        # use a DIFFERENT name to avoid clobbering any earlier dim_map
        size_pairs = {
            "Sagittal (X)": (orig.shape[0], ml.shape[0]),
            "Coronal (Y)": (orig.shape[1], ml.shape[1]),
            "Axial (Z)": (orig.shape[2], ml.shape[2]),
        }

        # get the shared max slice index safely
        ox, mx = size_pairs[axis]
        max_idx = max(0, min(ox, mx) - 1)

        c = st.columns([1, 1, 1])
        sidx = c[0].slider("Slice index", 0, max_idx, max_idx // 2, 1, key="sidx_compare")
        to = c[1].slider("Original time", 0, t1 - 1, 0, 1, key="to_compare")
        tm = c[2].slider("ML-inserted time", 0, t2 - 1, 0, 1, key="tm_compare")

        left, right = st.columns(2, gap="large")
        with left:
            st.caption(f"Original • shape={orig.shape}")
            _centered_image(_slice_2d(orig, axis, sidx, to), clamp_height_px=420)
        with right:
            st.caption(f"ML-Inserted • shape={ml.shape}")
            _centered_image(_slice_2d(ml, axis, sidx, tm), clamp_height_px=420)

        with st.expander("Debug logs", expanded=False):
            st.write("Uploaded path:", st.session_state.uploaded_nifti_path)
            st.write("ML output path:", st.session_state.ml_out_path)
            if st.session_state.ml_err:
                st.code(st.session_state.ml_err)
    else:
        st.info("Run the ML insertion to generate the comparison view.")
