"""
cvlib/imgio.py  -  loading, saving and displaying helpers for images.
(Named imgio, not imageio, so it does not clash with the imageio package.)
"""
from pathlib import Path
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder):
    """Sorted list of image files inside a folder (empty list if none)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(path, gray=False, max_side=1000):
    """Load an image as float64 in [0,1]; shrink it if its long side > max_side.
    gray=True -> (H, W);  gray=False -> (H, W, 3)."""
    im = Image.open(path).convert("L" if gray else "RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
    return np.asarray(im, dtype=np.float64) / 255.0


def save_image(arr, path):
    """Save a float image in [0,1] (values outside are clipped). Creates the
    parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).round().astype(np.uint8)).save(path)


def signed_for_show(d, scale=None):
    """Map values that can be negative so that 0 becomes mid-gray (0.5).
    Use it for derivatives, DoG, high-pass and Laplacian levels."""
    d = np.asarray(d, dtype=np.float64)
    scale = np.abs(d).max() if scale is None else scale
    scale = scale if scale > 0 else 1.0
    return 0.5 + d / (2 * scale)


def resize_image(img, new_w, new_h):
    """Normal (squashing) resize of a gray or colour float image in [0,1].
    A library resize is allowed by the assignment ONLY for the 3.2 comparison."""
    img = np.asarray(img, dtype=np.float64)
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    out = []
    for ch in chans:
        # mode "F" = 32-bit float image, so we do not lose precision
        pil = Image.fromarray(ch.astype(np.float32), mode="F")
        out.append(np.asarray(pil.resize((new_w, new_h), Image.LANCZOS), dtype=np.float64))
    return np.clip(out[0] if img.ndim == 2 else np.dstack(out), 0.0, 1.0)
