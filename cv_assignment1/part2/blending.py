"""
part2/blending.py  -  Section 2.4 multiresolution blending.

Blend each Laplacian level with a mask that is blurred MORE at coarse levels:
    L_blend(i) = Gm(i) * L_A(i) + (1 - Gm(i)) * L_B(i)      then collapse.
Coarse structure is therefore mixed over a wide seam, fine detail over a narrow one.
"""
import config  # noqa: F401  (puts cvlib on sys.path, so this module also works on its own)
import numpy as np
from cvlib import gaussian_blur
from pyramids import gaussian_pyramid, laplacian_pyramid, reconstruct


def vertical_mask(shape, split=0.5):
    """1 on the left `split` fraction of the width, 0 on the rest (hard edge)."""
    H, W = shape[:2]
    m = np.zeros((H, W))
    m[:, :int(round(W * split))] = 1.0
    return m


def circle_mask(shape, center=None, radius=None):
    """1 inside a disc, 0 outside (hard edge); defaults to a centred disc."""
    H, W = shape[:2]
    cy, cx = (H / 2, W / 2) if center is None else center
    radius = min(H, W) / 4 if radius is None else radius
    y, x = np.mgrid[0:H, 0:W]
    return (((y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2).astype(np.float64)


def ellipse_mask(shape, center, radii, angle_deg=0.0):
    """1 inside an ellipse, 0 outside. center = (x, y), radii = (rx, ry) in pixels,
    angle_deg rotates it (counter-clockwise on screen). For an irregular mask drawn
    by hand, load a black/white image instead."""
    H, W = shape[:2]
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    t = np.deg2rad(angle_deg)
    u = (x - center[0]) * np.cos(t) - (y - center[1]) * np.sin(t)
    v = (x - center[0]) * np.sin(t) + (y - center[1]) * np.cos(t)
    return ((u / radii[0]) ** 2 + (v / radii[1]) ** 2 <= 1.0).astype(np.float64)


def polygon_mask(shape, points):
    """1 inside the polygon given as [(x, y), ...], 0 outside (hard edge), drawn with
    cv2.fillPoly - the handout allows library functions to draw masks. Use it for an
    irregular shape traced around an object (read the points with show_with_grid)."""
    import cv2
    H, W = shape[:2]
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(np.float64)


def direct_paste(a, b, mask):
    """Baseline 1: cut and paste, visible seam."""
    m = _expand(mask, a)
    return m * a + (1 - m) * b


def feather_blend(a, b, mask, sigma=15):
    """Baseline 2: one blurred mask at a single scale (ghosting or a visible seam)."""
    m = _expand(gaussian_blur(mask, sigma), a)
    return m * a + (1 - m) * b


def multires_blend(a, b, mask, levels=None, return_parts=False):
    """The real thing. a, b: same-shape images; mask: (H, W) in [0, 1] (1 -> take a).
    At every level:  blended = M * L1 + (1 - M) * L2   (M = that level of the mask's
    Gaussian pyramid), then the blended pyramid is collapsed as in 2.3.
    With return_parts=True also returns the per-level pieces, for visualisation:
    la, lb (Laplacians), gm (mask levels), ml1 = M*L1, ml2 = (1-M)*L2, blended = ml1+ml2."""
    if a.shape != b.shape:
        raise ValueError(f"images must have the same shape, got {a.shape} and {b.shape}")
    la = laplacian_pyramid(a, levels)
    lb = laplacian_pyramid(b, levels)
    gm = gaussian_pyramid(mask, len(la))
    ml1, ml2, blended = [], [], []
    for La, Lb, Gm in zip(la, lb, gm):
        m = _expand(Gm, La)
        ml1.append(m * La)
        ml2.append((1 - m) * Lb)
        blended.append(ml1[-1] + ml2[-1])
    out = np.clip(reconstruct(blended), 0, 1)
    if return_parts:
        return out, {"la": la, "lb": lb, "gm": gm, "ml1": ml1, "ml2": ml2, "blended": blended}
    return out


def _expand(mask, like):
    """(H, W) mask -> broadcastable against a gray or colour image."""
    mask = np.asarray(mask, dtype=np.float64)
    return mask[..., None] if like.ndim == 3 and mask.ndim == 2 else mask
