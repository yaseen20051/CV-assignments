"""
part2/pyramids.py  -  Section 2.3 Gaussian and Laplacian pyramids.

Both functions blur with the 5-tap binomial kernel  1/16 [1 4 6 4 1]  along x and y
(as the handout requires):

    reduce(img)  : blur, then keep every other row and column      -> ceil(n/2) sizes
    expand(img)  : put each pixel at the top-left of a 2x2 block of zeros,
                   blur, multiply by 4 (three pixels in four are zeros)

    G0 = image,  G(i+1) = reduce(G(i))
    L(i) = G(i) - expand(G(i+1)),   L(last) = G(last)
    G(i) = L(i) + expand(G(i+1))    -> the pyramid reconstructs the image EXACTLY

All blurs are cvlib's separable convolution with reflect padding.
"""
import config  # noqa: F401  (puts cvlib on sys.path, so this module also works on its own)
import numpy as np
from cvlib import convolve

MIN_SIDE = 8                                           # never shrink below 8 px

_K1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
_KERNEL = np.outer(_K1D, _K1D)                         # rank 1: our convolve() splits it


def reduce(img):
    """Blur with 1/16[1 4 6 4 1] (x and y), then keep every second row and column."""
    img = np.asarray(img, dtype=np.float64)
    return convolve(img, _KERNEL, "reflect")[::2, ::2]


def expand(img, out_shape):
    """Place each pixel at the top-left of a 2x2 block of zeros, blur with the same
    kernel, multiply by 4. out_shape is the (H, W) of the finer level: when a side of
    that level is odd, the plain 2x expansion has one row/column too many, and
    building the zero grid at exactly out_shape drops that last one."""
    img = np.asarray(img, dtype=np.float64)
    H, W = out_shape[:2]
    if img.shape[:2] != ((H + 1) // 2, (W + 1) // 2):
        raise ValueError(f"cannot expand {img.shape[:2]} to {(H, W)}")
    out = np.zeros((H, W) + img.shape[2:])
    out[::2, ::2] = img
    return 4.0 * convolve(out, _KERNEL, "reflect")


def max_levels(shape):
    """How many levels fit before the short side drops under MIN_SIDE."""
    n, side = 1, min(shape[:2])
    while side // 2 >= MIN_SIDE:
        side = (side + 1) // 2
        n += 1
    return n


def gaussian_pyramid(img, levels=None):
    """List [G0, G1, ...] with G0 = img (finest first): reduce applied repeatedly."""
    img = np.asarray(img, dtype=np.float64)
    levels = max_levels(img.shape) if levels is None else min(levels, max_levels(img.shape))
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(reduce(pyr[-1]))
    return pyr


def laplacian_pyramid(img, levels=None, gpyr=None):
    """List [L0, L1, ..., G_last]. Each L(i) holds the detail between two scales."""
    g = gaussian_pyramid(img, levels) if gpyr is None else gpyr
    lap = [g[i] - expand(g[i + 1], g[i].shape) for i in range(len(g) - 1)]
    lap.append(g[-1])
    return lap


def reconstruct(lap):
    """Collapse a Laplacian pyramid back into an image (coarse to fine)."""
    img = lap[-1]
    for L in reversed(lap[:-1]):
        img = L + expand(img, L.shape)
    return img


def total_size_ratio(pyr):
    """Pixels in the whole pyramid / pixels in the image (-> 4/3 for many levels)."""
    return sum(p.shape[0] * p.shape[1] for p in pyr) / (pyr[0].shape[0] * pyr[0].shape[1])
