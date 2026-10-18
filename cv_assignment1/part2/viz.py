"""
part2/viz.py  -  every plot used by the Part 2 notebook / report.
Each function shows its figure (plt.show) and optionally saves it to save_path.
"""
import config  # noqa: F401  (puts cvlib on sys.path, so this module also works on its own)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cvlib import signed_for_show
from hybrid import hybrid_spectra, log_fft, to_gray
from pyramids import gaussian_pyramid

plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 10, "image.interpolation": "nearest"})


def _imshow(ax, im, title="", signed=False, spectrum=False, frame=False):
    if signed:
        im = signed_for_show(im)
    if im.ndim == 2:
        if spectrum:                                           # clip outliers so detail is visible
            ax.imshow(im, cmap="gray", vmin=np.percentile(im, 2), vmax=im.max())
        else:
            ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(np.clip(im, 0, 1))
    ax.set_title(title)
    if frame:                                                  # white masks vanish on a white page
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.6")
    else:
        ax.axis("off")


def _finish(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_row(images, titles=None, suptitle=None, size=3.2, save_path=None):
    """Quick row of images."""
    n = len(images)
    titles = titles or [""] * n
    fig, axes = plt.subplots(1, n, figsize=(size * n, size + 0.4))
    for ax, im, t in zip(np.atleast_1d(axes), images, titles):
        _imshow(ax, im, t)
    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    _finish(fig, save_path)


# ------------------------------------------------------------------ 2.1 hybrids
def show_hybrid_report(res, names=("Image A (low)", "Image B (high)"), save_path=None):
    """The handout's figure: top row = the five images, bottom row = their log-FFTs."""
    spectra = hybrid_spectra(res)
    top = [res["im_low"], res["im_high"], res["low"], res["high"], res["hybrid"]]
    titles = [names[0], names[1], f"{names[0]}, low-pass",
              f"{names[1]}, high-pass", "Hybrid"]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.6), constrained_layout=True)
    for j in range(5):
        _imshow(axes[0, j], top[j], titles[j], signed=(j == 3))   # high-pass has negatives
        _imshow(axes[1, j], spectra[j], "log |FFT|", spectrum=True)
    fig.suptitle(f"Hybrid image   sigma_low = {res['sigma_low']:g},  "
                 f"sigma_high = {res['sigma_high']:g}", fontsize=12)
    _finish(fig, save_path)


def show_distance_strip(img, levels=5, save_path=None):
    """Simulate viewing distance: the hybrid at full size, 1/2, 1/4 ... Each tile is
    drawn at its true pixel size, so the small ones look like 'far away'."""
    pyr = gaussian_pyramid(img, levels)
    widths = [p.shape[1] for p in pyr]
    fig = plt.figure(figsize=(1.0 + sum(widths) / 90, 3.2))
    gs = GridSpec(1, len(pyr), width_ratios=widths, figure=fig)
    for i, p in enumerate(pyr):
        ax = fig.add_subplot(gs[i])
        _imshow(ax, p, f"1/{2 ** i}")
        ax.set_anchor("S")
    fig.suptitle("Hybrid at decreasing size (= increasing viewing distance)", y=1.03)
    _finish(fig, save_path)


def show_sigma_grid(make_hybrid, im_low, im_high, sigma_lows, sigma_highs, save_path=None):
    """Grid of hybrids for several (sigma_low, sigma_high) pairs: helps you tune them."""
    fig, axes = plt.subplots(len(sigma_highs), len(sigma_lows),
                             figsize=(3 * len(sigma_lows), 3 * len(sigma_highs)), squeeze=False)
    for i, sh in enumerate(sigma_highs):
        for j, sl in enumerate(sigma_lows):
            r = make_hybrid(im_low, im_high, sl, sh)
            _imshow(axes[i, j], r["hybrid"], f"low {sl:g} / high {sh:g}")
    plt.tight_layout()
    _finish(fig, save_path)


# ------------------------------------------------------------------ 2.2 aliasing
def _blockup(img, k):
    """Repeat each pixel k x k times so a subsampled image has its original size again."""
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def show_aliasing(original, comparisons, save_path=None):
    """comparisons = aliasing.aliasing_comparison(...). One row per factor k:
    naive vs anti-aliased, each with its spectrum (images blown back up to full size)."""
    rows = len(comparisons)
    fig, axes = plt.subplots(rows, 5, figsize=(16, 3.3 * rows), squeeze=False)
    for r, (k, naive, anti) in enumerate(comparisons):
        trio = [(original, "original"),
                (_blockup(naive, k), f"k={k}: naive subsample"),
                (_blockup(anti, k), f"k={k}: blur, then subsample")]
        for c, (im, t) in enumerate(trio):
            _imshow(axes[r, c], im, t)
        _imshow(axes[r, 3], log_fft(to_gray(naive)), "spectrum: naive", spectrum=True)
        _imshow(axes[r, 4], log_fft(to_gray(anti)), "spectrum: blurred first", spectrum=True)
    plt.tight_layout()
    _finish(fig, save_path)


def show_quarter_side_by_side(naive, anti, k, sigma, original_shape, save_path=None):
    """The two shrunk results side by side (nearest-neighbour pixels, no smoothing), so
    every false circle is a real pixel pattern. constrained_layout keeps the titles apart."""
    h, w = naive.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4 * h / w + 0.9), constrained_layout=True)
    _imshow(axes[0], naive, f"keep every {k}th pixel")
    _imshow(axes[1], anti, f"blur (sigma={sigma:g}), then every {k}th pixel")
    fig.suptitle(f"{original_shape[1]}x{original_shape[0]} shrunk to {w}x{h}  (k = {k})", fontsize=10)
    _finish(fig, save_path)


# ------------------------------------------------------------------ 2.3 pyramids
def _collage(levels, signed):
    """Lay the levels side by side on one canvas (finest on the left)."""
    tiles = []
    for L in levels:
        if signed:
            tiles.append(signed_for_show(L, np.abs(L).max() or 1.0))   # per-level contrast
        else:
            tiles.append(np.clip(L, 0, 1))
    H = tiles[0].shape[0]
    gap = 6
    canvas = np.ones((H, sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)) + tiles[0].shape[2:])
    x = 0
    for t in tiles:
        canvas[:t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + gap
    return canvas


def show_pyramid(levels, title, signed=False, save_path=None):
    """One wide figure showing every pyramid level at its true relative size.
    signed=True (Laplacian): 0 -> mid-gray, each level stretched on its own."""
    canvas = _collage(levels, signed)
    fig, ax = plt.subplots(figsize=(15, 15 * canvas.shape[0] / canvas.shape[1] + 0.6))
    _imshow(ax, canvas, title)
    _finish(fig, save_path)


def show_pyramid_pair(gpyr, lpyr, save_path=None):
    """Gaussian pyramid above, Laplacian pyramid below."""
    g, l = _collage(gpyr, False), _collage(lpyr, True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 15 * (g.shape[0] + l.shape[0]) / g.shape[1] + 1))
    _imshow(axes[0], g, "Gaussian pyramid (blur + subsample)")
    _imshow(axes[1], l, "Laplacian pyramid (detail per scale; last = coarsest Gaussian)")
    plt.tight_layout()
    _finish(fig, save_path)


def show_reconstruction(img, rebuilt, save_path=None):
    """Original, rebuilt-from-Laplacian, and the (amplified) error."""
    err = np.abs(img - rebuilt)
    print(f"max |original - reconstruction| = {err.max():.2e}")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    _imshow(axes[0], img, "original")
    _imshow(axes[1], rebuilt, "reconstructed")
    _imshow(axes[2], err / (err.max() or 1.0), "error (stretched to full range)")
    plt.tight_layout()
    _finish(fig, save_path)


# ------------------------------------------------------------------ 2.4 blending
def show_blend_comparison(a, b, mask, outputs, titles, save_path=None):
    """A, B, mask, then any number of blend results."""
    ims = [a, b, mask] + list(outputs)
    tt = ["image A", "image B", "mask"] + list(titles)
    n = len(ims)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    for k, (ax, im, t) in enumerate(zip(axes, ims, tt)):
        _imshow(ax, im, t, frame=(k == 2))                     # the mask panel gets a border
    plt.tight_layout()
    _finish(fig, save_path)


def show_blend_levels(parts, picks=None, save_path=None):
    """Per-level story (the handout's figure): the mask level M, then M*L1, (1-M)*L2
    and their sum, which is the blended level. The last level is not a difference
    (it is the coarsest Gaussian), so it is shown unsigned."""
    n = len(parts["la"])
    picks = list(range(n)) if picks is None else picks
    fig, axes = plt.subplots(len(picks), 4, figsize=(12, 3 * len(picks)), squeeze=False)
    for r, i in enumerate(picks):
        last = i == n - 1
        _imshow(axes[r, 0], parts["gm"][i], f"mask M  level {i}", frame=True)
        _imshow(axes[r, 1], parts["ml1"][i], f"M * L1  level {i}", signed=not last)
        _imshow(axes[r, 2], parts["ml2"][i], f"(1-M) * L2  level {i}", signed=not last)
        _imshow(axes[r, 3], parts["blended"][i], f"sum  level {i}", signed=not last)
    plt.tight_layout()
    _finish(fig, save_path)
