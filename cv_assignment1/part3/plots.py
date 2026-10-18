"""part3/plots.py - small helpers that build the report figures."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # draw to files, never open a window
import matplotlib.pyplot as plt


def _save(path, dpi):
    """plt.savefig does NOT create missing folders, so we do it first."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _show(ax, im, title):
    im = np.clip(im, 0, 1)
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_row(images, titles, path):
    """Images side by side in ONE row. Each panel is as wide as its image, so a
    narrower result really looks narrower than the original."""
    widths = [im.shape[1] for im in images]
    fig, ax = plt.subplots(1, len(images), figsize=(sum(widths) / 100 + 0.4 * len(images), 4.2),
                           gridspec_kw={"width_ratios": widths})
    for a, im, t in zip(np.atleast_1d(ax), images, titles):
        _show(a, im, t)
    _save(path, 130)


def save_grid(images, titles, path, ncols=2):
    """Images in a grid (used for energy maps next to their seams)."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.4 * nrows))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for a, im, t in zip(ax, images, titles):
        _show(a, im, t)
    _save(path, 130)


def save_grid_with_axes(img, path, step=50):
    """The image with a labelled pixel grid, to read coordinates for a mask."""
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(W / 80 + 1, H / 80 + 1))
    ax.imshow(np.clip(img, 0, 1))
    ax.set_xticks(range(0, W, step))
    ax.set_yticks(range(0, H, step))
    ax.grid(True, color="yellow", alpha=0.5, linewidth=0.6)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    _save(path, 110)
