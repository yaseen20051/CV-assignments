"""
part1/run_edges.py  -  Sections 1.2 (finite differences) and 1.3 (DoG).
Figures go to outputs/part1/figures/<image name>/ for the report.
"""
import numpy as np
import matplotlib.pyplot as plt
from cvlib import (DX, DY, dog_filters, gradient_maps, threshold_edges,
                   load_image, save_image, signed_for_show)
from cli import pick_images, ask_float
from config import FIGURES_DIR, make_folders


def save_four(dx, dy, mag, edges, titles, path):
    """One figure with the 4 maps side by side."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    imgs = [signed_for_show(dx), signed_for_show(dy),      # negatives -> mid-gray
            mag / mag.max(), edges.astype(float)]
    for a, im, t in zip(ax, imgs, titles):
        a.imshow(im, cmap="gray", vmin=0, vmax=1)
        a.set_title(t)
        a.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_separate(dx, dy, mag, edges, out, prefix):
    """Save each map as its OWN image file (e.g. the edge image alone).
    `prefix` carries the settings (sigma / percentile) so runs never overwrite."""
    save_image(signed_for_show(dx), out / f"{prefix}_dx.png")     # 0 = mid-gray
    save_image(signed_for_show(dy), out / f"{prefix}_dy.png")
    save_image(mag / mag.max(), out / f"{prefix}_magnitude.png")
    save_image(edges.astype(float), out / f"{prefix}_edges.png")  # white = edge


def save_threshold_comparison(mag, pcts, path):
    """Edge maps at several thresholds: the evidence for 'Justify your threshold'."""
    fig, ax = plt.subplots(1, len(pcts), figsize=(4 * len(pcts), 4))
    for a, p in zip(ax, pcts):
        a.imshow(threshold_edges(mag, percentile=p)[0], cmap="gray")
        a.set_title(f"percentile {p}")
        a.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def run_12(path):
    """Section 1.2: finite difference operator."""
    out = FIGURES_DIR / path.stem
    out.mkdir(parents=True, exist_ok=True)
    img = load_image(path, gray=True)
    pct = ask_float("Edge threshold percentile (90 = keep strongest 10%)", 90)

    dx, dy, mag = gradient_maps(img, DX, DY)
    edges, t = threshold_edges(mag, percentile=pct)
    tag = f"1_2_p{pct:g}"                              # settings in the file names
    save_four(dx, dy, mag, edges,
              ["dI/dx", "dI/dy", "Gradient magnitude", "Edges"], out / f"{tag}_maps.png")
    save_separate(dx, dy, mag, edges, out, tag)         # edges alone, magnitude alone...
    save_threshold_comparison(mag, [80, 90, 95, 98], out / "1_2_thresholds.png")
    print(f"1.2 done. threshold = {t:.4f} (percentile {pct:g}). Saved in {out}")


def run_13(path):
    """Section 1.3: derivative of Gaussian (blur folded into the filter)."""
    out = FIGURES_DIR / path.stem
    out.mkdir(parents=True, exist_ok=True)
    img = load_image(path, gray=True)
    sigma = ask_float("Gaussian sigma", 2)
    pct = ask_float("Edge threshold percentile (tune it again for DoG)", 90)

    dog_x, dog_y = dog_filters(sigma)                 # the two DoG filters
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for a, f, name in zip(ax, [dog_x, dog_y], ["DoG in x", "DoG in y"]):
        a.imshow(signed_for_show(f), cmap="gray", vmin=0, vmax=1)
        a.set_title(f"{name}, sigma={sigma:g}")
        a.axis("off")
    plt.savefig(out / f"1_3_filters_s{sigma:g}.png", dpi=150, bbox_inches="tight")
    plt.close()

    dx, dy, mag = gradient_maps(img, dog_x, dog_y)    # one convolution per filter
    edges, t = threshold_edges(mag, percentile=pct)
    tag = f"1_3_s{sigma:g}_p{pct:g}"                   # settings in the file names
    save_four(dx, dy, mag, edges,
              ["DoG x response", "DoG y response", "Gradient magnitude", "Edges"],
              out / f"{tag}_maps.png")
    save_separate(dx, dy, mag, edges, out, tag)         # edges alone, magnitude alone...
    save_threshold_comparison(mag, [80, 90, 95, 98], out / f"1_3_thresholds_s{sigma:g}.png")
    print(f"1.3 done. sigma = {sigma:g}, threshold = {t:.4f} (percentile {pct:g}). "
          f"Saved in {out}")


def run(which):
    make_folders()
    paths = pick_images(allow_all=False)
    if paths:
        (run_12 if which == "1.2" else run_13)(paths[0])
