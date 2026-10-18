"""
part3/run_compare.py  -  Section 3.2: seam carving vs resize vs center crop,
with TWO energy functions (finite differences and DoG).

For every chosen image it saves (name = image name, p30 = 30 % narrower):
  figures/<name>/3_2_comparison_p30.png    original | resized | cropped | seam (finite) | seam (DoG)
  figures/<name>/3_2_energy_seams_p30.png  the energy maps and the seams each energy picks
  carved/<name>/3_2_*.png                  every result as its own image
"""
import time
import numpy as np
from cvlib import (load_image, save_image, make_energy_fn, carve, draw_seams,
                   seams_to_mask, center_crop, resize_image)
from cli import pick_images, ask_float, ask_int
from config import FIGURES_DIR, CARVED_DIR, make_folders
from plots import save_row, save_grid


def process_image(path, max_side, percent, sigma):
    img = load_image(path, gray=False, max_side=max_side)
    H, W = img.shape[:2]
    n = int(round(W * percent / 100))                  # seams to remove
    new_w = W - n                                      # same final width for ALL methods
    print(f"\n{path.name}: {W}x{H} -> width {new_w}  ({n} seams)")

    fig_dir, car_dir = FIGURES_DIR / path.stem, CARVED_DIR / path.stem
    p = f"p{percent:g}"

    # the two simple options
    resized = resize_image(img, new_w, H)              # squashes everything
    cropped = center_crop(img, new_w)                  # throws the sides away

    # seam carving with each energy function
    energies = [("finite", make_energy_fn("finite")),
                (f"dog_s{sigma:g}", make_energy_fn("dog", sigma))]
    carved, marked, energy_maps, removed = {}, {}, {}, {}
    for tag, fn in energies:
        t0 = time.perf_counter()
        carved[tag], seams = carve(img, n, fn)
        print(f"  seam carving with {tag}: {time.perf_counter() - t0:.1f} s")
        marked[tag] = draw_seams(img, seams, (1, 0, 0))
        e = fn(img)
        energy_maps[tag] = e / e.max()
        removed[tag] = seams_to_mask((H, W), seams)

    tags = [t for t, _ in energies]
    # how similar are the two choices? (share of removed pixels picked by both)
    both = (removed[tags[0]] & removed[tags[1]]).sum() / removed[tags[0]].sum()
    print(f"  the two energies picked the same pixels for {100 * both:.0f}% of the seams")

    # --- individual images ----------------------------------------------------
    save_image(resized, car_dir / f"3_2_resized_{p}.png")
    save_image(cropped, car_dir / f"3_2_cropped_{p}.png")
    for t in tags:
        save_image(carved[t], car_dir / f"3_2_seam_{t}_{p}.png")
        save_image(marked[t], fig_dir / f"3_2_seams_{t}_{p}.png")
        save_image(energy_maps[t], fig_dir / f"3_2_energy_{t}.png")

    # --- the figures for the report --------------------------------------------
    save_row([img, resized, cropped, carved[tags[0]], carved[tags[1]]],
             [f"Original ({W} px)", f"Resized ({new_w} px)", f"Cropped ({new_w} px)",
              f"Seam carved, {tags[0]}", f"Seam carved, {tags[1]}"],
             fig_dir / f"3_2_comparison_{p}.png")
    save_grid([energy_maps[tags[0]], energy_maps[tags[1]], marked[tags[0]], marked[tags[1]]],
              [f"Energy: {tags[0]}", f"Energy: {tags[1]}",
               f"Seams picked by {tags[0]}", f"Seams picked by {tags[1]}"],
             fig_dir / f"3_2_energy_seams_{p}.png")
    print(f"  saved in {fig_dir}")


def run():
    make_folders()
    paths = pick_images(allow_all=True)
    if not paths:
        return
    max_side = ask_int("Max image side in pixels (smaller = faster)", 600)
    percent = ask_float("Reduce the width by how many percent", 30)
    sigma = ask_float("DoG sigma", 2)
    for path in paths:
        process_image(path, max_side, percent, sigma)
