"""
part3/run_carve.py  -  Section 3.1: find and remove seams.
  * shows the ENERGY of the image
  * removes seams (vertical = narrower, horizontal = shorter)
  * draws the removed seams in red on the ORIGINAL image
Saved: figures/<name>/3_1_*.png  and  carved/<name>/3_1_carved_*.png
"""
import time
import numpy as np
from cvlib import load_image, save_image, carve, draw_seams
from cli import pick_images, choose_energy, ask, ask_float, ask_int
from config import FIGURES_DIR, CARVED_DIR, make_folders
from plots import save_row


def run():
    make_folders()
    paths = pick_images(allow_all=False)
    if not paths:
        return
    path = paths[0]
    max_side = ask_int("Max image side in pixels (smaller = faster)", 600)
    percent = ask_float("Reduce the size by how many percent", 30)
    direction = "horizontal" if ask("Seams: v = vertical (narrower), h = horizontal (shorter)",
                                    "v").lower().startswith("h") else "vertical"
    energy_fn, etag = choose_energy()

    img = load_image(path, gray=False, max_side=max_side)
    H, W = img.shape[:2]
    size = H if direction == "horizontal" else W
    n = int(round(size * percent / 100))                   # how many seams to remove
    print(f"\n{path.name}: {W}x{H} px, removing {n} {direction} seams ...")

    t0 = time.perf_counter()
    carved, seams = carve(img, n, energy_fn, direction)    # <- the real work
    print(f"done in {time.perf_counter() - t0:.1f} s, new size {carved.shape[1]}x{carved.shape[0]}")

    # --- the three pictures asked for in 3.1 ---------------------------------
    energy = energy_fn(img)                                # numbers, not a picture yet
    energy_img = energy / energy.max()                     # scale to [0,1] -> a picture
    marked = draw_seams(img, seams, (1, 0, 0), direction)  # removed seams in red

    tag = f"{etag}_{direction[0]}_p{percent:g}"            # settings in the file names
    fig_dir = FIGURES_DIR / path.stem
    car_dir = CARVED_DIR / path.stem
    save_image(energy_img, fig_dir / f"3_1_energy_{etag}.png")
    save_image(marked, fig_dir / f"3_1_seams_{tag}.png")
    save_image(carved, car_dir / f"3_1_carved_{tag}.png")
    save_row([energy_img, marked, carved],
             [f"Energy ({etag})", f"Removed seams ({percent:g}%) in red", "Seam carved"],
             fig_dir / f"3_1_energy_and_seams_{tag}.png")
    print(f"Saved in:\n  {fig_dir}\n  {car_dir}")
