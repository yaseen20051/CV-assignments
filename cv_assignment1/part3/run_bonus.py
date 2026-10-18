"""
part3/run_bonus.py  -  Bonus: remove an object, then grow the photo back.

  1. you give a mask over the object (type a shape, or load a mask file)
  2. seams are carved THROUGH the mask until no masked pixel is left (k seams)
  3. the photo is widened back by k inserted seams (width restored)

Saved in outputs/part3/bonus/<name>/ :
  bonus_0_grid.png        the photo with a pixel grid (to read coordinates)
  bonus_1_mask.png        the photo with the mask in red
  bonus_2_removed.png     the photo without the object (narrower)
  bonus_3_restored.png    width restored
  bonus_3_restored_seams.png   same, with the inserted seams in green
  bonus_overview.png      all of the above in one figure
"""
import numpy as np
from PIL import Image, ImageDraw
from cvlib import load_image, save_image, remove_object, enlarge
from cli import pick_images, choose_energy, ask, ask_int
from config import BONUS_DIR, MASKS_DIR, make_folders
from plots import save_row, save_grid_with_axes


def mask_from_shape(W, H):
    """Ask for an ellipse / rectangle in pixel coordinates -> bool mask (H, W)."""
    shape = ask("Shape: e = ellipse, r = rectangle", "e").lower()
    print("Type the bounding box:  x0 y0 x1 y1  (read it from bonus_0_grid.png)")
    x0, y0, x1, y1 = [float(v) for v in ask("box").replace(",", " ").split()]
    canvas = Image.new("L", (W, H), 0)                 # black canvas
    draw = ImageDraw.Draw(canvas)
    (draw.ellipse if shape.startswith("e") else draw.rectangle)([x0, y0, x1, y1], fill=255)
    return np.asarray(canvas) > 127                    # white = object


def mask_from_file(stem, W, H):
    """Load images/masks/<name>_mask.png (white = object), resized to the photo."""
    path = MASKS_DIR / f"{stem}_mask.png"
    if not path.exists():
        raise FileNotFoundError(f"mask file not found: {path}")
    m = Image.open(path).convert("L").resize((W, H), Image.NEAREST)
    return np.asarray(m) > 127


def run():
    make_folders()
    paths = pick_images(allow_all=False)
    if not paths:
        return
    path = paths[0]
    max_side = ask_int("Max image side in pixels (smaller = faster)", 500)
    img = load_image(path, gray=False, max_side=max_side)
    H, W = img.shape[:2]
    out = BONUS_DIR / path.stem

    save_grid_with_axes(img, out / "bonus_0_grid.png")
    print(f"\nPhoto size: {W} x {H} px. A grid was saved to read coordinates:\n  {out / 'bonus_0_grid.png'}")

    how = ask("Mask: 1 = type a shape, 2 = load images/masks/<name>_mask.png", 1)
    mask = mask_from_file(path.stem, W, H) if how == "2" else mask_from_shape(W, H)
    if not mask.any():
        print("The mask is empty, nothing to remove.")
        return
    energy_fn, _ = choose_energy()

    print("\nRemoving the object ...")
    reduced, k = remove_object(img, mask, energy_fn)
    print(f"Removed {k} seams -> width {reduced.shape[1]}. Growing back by {k} seams ...")
    restored, marks = enlarge(reduced, k, energy_fn)

    # --- pictures ---------------------------------------------------------------
    with_mask = img.copy()
    with_mask[mask] = 0.5 * with_mask[mask] + 0.5 * np.array([1.0, 0, 0])   # red overlay
    with_seams = restored.copy()
    with_seams[marks] = (0.0, 1.0, 0.0)                                      # inserted = green

    save_image(with_mask, out / "bonus_1_mask.png")
    save_image(reduced, out / "bonus_2_removed.png")
    save_image(restored, out / "bonus_3_restored.png")
    save_image(with_seams, out / "bonus_3_restored_seams.png")
    save_row([with_mask, reduced, with_seams],
             ["Mask over the object, in red", f"Object gone ({k} seams removed)",
              "Width restored, new seams in green"], out / "bonus_overview.png")
    print(f"Saved in {out}  (restored width {restored.shape[1]} = original {W})")
