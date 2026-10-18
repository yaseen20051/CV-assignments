"""part3/config.py - where Part 3 reads and writes files.
Everything Part 3 writes goes to outputs/part3/ ONLY, so it never mixes with
Part 1 (outputs/part1/) or Part 2 (outputs/part2/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"                       # shared input images
MASKS_DIR = IMAGES_DIR / "masks"                   # optional mask files for the Bonus
OUT_DIR = ROOT / "outputs" / "part3"
CARVED_DIR = OUT_DIR / "carved"                    # result images, one folder per photo
FIGURES_DIR = OUT_DIR / "figures"                  # figures for the report
BONUS_DIR = OUT_DIR / "bonus"                      # Bonus results


def make_folders():
    for d in (IMAGES_DIR, MASKS_DIR, CARVED_DIR, FIGURES_DIR, BONUS_DIR):
        d.mkdir(parents=True, exist_ok=True)
