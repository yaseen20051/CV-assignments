"""part1/config.py - where Part 1 reads and writes files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"                       # shared input images
OUT_DIR = ROOT / "outputs" / "part1"
MANUAL_DIR = OUT_DIR / "filtered_manual"           # made by OUR convolution
SCIPY_DIR = OUT_DIR / "filtered_scipy"             # scipy's correct answer
FIGURES_DIR = OUT_DIR / "figures"                  # figures for the report


def make_folders():
    for d in (IMAGES_DIR, MANUAL_DIR, SCIPY_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
