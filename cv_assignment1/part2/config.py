"""part2/config.py - where Part 2 reads and writes files, and how it finds cvlib."""
import sys
from pathlib import Path

PART2 = Path(__file__).resolve().parent


def _find_root():
    """The project root = the nearest folder above this one that contains cvlib/.
    (Searching upward means a nested copy like part2/part2/ still works.)"""
    for p in (PART2, *PART2.parents):
        if (p / "cvlib").is_dir():
            return p
    raise FileNotFoundError(
        f"Could not find the 'cvlib' folder above {PART2}.\n"
        "Part 2 must live inside cv_assignment1/ (next to cvlib/), i.e.\n"
        "  cv_assignment1/cvlib/  and  cv_assignment1/part2/part2_demo.ipynb")


ROOT = _find_root()                                  # cv_assignment1/
for p in (ROOT, PART2):                              # make `cvlib` and the part2 modules importable
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

IMAGES_DIR = PART2 / "images"                        # Part 2 inputs (starter-pack images live here)
SHARED_IMAGES_DIR = ROOT / "images"                  # also searched, for your own extra images
OUT_DIR = PART2 / "outputs"                         # everything Part 2 saves, inside part2/
FIG_DIR = OUT_DIR / "figures"


def make_folders():
    for d in (IMAGES_DIR, OUT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
