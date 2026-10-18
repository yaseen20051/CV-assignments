"""
part3/main.py  -  START HERE:  python part3/main.py   (from the project root)
Menu for Part 3 (seam carving). Put your images in the shared images/ folder.
"""
import sys
from pathlib import Path

# Make `cvlib` and `tests` importable no matter where the script is launched from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_carve
import run_compare
import run_bonus
from config import IMAGES_DIR, OUT_DIR, make_folders
from tests.test_cvlib import run_all as run_conv_tests
from tests.test_seam_carving import run_all as run_seam_tests

MENU = """
================ Part 3 ================
 1) Energy + seams: find and remove seams (3.1)
 2) Compare: seam carving vs resize vs crop, 2 energies (3.2)
 3) Bonus: remove an object and grow the photo back
 4) Library tests
 q) Quit
"""

if __name__ == "__main__":
    make_folders()                              # creates all folders on first run
    print(f"Input images folder: {IMAGES_DIR}")
    print(f"Part 3 results go to: {OUT_DIR}")
    while True:
        print(MENU)
        c = input("Choose: ").strip().lower()
        if c == "1":
            run_carve.run()
        elif c == "2":
            run_compare.run()
        elif c == "3":
            run_bonus.run()
        elif c == "4":
            print(f"All {run_conv_tests()} convolution tests passed.")
            print(f"All {run_seam_tests()} seam-carving tests passed.")
        elif c == "q":
            break
