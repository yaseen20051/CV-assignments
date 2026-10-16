"""
part1/main.py  -  START HERE:  python part1/main.py   (from the project root)
Menu for Part 1. Put your images in the shared images/ folder first.
"""
import sys
from pathlib import Path

# Make `cvlib` and `tests` importable no matter where the script is launched from.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_filter
import run_edges
import run_timing
from config import IMAGES_DIR, make_folders
from tests.test_cvlib import run_all

MENU = """
================ Part 1 ================
 1) Filter an image (1.1) + check vs scipy
 2) Library tests + timing table (1.1)
 3) Finite differences / edges (1.2)
 4) Derivative of Gaussian (1.3)
 q) Quit
"""

if __name__ == "__main__":
    make_folders()                              # creates all folders on first run
    print(f"Input images folder: {IMAGES_DIR}")
    while True:
        print(MENU)
        c = input("Choose: ").strip().lower()
        if c == "1":
            run_filter.run()
        elif c == "2":
            print(f"All {run_all()} library tests passed.")
            run_timing.run()
        elif c == "3":
            run_edges.run("1.2")
        elif c == "4":
            run_edges.run("1.3")
        elif c == "q":
            break
