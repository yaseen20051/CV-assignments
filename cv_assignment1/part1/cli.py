"""
part1/cli.py  -  everything that talks to the user (questions and menus).
Kept out of cvlib on purpose: the library must stay free of input()/print().
"""
import numpy as np
from cvlib import list_images, gaussian_kernel, DX, DY
from config import IMAGES_DIR


def ask(prompt, default=None):
    """input() that returns `default` when the user just presses Enter."""
    text = input(f"{prompt} [{default}]: " if default is not None else f"{prompt}: ")
    text = text.strip()
    return text if text else (str(default) if default is not None else "")


def ask_float(prompt, default):
    return float(ask(prompt, default))


def pick_images(allow_all=True):
    """List the images in images/ and let the user pick one (or all)."""
    files = list_images(IMAGES_DIR)
    if not files:
        print(f"\nNo images found. Put some images in:\n  {IMAGES_DIR}")
        return []
    print("\nImages found:")
    for n, p in enumerate(files, 1):
        print(f"  {n}) {p.name}")
    hint = "number" + (", or 'a' for all" if allow_all else "")
    while True:
        c = ask(f"Choose an image ({hint})").lower()
        if allow_all and c == "a":
            return files
        if c.isdigit() and 1 <= int(c) <= len(files):
            return [files[int(c) - 1]]
        print("  invalid choice, try again")


def parse_filter(text):
    """'1 0 -1; 2 0 -2; 1 0 -1' -> 2D array. Rows split by ';', numbers by
    spaces or commas. Both sides must be odd."""
    rows = [[float(x) for x in r.replace(",", " ").split()]
            for r in text.split(";") if r.strip()]
    K = np.array(rows)                         # fails if rows differ in length
    if K.ndim != 2:
        raise ValueError("all rows must have the same number of entries")
    if K.shape[0] % 2 == 0 or K.shape[1] % 2 == 0:
        raise ValueError("filter height and width must both be odd (1, 3, 5, 7 ...)")
    return K


def choose_filter():
    """Menu of preset filters + a custom matrix. Returns (K, short_name)."""
    print("\nFilters:")
    print("  1) Gaussian blur")
    print("  2) Box blur (mean)")
    print("  3) Finite difference Dx  [1 0 -1]")
    print("  4) Finite difference Dy  (Dx transposed)")
    print("  5) Sobel x")
    print("  6) Sharpen")
    print("  7) Custom: type your own matrix")
    while True:
        c = ask("Choose a filter", 1)
        if c == "1":
            s = ask_float("  sigma", 2)
            return gaussian_kernel(s), f"gauss_s{s:g}"
        if c == "2":
            k = int(ask("  size (odd number)", 5))
            return np.ones((k, k)) / (k * k), f"box{k}"
        if c == "3":
            return DX, "Dx"
        if c == "4":
            return DY, "Dy"
        if c == "5":
            return np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float), "sobel_x"
        if c == "6":
            return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float), "sharpen"
        if c == "7":
            print("  Example (sides must be odd):  1 0 -1; 2 0 -2; 1 0 -1")
            try:
                K = parse_filter(ask("  matrix"))
                total = K.sum()
                # An entry sum above 1 makes the image brighter (sum 31 -> 31x
                # brighter -> all white). Dividing by the sum keeps the brightness.
                # The convolution itself never normalizes (it must match scipy);
                # this choice belongs to the menu. Filters that sum to 0 or less
                # (derivatives) are left alone on purpose.
                if total > 0 and abs(total - 1) > 1e-9:
                    answer = ask(f"  entries sum to {total:g}: normalize so they sum to 1? (y/n)", "y")
                    if answer.lower().startswith("y"):
                        K = K / total
                return K, "custom"
            except ValueError as e:
                print(f"  invalid matrix: {e}")
                continue
        print("  invalid choice, try again")