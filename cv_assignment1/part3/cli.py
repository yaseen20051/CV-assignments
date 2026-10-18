"""
part3/cli.py  -  everything that talks to the user (questions and menus).
Kept out of cvlib on purpose: the library must stay free of input()/print().
"""
from cvlib import list_images, make_energy_fn
from config import IMAGES_DIR


def ask(prompt, default=None):
    """input() that returns `default` when the user just presses Enter."""
    text = input(f"{prompt} [{default}]: " if default is not None else f"{prompt}: ")
    text = text.strip()
    return text if text else (str(default) if default is not None else "")


def ask_float(prompt, default):
    return float(ask(prompt, default))


def ask_int(prompt, default):
    return int(ask(prompt, default))


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


def choose_energy():
    """Menu of energy functions. Returns (energy_fn, short_tag)."""
    print("\nEnergy function:")
    print("  1) Finite differences (Section 1.2)")
    print("  2) Derivative of Gaussian, DoG (Section 1.3)")
    while True:
        c = ask("Choose an energy function", 1)
        if c == "1":
            return make_energy_fn("finite"), "finite"
        if c == "2":
            s = ask_float("  sigma", 2)
            return make_energy_fn("dog", s), f"dog_s{s:g}"
        print("  invalid choice, try again")
