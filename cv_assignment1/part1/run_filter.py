"""
part1/run_filter.py  -  filter images with OUR convolution and check them.
  1. pick an image (or all) from images/        2. pick a filter and a version
  3. our result   -> outputs/part1/filtered_manual/
  4. scipy result -> outputs/part1/filtered_scipy/   (the correct answer)
  5. compare the two raw results and print MATCH / MISMATCH
"""
import numpy as np
from scipy.signal import convolve2d
from cvlib import conv_loop, conv_vec, conv_sep, conv_color, load_image, save_image, signed_for_show
from cli import pick_images, choose_filter, ask
from config import MANUAL_DIR, SCIPY_DIR, make_folders

VERSIONS = {"1": ("loop", conv_loop), "2": ("vec", conv_vec), "3": ("sep", conv_sep)}


def scipy_reference(img, K):
    """The correct answer: scipy convolution per channel, zero padding."""
    if img.ndim == 2:
        return convolve2d(img, K, mode="same")
    return np.dstack([convolve2d(img[..., c], K, mode="same")
                      for c in range(img.shape[2])])


def run():
    make_folders()
    paths = pick_images()
    if not paths:
        return
    K, kname = choose_filter()
    print("\nConvolution version:  1) loop (very slow)  2) vectorized  3) separable")
    vname, conv_fn = VERSIONS.get(ask("Choose a version", 3), VERSIONS["3"])

    # The loop version needs minutes on big images, so shrink the image for it.
    max_side = 100 if vname == "loop" else 1000
    if vname == "loop":
        print("  (loop version: image shrunk to 100 px, otherwise it takes forever)")

    # Derivative-like filters give negative values: show them with 0 as mid-gray.
    signed = K.min() < 0 and abs(K.sum()) < 1e-6

    for path in paths:
        img = load_image(path, gray=False, max_side=max_side)
        mine = conv_color(img, K, conv=conv_fn)              # OUR convolution
        ref = scipy_reference(img, K)                        # scipy answer
        max_diff = np.abs(mine - ref).max()
        ok = np.allclose(mine, ref)

        if signed:
            scale = np.abs(ref).max()
            mine_out, ref_out = signed_for_show(mine, scale), signed_for_show(ref, scale)
        else:
            mine_out, ref_out = mine, ref

        name = f"{path.stem}__{kname}__{vname}.png"
        save_image(mine_out, MANUAL_DIR / name)
        save_image(ref_out, SCIPY_DIR / name)
        print(f"\n{path.name}: max difference vs scipy = {max_diff:.2e}  ->  "
              f"{'MATCH' if ok else 'MISMATCH  (there is a bug!)'}")
        print(f"  ours : {MANUAL_DIR / name}\n  scipy: {SCIPY_DIR / name}")
