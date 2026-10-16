"""part1/run_timing.py - Section 1.1 timing: 3 versions, 256x256, Gaussian 3..31."""
import time
import numpy as np
import matplotlib.pyplot as plt
from cvlib import conv_loop, conv_vec, conv_sep, gaussian_kernel, save_image
from config import FIGURES_DIR, make_folders

LOOP_MAX_K = 31   # biggest filter timed with the loop version (lower = faster)
REPEATS = 3       # repeats for vec / sep (the loop is timed only once)


def run():
    make_folders()
    np.random.seed(0)
    img = np.random.rand(256, 256)                      # random 256x256 test image
    save_image(img, FIGURES_DIR / "timing_test_image.png")  # so you can look at it
    sizes = list(range(3, 32, 2))                       # 3, 5, 7 ... 31
    res = {"loop": [], "vec": [], "sep": []}

    print("\nfilter   loop(s)     vec(s)      sep(s)")
    for k in sizes:
        K = gaussian_kernel(k / 6, size=k)              # sigma = k/6: proper Gaussian
        for name, fn in [("loop", conv_loop), ("vec", conv_vec), ("sep", conv_sep)]:
            if name == "loop" and k > LOOP_MAX_K:
                res[name].append(np.nan)
                continue
            times = []
            for _ in range(1 if name == "loop" else REPEATS):
                t0 = time.perf_counter()
                fn(img, K)
                times.append(time.perf_counter() - t0)
            res[name].append(np.mean(times))
        print(f"{k:>2}x{k:<2}  {res['loop'][-1]:9.4f}  {res['vec'][-1]:10.5f}  "
              f"{res['sep'][-1]:10.5f}")

    plt.figure(figsize=(7, 4.5))
    for name, style in [("loop", "o-"), ("vec", "s-"), ("sep", "^-")]:
        plt.plot(sizes, res[name], style, label=name)
    plt.yscale("log")                                   # loop is far slower
    plt.xlabel("filter size k (k x k)")
    plt.ylabel("time in seconds (log scale)")
    plt.title("Convolution time on a 256x256 image")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(FIGURES_DIR / "timing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {FIGURES_DIR / 'timing.png'}")
