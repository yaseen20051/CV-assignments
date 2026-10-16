"""
tests/test_cvlib.py  -  safety net for the SHARED library.
Run it after ANY change to cvlib:   python tests/test_cvlib.py
(also works with pytest). Every test is a plain assert, nothing is printed
unless something fails.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find cvlib

import numpy as np
from scipy.signal import convolve2d
from cvlib import (conv_loop, conv_vec, conv_sep, separate, convolve,
                   gaussian_kernel, gaussian_blur, dog_filters)


def test_three_versions_match_scipy(n_trials=30):
    """Section 1.1 sanity check: random small images, random odd filters,
    half separable and half not. Zero padding, compared with scipy."""
    np.random.seed(0)
    for t in range(n_trials):
        H, W = np.random.randint(8, 16, 2)
        kh, kw = np.random.choice([1, 3, 5, 7]), np.random.choice([1, 3, 5, 7])
        img = np.random.rand(H, W)
        K = (np.outer(np.random.randn(kh), np.random.randn(kw)) if t % 2 == 0
             else np.random.randn(kh, kw))
        ref = convolve2d(img, K, mode="same")
        for fn in (conv_loop, conv_vec, conv_sep):
            assert np.allclose(fn(img, K), ref), f"{fn.__name__} failed, K {kh}x{kw}"


def test_reflect_padding_versions_agree():
    """With reflect padding the three versions must agree with each other."""
    np.random.seed(1)
    img, K = np.random.rand(12, 14), np.random.randn(5, 3)
    a, b, c = (fn(img, K, "reflect") for fn in (conv_loop, conv_vec, conv_sep))
    assert np.allclose(a, b) and np.allclose(b, c)


def test_identity_filter_returns_image():
    img = np.random.rand(9, 11)
    K = np.zeros((3, 3)); K[1, 1] = 1
    assert np.allclose(convolve(img, K), img)


def test_separate_detects_rank():
    assert separate(gaussian_kernel(2)) is not None
    np.random.seed(2)
    assert separate(np.random.randn(5, 5)) is None


def test_even_filter_rejected():
    try:
        conv_vec(np.zeros((5, 5)), np.ones((2, 2)))
    except ValueError:
        return
    raise AssertionError("an even-sized filter should raise ValueError")


def test_gaussian_properties():
    G = gaussian_kernel(2)
    assert G.shape == (13, 13) and abs(G.sum() - 1) < 1e-12
    # blurring a constant image (with reflect padding) leaves it unchanged
    assert np.allclose(gaussian_blur(np.full((20, 20), 0.7), 1.5), 0.7)


def test_dog_matches_blur_then_derivative():
    """Associativity: (Dx * G) * I  ==  Dx * (G * I) away from the borders.
    A big kernel (31) is used so truncating the DoG to the Gaussian's size
    (which is a tiny approximation, ~1e-3 for 13x13) does not matter."""
    from cvlib import DX
    np.random.seed(3)
    img = np.random.rand(60, 60)
    dog_x, _ = dog_filters(2, size=31)
    a = conv_sep(img, dog_x)
    b = conv_sep(conv_sep(img, gaussian_kernel(2, 31)), DX)
    assert np.allclose(a[16:-16, 16:-16], b[16:-16, 16:-16], atol=1e-10)


def test_color_matches_per_channel():
    np.random.seed(4)
    img, K = np.random.rand(10, 10, 3), np.random.randn(3, 5)
    out = convolve(img, K)
    for c in range(3):
        assert np.allclose(out[..., c], convolve2d(img[..., c], K, mode="same"))


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
    return len(tests)


if __name__ == "__main__":
    print(f"All {run_all()} library tests passed.")
