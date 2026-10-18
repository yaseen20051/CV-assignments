"""
part2/test_part2.py  -  safety net for the Part 2 modules.   python part2/test_part2.py
(also works with pytest). Plain asserts: silent unless something fails.
"""
import config                                    # noqa: F401  (sets up sys.path)
import numpy as np
from hybrid import make_hybrid, high_pass, low_pass, log_fft, to_gray
from aliasing import subsample, antialiased_subsample, stripes
from pyramids import (gaussian_pyramid, laplacian_pyramid, reconstruct, expand,
                      reduce, max_levels, total_size_ratio)
from blending import (multires_blend, vertical_mask, direct_paste, feather_blend,
                      ellipse_mask, polygon_mask)
from synthetic import blobs, texture, sunset, bars


def test_pyramid_shapes_halve():
    pyr = gaussian_pyramid(np.random.rand(65, 100), 4)
    assert [p.shape for p in pyr] == [(65, 100), (33, 50), (17, 25), (9, 13)]


def test_laplacian_reconstructs_exactly_gray_and_colour_odd_sizes():
    np.random.seed(0)
    for shape in [(64, 64), (65, 97), (50, 33, 3)]:
        img = np.random.rand(*shape)
        assert np.allclose(reconstruct(laplacian_pyramid(img)), img, atol=1e-12), shape


def test_expand_rejects_wrong_shape():
    try:
        expand(np.zeros((5, 5)), (20, 20))
    except ValueError:
        return
    raise AssertionError("expand should reject an impossible target shape")


def test_expand_keeps_brightness_even_and_odd_sizes():
    for shape in [(20, 20), (21, 33)]:                       # odd sides: the extra row/col is dropped
        small = np.full(((shape[0] + 1) // 2, (shape[1] + 1) // 2), 0.6)
        assert np.allclose(expand(small, shape), 0.6, atol=1e-12), shape   # 3 zeros in 4, x4: flat stays flat


def test_reduce_and_expand_use_the_handout_kernel():
    k = np.array([1, 4, 6, 4, 1]) / 16.0
    # reduce: blur, then keep rows/cols 0, 2, 4...  An impulse at (10, 10) of a 21x21 image
    # leaves the 3x3 patch outer([1, 6, 1], [1, 6, 1]) / 256 at reduced rows/cols 4..6.
    img = np.zeros((21, 21)); img[10, 10] = 1.0
    patch = reduce(img)[4:7, 4:7]
    assert np.allclose(patch, np.outer(k[[0, 2, 4]], k[[0, 2, 4]]))
    # expand: the pixel goes to the top-left of its 2x2 block (-> (4, 4)), is blurred, times 4
    one = np.zeros((5, 5)); one[2, 2] = 1.0
    e = expand(one, (9, 9))
    assert np.allclose(e[2:7, 2:7], 4 * np.outer(k, k))


def test_total_pyramid_size_is_four_thirds():
    r = total_size_ratio(gaussian_pyramid(np.random.rand(512, 512)))
    assert 1.33 < r < 4 / 3 + 1e-9


def test_max_levels_stops_at_min_side():
    assert max_levels((512, 512)) == 7           # 512 .. 8
    assert max_levels((8, 8)) == 1


def test_hybrid_frequency_split():
    a, b = blobs(), texture()
    r = make_hybrid(a, b, 8, 4, clip=False)
    fa, fb = log_fft(to_gray(a)), log_fft(to_gray(b))
    assert np.allclose(r["hybrid"], r["low"] + r["high"])
    # high-pass of b has ~zero mean, low-pass of a keeps its mean (brightness conserved)
    assert abs(r["high"].mean()) < 1e-2
    assert abs(r["low"].mean() - a.mean()) < 1e-2
    assert fa.shape == fb.shape


def test_hybrid_needs_aligned_images():
    try:
        make_hybrid(np.zeros((10, 10)), np.zeros((12, 10)), 3, 3)
    except ValueError:
        return
    raise AssertionError("misaligned shapes should raise")


def test_antialiasing_removes_fake_low_frequency():
    s = stripes(256, period=2.2)                 # nearly at Nyquist: k=4 must alias
    naive, anti = subsample(s, 4), antialiased_subsample(s, 4)
    assert naive.std() > 3 * anti.std()          # naive keeps a bogus strong pattern; blur kills it


def test_blend_limits():
    a, b = sunset(96, 96), bars(96, 96)
    ones, zeros = np.ones((96, 96)), np.zeros((96, 96))
    assert np.allclose(multires_blend(a, b, ones), a, atol=1e-6)
    assert np.allclose(multires_blend(a, b, zeros), b, atol=1e-6)
    assert np.allclose(multires_blend(a, a, vertical_mask(a.shape)), a, atol=1e-6)


def test_blend_is_smoother_than_direct_paste_at_the_seam():
    a, b = sunset(128, 128), bars(128, 128)
    m = vertical_mask(a.shape)
    seam = slice(60, 68)
    jump = lambda im: np.abs(np.diff(im[:, seam], axis=1)).mean()
    assert jump(feather_blend(a, b, m)) < jump(direct_paste(a, b, m))


def test_blend_level_parts_sum_to_the_blended_level():
    a, b = sunset(96, 96), bars(96, 96)
    out, parts = multires_blend(a, b, vertical_mask(a.shape), levels=4, return_parts=True)
    for ml1, ml2, bl in zip(parts["ml1"], parts["ml2"], parts["blended"]):
        assert np.allclose(ml1 + ml2, bl)
    assert np.allclose(out, np.clip(reconstruct(parts["blended"]), 0, 1))


def test_polygon_mask_square():
    m = polygon_mask((50, 60), [(10, 10), (30, 10), (30, 30), (10, 30)])
    assert m[20, 20] == 1 and m[0, 0] == 0 and 380 <= m.sum() <= 460     # ~ 20 x 20


def test_ellipse_mask_area_and_center():
    m = ellipse_mask((100, 200), center=(100, 50), radii=(40, 20))
    assert m[50, 100] == 1 and m[0, 0] == 0
    assert abs(m.sum() - np.pi * 40 * 20) < 0.03 * np.pi * 40 * 20


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
    return len(tests)


if __name__ == "__main__":
    print(f"All {run_all()} part2 tests passed.")
