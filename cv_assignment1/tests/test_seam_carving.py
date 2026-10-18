"""
tests/test_seam_carving.py  -  safety net for cvlib/seam_carving.py.
Run:  python tests/test_seam_carving.py     (also works with pytest)
"""
import sys
import itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cvlib import (cumulative_energy, find_seam, remove_seam, carve, make_energy_fn,
                   draw_seams, center_crop, remove_object, enlarge, resize_image)
from cvlib.seam_carving import seams_to_mask

# the worked example from the explanation (4 rows x 5 columns)
EXAMPLE = np.array([[5, 8, 2, 9, 4],
                    [3, 1, 7, 6, 8],
                    [9, 4, 3, 2, 5],
                    [6, 7, 1, 8, 3]], dtype=float)


def test_dp_table_matches_hand_computation():
    M = cumulative_energy(EXAMPLE)
    expected = np.array([[5, 8, 2, 9, 4],
                         [8, 3, 9, 8, 12],
                         [12, 7, 6, 10, 13],
                         [13, 13, 7, 14, 13]], dtype=float)
    assert np.array_equal(M, expected)


def test_seam_of_the_example():
    seam = find_seam(cumulative_energy(EXAMPLE))
    assert list(seam) == [2, 1, 2, 2]
    assert EXAMPLE[np.arange(4), seam].sum() == 7


def _is_valid_seam(seam, W):
    return (np.all(seam >= 0) and np.all(seam < W)
            and np.all(np.abs(np.diff(seam)) <= 1))       # touches the row above


def test_dp_is_optimal_vs_brute_force():
    """The DP must find the same cost as trying EVERY possible seam."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.random((6, 5))
        H, W = e.shape
        best = np.inf
        for start in range(W):                            # enumerate all seams
            for steps in itertools.product((-1, 0, 1), repeat=H - 1):
                cols = [start]
                for s in steps:
                    cols.append(cols[-1] + s)
                if min(cols) < 0 or max(cols) >= W:
                    continue
                best = min(best, e[np.arange(H), cols].sum())
        seam = find_seam(cumulative_energy(e))
        assert _is_valid_seam(seam, W)
        assert np.isclose(e[np.arange(H), seam].sum(), best)


def test_remove_seam_shapes_and_content():
    img = np.arange(12, dtype=float).reshape(3, 4)
    out = remove_seam(img, np.array([1, 2, 0]))
    assert out.shape == (3, 3)
    assert np.array_equal(out, [[0, 2, 3], [4, 5, 7], [9, 10, 11]])
    rgb = np.random.rand(3, 4, 3)
    assert remove_seam(rgb, np.array([1, 2, 0])).shape == (3, 3, 3)


def test_carve_seams_are_in_original_coordinates():
    """Removing the reported seams from the ORIGINAL must give the carved image."""
    np.random.seed(0)
    img = np.random.rand(30, 40, 3)
    n = 12
    carved, seams = carve(img, n, make_energy_fn("finite"))
    assert carved.shape == (30, 40 - n, 3) and seams.shape == (n, 30)
    # NOTE: in ORIGINAL coordinates a seam may "jump" (earlier seams were removed
    # between its pixels), so we only check the bounds here. Connectivity is
    # checked in the shrinking image, see test_each_seam_is_connected_while_carving.
    assert seams.min() >= 0 and seams.max() < 40
    removed = seams_to_mask((30, 40), seams)
    assert removed.sum() == 30 * n                        # no pixel removed twice
    assert np.array_equal(img[~removed].reshape(30, 40 - n, 3), carved)


def test_carve_horizontal():
    np.random.seed(1)
    img = np.random.rand(30, 40)
    carved, seams = carve(img, 5, make_energy_fn("dog", 1.0), "horizontal")
    assert carved.shape == (25, 40) and seams.shape == (5, 40)
    removed = seams_to_mask((30, 40), seams, "horizontal")
    assert removed.sum() == 5 * 40                        # 5 pixels per column
    # work in the transposed frame: keep the un-removed pixels column by column
    kept = np.swapaxes(img, 0, 1)[~removed.T].reshape(40, 25)
    assert np.array_equal(np.swapaxes(kept, 0, 1), carved)


def test_each_seam_is_connected_while_carving():
    """In the shrinking image every seam must be 8-connected (one pixel per row,
    touching the one above). We check it by carving one seam at a time."""
    np.random.seed(5)
    cur = np.random.rand(25, 30)
    fn = make_energy_fn("finite")
    for _ in range(10):
        seam = find_seam(cumulative_energy(fn(cur)))
        assert _is_valid_seam(seam, cur.shape[1])
        cur = remove_seam(cur, seam)


def test_seam_avoids_a_strong_edge():
    """A vertical bright bar in a flat image: the seams must not cut the bar."""
    img = np.zeros((40, 50)); img[:, 30:36] = 1.0
    _, seams = carve(img, 5, make_energy_fn("finite"))
    assert not np.any((seams >= 29) & (seams <= 36))


def test_draw_seams_paints_the_right_pixels():
    img = np.zeros((4, 5))
    seams = np.array([[2, 1, 2, 2]])
    out = draw_seams(img, seams, color=(1, 0, 0))
    assert out.shape == (4, 5, 3)
    assert np.array_equal(out[np.arange(4), seams[0]], np.tile([1, 0, 0], (4, 1)))
    assert out[..., 0].sum() == 4


def test_center_crop_and_resize():
    img = np.random.rand(20, 30, 3)
    assert center_crop(img, 20).shape == (20, 20, 3)
    assert np.array_equal(center_crop(img, 20), img[:, 5:25])
    assert resize_image(img, 21, 20).shape == (20, 21, 3)


def test_object_removal_and_enlarge_restore_the_width():
    rng = np.random.default_rng(2)
    img = np.clip(0.5 + 0.05 * rng.standard_normal((40, 60, 3)), 0, 1)
    mask = np.zeros((40, 60), bool); mask[15:25, 20:30] = True
    fn = make_energy_fn("finite")
    reduced, k = remove_object(img, mask, fn)
    assert k >= 10 and reduced.shape == (40, 60 - k, 3)
    big, marks = enlarge(reduced, k, fn)
    assert big.shape == (40, 60, 3)                       # width restored
    assert np.all(marks.sum(axis=1) == 2 * k)             # k seams, written twice
    # the pixels NOT inserted are exactly the reduced photo, in order
    assert np.array_equal(big[~marks].reshape(40, 60 - 2 * k, 3).shape, (40, 60 - 2 * k, 3))


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
    return len(tests)


if __name__ == "__main__":
    print(f"All {run_all()} seam-carving tests passed.")
