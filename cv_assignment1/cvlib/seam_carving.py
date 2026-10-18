"""
cvlib/seam_carving.py  -  seam carving (Part 3 + Bonus). Pure functions only.

THE BIG PICTURE
  1. ENERGY   : a number per pixel = "how important is this pixel?"
                (edges are important, flat sky is not).
  2. SEAM     : one pixel per row, each touching the one above it
                (directly below it, or diagonally). Its COST = sum of energies.
  3. CHEAPEST : dynamic programming finds the seam with the lowest cost.
  4. REMOVE   : delete that seam, recompute the energy, repeat.

Conventions: images are float64 in [0,1], (H, W) or (H, W, 3).
A seam is an int array of length H: seam[i] = column of the seam in row i.
"""
import numpy as np
from .filters import DX, DY, dog_filters
from .gradients import gradient_magnitude

BIG_NEG = -1e6      # energy given to masked pixels in the Bonus (cheapest seam goes there)


# --------------------------------------------------------------------------
# 1. ENERGY
# --------------------------------------------------------------------------
def to_gray(img):
    """Luminance gray image (H, W). Colour is mixed with the usual eye weights."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return img
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])


def make_energy_fn(kind="finite", sigma=2.0):
    """Build an energy function  energy_fn(img) -> (H, W) array.

    kind="finite": gradient magnitude with Dx / Dy        (Section 1.2)
    kind="dog"   : gradient magnitude with DoG filters    (Section 1.3), uses sigma

    The filters are built ONCE here, not on every call. Reflect padding is used
    so the image frame does not create fake edges (zero padding would).
    """
    if kind == "finite":
        kx, ky = DX, DY
    elif kind == "dog":
        kx, ky = dog_filters(sigma)
    else:
        raise ValueError("kind must be 'finite' or 'dog'")

    def energy_fn(img):
        return gradient_magnitude(to_gray(img), kx, ky, pad_mode="reflect")
    return energy_fn


# --------------------------------------------------------------------------
# 2./3. DYNAMIC PROGRAMMING: the cheapest seam
# --------------------------------------------------------------------------
def cumulative_energy(energy):
    """M[i, j] = cost of the CHEAPEST seam that starts anywhere in row 0 and
    ends at pixel (i, j).

    HINT (the key idea of the whole algorithm):
      A seam that ends at (i, j) must have come from one of the three pixels
      just above it:  (i-1, j-1), (i-1, j), (i-1, j+1).
      So the cheapest way to reach (i, j) is:
          energy[i, j]  +  the cheapest of those three.
      We already know their costs (they are in row i-1), so we fill M row by
      row, top to bottom, and never compute anything twice:
          M[i, j] = energy[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
      First row: M[0, j] = energy[0, j]  (nothing above it).
      This costs H*W steps. Trying every seam would cost about 3^H.
    """
    M = np.array(energy, dtype=np.float64)          # copy; row 0 is already right
    H, W = M.shape
    for i in range(1, H):                           # row by row, top to bottom
        above = M[i - 1]
        # Shift the row above to line up the left / right candidates.
        # HINT: at the borders a neighbour does not exist -> use +inf so that
        # min() never picks it.
        left = np.empty(W)
        left[0] = np.inf
        left[1:] = above[:-1]                       # left[j]  = M[i-1, j-1]
        right = np.empty(W)
        right[-1] = np.inf
        right[:-1] = above[1:]                      # right[j] = M[i-1, j+1]
        M[i] += np.minimum(np.minimum(left, above), right)
    return M


def find_seam(M):
    """Trace the cheapest seam back from the bottom row. Returns int array (H,).

    HINT: M tells us the COST of the best seam, not its PATH. To get the path
    we walk backwards. The best seam ends at the smallest value of the last
    row. Then, from each pixel, we step to whichever of the three pixels above
    it has the smallest M (that is exactly where the DP value came from).
    """
    H, W = M.shape
    seam = np.zeros(H, dtype=int)
    seam[-1] = np.argmin(M[-1])                     # cheapest end point
    for i in range(H - 2, -1, -1):                  # walk up: row H-2 ... 0
        j = seam[i + 1]                             # where we are in the row below
        lo, hi = max(0, j - 1), min(W, j + 2)       # allowed columns: j-1, j, j+1
        seam[i] = lo + np.argmin(M[i, lo:hi])       # pick the cheapest of them
    return seam


# --------------------------------------------------------------------------
# 4. REMOVE a seam, and carve many seams
# --------------------------------------------------------------------------
def remove_seam(arr, seam):
    """Delete pixel (i, seam[i]) from every row i. Works for (H,W), (H,W,C), bool...

    HINT: build a True/False mask the size of the image, set the seam pixels to
    False, and keep the True ones. Every row loses exactly one pixel, so the
    kept pixels (read row by row) reshape neatly into (H, W-1).
    """
    arr = np.asarray(arr)
    H, W = arr.shape[:2]
    keep = np.ones((H, W), dtype=bool)
    keep[np.arange(H), seam] = False
    return arr[keep].reshape((H, W - 1) + arr.shape[2:])


def carve(img, n, energy_fn, direction="vertical"):
    """Remove n seams. Returns (carved_image, seams).

    direction="vertical"  : n vertical seams  -> image gets n columns narrower
    direction="horizontal": n horizontal seams -> image gets n rows shorter
    seams: int array (n, H): seams[k, i] = column of the k-th removed seam in
           row i, measured in the ORIGINAL image (not the shrinking one).
           For horizontal carving the roles of rows and columns are swapped.
    """
    img = np.asarray(img, dtype=np.float64)
    if direction == "horizontal":
        # HINT: a horizontal seam is just a vertical seam of the transposed
        # image, so swap rows and columns, carve, and swap back.
        carved_t, seams = carve(np.swapaxes(img, 0, 1), n, energy_fn, "vertical")
        return np.swapaxes(carved_t, 0, 1), seams
    if direction != "vertical":
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    H, W = img.shape[:2]
    if not 0 <= n < W:
        raise ValueError(f"cannot remove {n} seams from an image {W} pixels wide")

    rows = np.arange(H)
    cur = img.copy()
    # HINT: after each removal the columns shift, so "column 5 now" may have
    # been "column 6" before. col_map remembers, for every pixel still in the
    # image, which column it had in the ORIGINAL. We remove the same seams
    # from it as from the image, so it always stays aligned with the image.
    col_map = np.tile(np.arange(W), (H, 1))
    seams = np.zeros((n, H), dtype=int)

    for k in range(n):
        energy = energy_fn(cur)                     # recompute: neighbours changed
        seam = find_seam(cumulative_energy(energy))
        seams[k] = col_map[rows, seam]              # translate to original columns
        cur = remove_seam(cur, seam)                # all channels at once
        col_map = remove_seam(col_map, seam)
    return cur, seams


# --------------------------------------------------------------------------
# Drawing and the simple resizing options (for the 3.2 comparison)
# --------------------------------------------------------------------------
def draw_seams(img, seams, color=(1.0, 0.0, 0.0), direction="vertical"):
    """Return a colour copy of img with every seam painted in `color`.
    seams are in the ORIGINAL image's coordinates (as returned by carve)."""
    out = np.asarray(img, dtype=np.float64)
    if out.ndim == 2:
        out = np.dstack([out] * 3)
    out = out.copy()
    if direction == "horizontal":
        out = np.swapaxes(out, 0, 1).copy()         # work in the transposed frame
    H = out.shape[0]
    rows = np.broadcast_to(np.arange(H), seams.shape)
    out[rows, seams] = color                        # paint all seams at once
    return np.swapaxes(out, 0, 1) if direction == "horizontal" else out


def seams_to_mask(shape_hw, seams, direction="vertical"):
    """Boolean (H, W) map: True where a pixel was removed by some seam."""
    H, W = shape_hw
    if direction == "horizontal":
        return seams_to_mask((W, H), seams, "vertical").T
    mask = np.zeros((H, W), dtype=bool)
    mask[np.broadcast_to(np.arange(H), seams.shape), seams] = True
    return mask


def center_crop(img, new_width):
    """Keep the middle `new_width` columns (the 'just crop' option)."""
    W = img.shape[1]
    x0 = (W - new_width) // 2
    return img[:, x0:x0 + new_width]


# --------------------------------------------------------------------------
# BONUS: object removal and seam insertion
# --------------------------------------------------------------------------
def remove_object(img, mask, energy_fn):
    """Remove the masked object by carving seams through it.
    Returns (reduced_image, k) where k = number of seams removed.

    HINT: giving the masked pixels a hugely NEGATIVE energy makes them the
    cheapest pixels in the image, so the cheapest seam is pulled through the
    object. The mask loses the same seam as the image, and we repeat until
    no masked pixel is left.
    """
    cur = np.asarray(img, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=bool).copy()
    k = 0
    while m.any():
        if cur.shape[1] <= 2:
            raise RuntimeError("the mask is too wide: the whole image would vanish")
        energy = np.asarray(energy_fn(cur), dtype=np.float64)   # float64 as required
        energy[m] = BIG_NEG
        seam = find_seam(cumulative_energy(energy))
        cur = remove_seam(cur, seam)
        m = remove_seam(m, seam)
        k += 1
    return cur, k


def enlarge(img, k, energy_fn):
    """Widen img by k columns by DUPLICATING k seams (Avidan & Shamir 4.3).
    Returns (enlarged_image, marks); marks is a bool (H, W+k) map of the
    inserted pixels (both copies of every seam pixel), for drawing.

    HINT: why find ALL k seams on a copy first, instead of duplicating the
    cheapest seam k times? After duplicating the cheapest seam once, the copy
    sits right next to the original and is STILL the cheapest, so it would be
    picked again and again and the same strip would be stretched k times
    (an ugly smear). Removing k seams from a copy finds k DIFFERENT seams
    spread over the image; we then duplicate each of them once.
    """
    img = np.asarray(img, dtype=np.float64)
    H, W = img.shape[:2]
    rows = np.arange(H)

    # Step 1: find k seams on a copy. `carve` already reports them in the
    # coordinates of `img` (through its column map), which is what we need.
    _, seams = carve(img, k, energy_fn)
    is_seam = seams_to_mask((H, W), seams)                      # (H, W) bool

    # Step 2: build the two copies of every pixel.
    a = img if img.ndim == 3 else img[..., None]                # always (H, W, C)
    left = np.concatenate([a[:, :1], a[:, :-1]], axis=1)        # left neighbour (border: itself)
    right = np.concatenate([a[:, 1:], a[:, -1:]], axis=1)       # right neighbour (border: itself)
    first = np.where(is_seam[..., None], (a + left) / 2, a)     # seam pixel: average with left
    second = (a + right) / 2                                    # seam pixel: average with right

    # Step 3: write each pixel once, each seam pixel twice, row by row.
    # HINT: stack both copies -> shape (H, W, 2, C). `keep` says which copies
    # to write: the first always, the second only on seam pixels. Boolean
    # indexing reads in row-major order (row, column, copy), which is exactly
    # the left-to-right order of the widened row, and every row ends with
    # exactly W + k pixels because every row has exactly k seam pixels.
    stacked = np.stack([first, second], axis=2)
    keep = np.stack([np.ones((H, W), bool), is_seam], axis=2)
    out = stacked[keep].reshape(H, W + k, -1)

    both = np.stack([is_seam, is_seam], axis=2)
    marks = both[keep].reshape(H, W + k)
    return (out[..., 0] if img.ndim == 2 else out), marks
