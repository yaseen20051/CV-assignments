"""
cvlib/convolution.py  -  2D convolution written from scratch (Section 1.1).

SHARED LIBRARY RULES (read before editing):
  * Pure functions only: no input(), no print(), no file or folder access.
  * Images are float64 in [0, 1], shape (H, W) gray or (H, W, 3) colour.
  * Inputs are never modified in place; every function returns a new array.
  * Every conv function has the same signature:  fn(img, K, pad_mode="constant")
        pad_mode="constant" -> zero padding   (Part 1)
        pad_mode="reflect"  -> mirror padding (Part 2 and 3)

Which one should I call?
  convolve(img, K, pad_mode)  <- use this in Parts 2 and 3 (fast, handles colour)
  conv_loop / conv_vec / conv_sep <- the three versions, for tests and timing
"""
import numpy as np

# Singular values below SVD_TOL * (largest one) count as zero. Needed because
# computers never return an exact 0 (rounding error is about 1e-16).
SVD_TOL = 1e-10


def _prepare(img, K, pad_mode):
    """Validate inputs, flip the filter, pad the image. Shared by loop and vec."""
    img = np.asarray(img, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2D (one channel); use conv_color for colour")
    if K.ndim != 2:
        raise ValueError("filter must be 2D")

    kh, kw = K.shape                          # the filter does NOT have to be square
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"filter sides must be odd, got {kh}x{kw}")

    # True convolution = correlation with the filter flipped on both axes.
    K_flipped = K[::-1, ::-1]

    # Pad kh//2 rows (top/bottom) and kw//2 columns (left/right) so the output
    # has exactly the same size as the input.
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode=pad_mode)
    return img, K_flipped, padded


def conv_loop(img, K, pad_mode="constant"):
    """Version 1: loops over every pixel AND every filter entry (slowest)."""
    img, Kf, padded = _prepare(img, K, pad_mode)
    H, W = img.shape
    kh, kw = Kf.shape
    out = np.zeros((H, W))
    for i in range(H):                        # every output row
        for j in range(W):                    # every output column
            s = 0.0
            for u in range(kh):               # every filter row
                for v in range(kw):           # every filter column
                    s += Kf[u, v] * padded[i + u, j + v]
            out[i, j] = s
    return out


def conv_vec(img, K, pad_mode="constant"):
    """Version 2: loops over filter entries only; whole image per step."""
    img, Kf, padded = _prepare(img, K, pad_mode)
    H, W = img.shape
    kh, kw = Kf.shape
    out = np.zeros((H, W))
    for u in range(kh):
        for v in range(kw):
            # padded[u:u+H, v:v+W] is the image shifted by (u, v). One filter
            # number times it adds that entry's share to ALL pixels at once.
            out += Kf[u, v] * padded[u:u + H, v:v + W]
    return out


def separate(K, tol=SVD_TOL):
    """Split a rank-1 filter into (column kh x 1, row 1 x kw); None if rank > 1."""
    K = np.asarray(K, dtype=np.float64)
    U, S, Vt = np.linalg.svd(K)               # singular values, biggest first
    # Rank 1 <=> every singular value after the first is (almost) zero.
    # A 1xN or Nx1 filter has a single singular value, so it is always rank 1.
    if len(S) > 1 and S[1] > tol * S[0]:
        return None
    col = (U[:, 0] * np.sqrt(S[0])).reshape(-1, 1)
    row = (Vt[0] * np.sqrt(S[0])).reshape(1, -1)
    return col, row                           # outer(col, row) == K


def conv_sep(img, K, pad_mode="constant"):
    """Version 3: column pass then row pass if K is separable, else conv_vec."""
    parts = separate(K)
    if parts is None:
        return conv_vec(img, K, pad_mode)
    col, row = parts
    return conv_vec(conv_vec(img, col, pad_mode), row, pad_mode)


def conv_color(img, K, conv=conv_sep, pad_mode="constant"):
    """Run a grayscale convolution on each channel of an (H, W, C) image."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return conv(img, K, pad_mode)
    return np.dstack([conv(img[..., c], K, pad_mode) for c in range(img.shape[2])])


def convolve(img, K, pad_mode="constant"):
    """THE function to use in Parts 2 and 3: separable convolution, gray or colour."""
    return conv_color(img, K, conv=conv_sep, pad_mode=pad_mode)
