"""
part2/aliasing.py  -  Section 2.2 aliasing.

Keeping every k-th pixel is sampling. If the image still has frequencies above
the new Nyquist limit (1 / (2k) cycles per pixel) they fold back as false low
frequencies: that is aliasing. Blurring first (an anti-aliasing low-pass) removes
them before they can fold.
"""
import config  # noqa: F401  (puts cvlib on sys.path, so this module also works on its own)
import numpy as np
from cvlib import gaussian_blur


def subsample(img, k):
    """Naive: keep every k-th pixel in both directions. Aliases."""
    return np.asarray(img)[::k, ::k]


def antialiased_subsample(img, k, sigma=None):
    """Blur first, then subsample. Rule of thumb: sigma ~ k / 2 puts the cutoff
    near the new Nyquist limit (try other values and look at the spectra)."""
    sigma = k / 2 if sigma is None else sigma
    return subsample(gaussian_blur(img, sigma), k)


def zone_plate(n=512, strength=0.5):
    """Chirp image cos(a * r^2): its local frequency grows with the distance from
    the centre, so it sweeps through ALL frequencies and makes aliasing obvious
    (rings that should get finer instead reappear as coarse moire rings)."""
    y, x = np.mgrid[-n // 2:n // 2, -n // 2:n // 2].astype(np.float64)
    r2 = x ** 2 + y ** 2
    a = strength * np.pi / n                         # reaches Nyquist near the edge
    return 0.5 + 0.5 * np.cos(a * r2)


def stripes(n=256, period=3.0, angle_deg=0.0):
    """Sinusoidal stripes with a known frequency 1/period (cycles per pixel).
    Subsampling with k > period / 2 makes them alias to a new, lower frequency."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    t = np.deg2rad(angle_deg)
    u = x * np.cos(t) + y * np.sin(t)
    return 0.5 + 0.5 * np.sin(2 * np.pi * u / period)


def aliasing_comparison(img, ks=(2, 4, 8), sigma=None):
    """For each k return (k, naive, antialiased), all as arrays. Use viz.py to show."""
    return [(k, subsample(img, k), antialiased_subsample(img, k, sigma)) for k in ks]


def outer_ring_std(img, frac=0.35):
    """Numeric aid for 'the smallest sigma with no false circles' on a chirp-like image
    such as rings.png: the outer rings are too fine to survive a k-fold shrink, so a
    correctly anti-aliased result is flat gray out there, and any leftover contrast
    (std) is false circles. Confirm the chosen sigma by eye as well."""
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(y - h / 2, x - w / 2)
    return float(img[r > frac * min(h, w)].std())
