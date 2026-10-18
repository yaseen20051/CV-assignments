"""
part2/synthetic.py  -  tiny generated images so the notebook and the tests run
even before you have put real photos in images/. NOT for the report: use real pairs.
"""
import numpy as np


def _grid(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return y / h, x / w


def blobs(h=320, w=320):
    """Smooth, low-frequency colour image (soft shapes): a good 'low-pass' source."""
    y, x = _grid(h, w)
    r = 0.5 + 0.5 * np.sin(6 * x + 2 * y)
    g = np.exp(-((x - .35) ** 2 + (y - .45) ** 2) / .05)
    b = np.exp(-((x - .7) ** 2 + (y - .6) ** 2) / .08)
    return np.clip(np.dstack([0.6 * r + 0.3 * g, 0.8 * g + 0.2 * r, b]), 0, 1)


def texture(h=320, w=320, seed=0):
    """Sharp, high-frequency colour image (stripes, checks, edges): a good 'high-pass' source."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    checks = ((x // 6 + y // 6) % 2).astype(float)
    stripes = 0.5 + 0.5 * np.sin(2 * np.pi * (x + y) / 9)
    ring = (np.hypot(x - w / 2, y - h / 2) % 24 < 3).astype(float)
    noise = rng.random((h, w))
    img = np.dstack([checks, stripes, 0.5 * ring + 0.5 * noise])
    return np.clip(img, 0, 1)


def sunset(h=320, w=320):
    """Smooth vertical colour gradient (blend demo, image A)."""
    y, _ = _grid(h, w)
    return np.clip(np.dstack([1.0 - 0.3 * y, 0.45 + 0.2 * y, 0.2 + 0.6 * y]), 0, 1)


def bars(h=320, w=320):
    """Strong textured colour image (blend demo, image B)."""
    y, x = np.mgrid[0:h, 0:w]
    s = 0.5 + 0.5 * np.sin(2 * np.pi * x / 24)
    c = ((y // 20) % 2).astype(float)
    return np.clip(np.dstack([0.2 + 0.7 * s, 0.7 * c + 0.2, 0.9 - 0.6 * s]), 0, 1)


def scene(h=320, w=320):
    """Structure at several scales (big blobs + medium rings + fine texture): shows
    pyramids well."""
    base = blobs(h, w)
    y, x = np.mgrid[0:h, 0:w]
    rings = 0.5 + 0.5 * np.cos(np.hypot(x - w * .3, y - h * .5) / 5)
    fine = ((x // 3 + y // 3) % 2).astype(float)
    wx = np.clip((x / w - 0.5) * 3 + 0.5, 0, 1)[..., None]          # fine texture only on the right
    img = 0.75 * base + 0.25 * rings[..., None]
    return np.clip((1 - wx) * img + wx * (0.7 * img + 0.3 * fine[..., None]), 0, 1)


DEMOS = {"blobs": blobs, "texture": texture, "sunset": sunset, "bars": bars, "scene": scene}
