"""
part2/hybrid.py  -  Section 2.1 hybrid images.

    low  = gaussian_blur(image_a, sigma_low)               (keeps low frequencies)
    high = image_b - gaussian_blur(image_b, sigma_high)    (keeps high frequencies)
    hybrid = low + high

Pure functions. Images are float64 in [0, 1]; every blur is OUR separable
convolution from cvlib, with reflect padding (zeros would darken the border).
"""
import config  # noqa: F401  (puts cvlib on sys.path, so this module also works on its own)
import numpy as np
from cvlib import gaussian_blur


def to_gray(img):
    """(H, W, 3) -> (H, W) with the usual luminance weights; 2D is returned as is."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return img
    return img @ np.array([0.299, 0.587, 0.114])


def low_pass(img, sigma):
    return gaussian_blur(img, sigma)                  # reflect padding by default


def high_pass(img, sigma):
    """The image minus its blur. Values can be negative (show with signed_for_show)."""
    return img - gaussian_blur(img, sigma)


def make_hybrid(im_low, im_high, sigma_low, sigma_high, clip=True):
    """Return a dict with everything the report figure needs.

    im_low   : image whose LOW frequencies are kept (seen from far away)
    im_high  : image whose HIGH frequencies are kept (seen up close)
    sigma_*  : the two cutoff controls. Bigger sigma_low = blurrier low image;
               bigger sigma_high = LOWER cutoff for the high-pass image.
    """
    if im_low.shape != im_high.shape:
        raise ValueError(f"images must be aligned to the same shape, got "
                         f"{im_low.shape} and {im_high.shape}")
    low = low_pass(im_low, sigma_low)
    high = high_pass(im_high, sigma_high)
    hybrid = low + high
    return {
        "im_low": im_low, "im_high": im_high,
        "low": low, "high": high,
        "hybrid": np.clip(hybrid, 0, 1) if clip else hybrid,
        "sigma_low": sigma_low, "sigma_high": sigma_high,
    }


def log_fft(gray):
    """Log magnitude of the centred 2D Fourier transform (the handout's formula).
    The tiny epsilon only avoids log(0) for perfectly empty frequencies."""
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


def hybrid_spectra(result):
    """Log-FFT of the five grayscale images of a make_hybrid() result, in the order
    (low-source, high-source, low-passed, high-passed, hybrid)."""
    keys = ["im_low", "im_high", "low", "high", "hybrid"]
    return [log_fft(to_gray(result[k])) for k in keys]


def center_crop_resize(img, shape):
    """Fallback aligner for quick tests ONLY: centre-crop to the target aspect
    ratio then resize. The assignment requires align_images.py from the starter
    pack for real pairs, because it aligns by eye points."""
    from PIL import Image
    h, w = img.shape[:2]
    th, tw = shape[:2]
    scale = max(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    im = im.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(im, dtype=np.float64) / 255.0
    top, left = (nh - th) // 2, (nw - tw) // 2
    return arr[top:top + th, left:left + tw]
