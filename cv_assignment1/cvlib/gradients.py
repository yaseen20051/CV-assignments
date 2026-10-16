"""
cvlib/gradients.py  -  gradient magnitude and edge thresholding (Sections 1.2/1.3).
Part 3 (seam carving) uses gradient_magnitude as its energy function.
All functions take a 2D (grayscale) image.
"""
import numpy as np
from .convolution import conv_sep
from .filters import DX, DY


def gradient_maps(img, kx=DX, ky=DY, pad_mode="constant"):
    """Return (dx, dy, magnitude).
    Pass kx, ky = dog_filters(sigma) for the Derivative of Gaussian version."""
    dx = conv_sep(img, kx, pad_mode)          # derivative along x
    dy = conv_sep(img, ky, pad_mode)          # derivative along y
    return dx, dy, np.sqrt(dx ** 2 + dy ** 2)


def gradient_magnitude(img, kx=DX, ky=DY, pad_mode="constant"):
    """Just the magnitude sqrt(dx^2 + dy^2)."""
    return gradient_maps(img, kx, ky, pad_mode)[2]


def threshold_edges(magnitude, percentile=None, value=None):
    """Binarize a magnitude map. Give EITHER a percentile (90 = keep the
    strongest 10% of pixels) OR a fixed value. Returns (edges_bool, threshold)."""
    if (percentile is None) == (value is None):
        raise ValueError("give exactly one of percentile or value")
    t = np.percentile(magnitude, percentile) if value is None else value
    return magnitude > t, t
