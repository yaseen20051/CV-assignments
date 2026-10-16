"""
cvlib/filters.py  -  filter builders (Gaussian, finite differences, DoG).
Pure functions: they only build arrays or call the convolution.
"""
import numpy as np
from .convolution import conv_sep, convolve

DX = np.array([[1.0, 0.0, -1.0]])             # finite difference along x (1x3)
DY = DX.T                                     # finite difference along y (3x1)


def gaussian_1d(sigma, size=None):
    """1D Gaussian weights summing to 1. Default size = 2*ceil(3*sigma)+1."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1    # about 6*sigma + 1, always odd
    if size % 2 == 0:
        raise ValueError("size must be odd")
    r = size // 2
    x = np.arange(-r, r + 1)                  # positions -r ... r
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()                        # normalize: image brightness is kept


def gaussian_kernel(sigma, size=None):
    """2D Gaussian = outer product of the 1D one (that is why it is separable)."""
    g = gaussian_1d(sigma, size)
    return np.outer(g, g)


def gaussian_blur(img, sigma, size=None, pad_mode="reflect"):
    """Blur a gray or colour image with a Gaussian. Reflect padding by default
    (Part 2+), because zeros would darken the border."""
    return convolve(img, gaussian_kernel(sigma, size), pad_mode)


def dog_filters(sigma, size=None):
    """Derivative of Gaussian filters (dog_x, dog_y): filter the Gaussian kernel
    itself with Dx and Dy (zero padding, as the assignment describes)."""
    G = gaussian_kernel(sigma, size)
    return conv_sep(G, DX), conv_sep(G, DY)
