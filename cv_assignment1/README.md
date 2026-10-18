# CV Assignment 1 - project layout

```
cv_assignment/
├── cvlib/        SHARED library (Parts 1, 2, 3 all import from it)
├── tests/        tests for cvlib  -> python tests/test_cvlib.py
├── images/       shared input images
├── outputs/      results, one sub-folder per part (part1/, part2/, part3/)
├── part1/        Part 1 app (menus, folders, figures)  -> python part1/main.py
├── part2/        Part 2 app (to do)
└── part3/        Part 3 app (to do)
```

## Using the library in Part 2 / Part 3
Start your `main.py` with this so `cvlib` can be found:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cvlib import convolve, gaussian_blur, gaussian_kernel, gradient_magnitude
from cvlib import load_image, save_image, signed_for_show
```

Common calls:
- `convolve(img, K, pad_mode="reflect")`  separable conv, gray or colour image
- `gaussian_blur(img, sigma)`              reflect padding by default
- `gradient_magnitude(gray_img)`           the energy for seam carving
- `signed_for_show(x)`                     show negative maps with 0 = mid-gray

## Library rules
1. Pure functions: no `input()`, no `print()`, no folders, no globals.
2. Images are float64 in [0, 1]; (H, W) or (H, W, 3). Never edited in place.
3. Conv signature is always `fn(img, K, pad_mode="constant")`; use `"reflect"` from Part 2 on.
4. Filter sides must be odd (raises `ValueError` otherwise).
5. Changed something in `cvlib`? Run `python tests/test_cvlib.py` before committing.
6. New shared helpers (pyramids, blending, seams) go in NEW files inside `cvlib/`
   and get exported in `cvlib/__init__.py`. Menus and file handling stay in `partN/`.

## Part 3 (seam carving)
Library: `cvlib/seam_carving.py`  (energy, DP, remove/carve, draw, bonus).
App: `python part3/main.py`. Everything it writes goes to `outputs/part3/`:
```
outputs/part3/
├── figures/<photo>/   energy, seams in red, comparison figures (3_1_*, 3_2_*)
├── carved/<photo>/    each result as its own image
└── bonus/<photo>/     mask, object removed, width restored (bonus_*)
```
Part 2 must write to `outputs/part2/` the same way, so results never mix.
Optional Bonus mask files: `images/masks/<photo name>_mask.png` (white = object).
Tests: `python tests/test_seam_carving.py`
