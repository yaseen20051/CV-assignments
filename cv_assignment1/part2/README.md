# Part 2 - Frequencies, Pyramids and Blending

Everything for Part 2 lives in this folder. It imports only `cvlib` (our own separable
convolution, reflect padding) and never edits it. Install: `pip install -r requirements.txt`.

```
part2/
├── config.py          paths + sys.path setup (import it first)
├── hybrid.py          2.1  make_hybrid, low/high_pass, log_fft, to_gray
├── aliasing.py        2.2  subsample, antialiased_subsample, zone_plate, stripes
├── pyramids.py        2.3  downsample, upsample, gaussian_pyramid, laplacian_pyramid, reconstruct
├── blending.py        2.4  multires_blend, vertical/circle/ellipse masks, direct_paste, feather_blend
├── align_images.py    starter-pack aligner (rotate + scale + shift from two matching points)
├── viz.py             every figure used in the report
├── synthetic.py       generated demo images (fallback when a file is missing)
├── images/            starter-pack images for Part 2 (+ facade.png)
├── test_part2.py      python part2/test_part2.py
├── requirements.txt
└── part2_notebook.ipynb   the notebook: edit the Settings cell, then Run All
```

## How to use
1. Open `part2_notebook.ipynb`, edit the **Settings** cell (file names, alignment points, sigmas), Run All.
   Your own images go in `part2/images/` (or `cv_assignment1/images/`). 
2. For a hybrid pair of your own: `show_with_grid(image)` -> read two matching points -> put them in
   `HYBRID_PAIRS` (`pts_low`, `pts_high`, as `(x, y)`).
3. Figures are saved to `part2/outputs/figures/`.

## Design notes
- Hybrid: `blur(A, s_low) + (B - blur(B, s_high))`, B warped onto A first. Clipped to [0, 1] for display only.
- Images used with alignment points are loaded at full size (`max_side` huge): `cvlib.load_image`
  would otherwise shrink them and the point coordinates would no longer match.
- Upsample = zero insertion + bilinear kernel `[1 2 1]/4` (x4), exact on flat/linear areas, so
  `reconstruct(laplacian_pyramid(img))` returns the image to ~1e-15.
- Pyramid sizes use `ceil(n/2)`, so odd image sizes round-trip.
- Blend: Laplacian pyramids of A and B, Gaussian pyramid of the mask, blend per level, collapse.
