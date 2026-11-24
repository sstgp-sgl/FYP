# Selectable reconstruction methods

Three lazy-loaded backends share subject preparation and ordinary RGB PLY
export. No model is imported by `--help` or `--list-methods`.

- `vggt`: the existing Zero123++ → VGGT path. Optionally reuse a generated
  `views.json` directory to skip view generation.
- `triposr`: single-image TripoSR, 256 grid by default, density threshold 25.
  Supports predicted RGB on sampled points and a vertex-colored triangle mesh.
- `hunyuan`: Hunyuan3D-2mini standard fp16 shape model, 30 steps and 256
  extraction resolution by default. Texture generation is separate.

## Input and subject selection

Keep the existing SAM2 point/box selection. Use the original image and its
selected `mask.png`. Repeat the command with a different mask and output
folder for each subject in a multi-subject image. The reusable Python API and CLI can also be used independently. The P6
`web/storyboard/` page now connects this subject-selection step to the same
three reconstruction backends; see [web setup](storyboard-web.md).

A white-background `subject.png` is **not** a transparent foreground. Shared
preparation explicitly puts the mask into alpha. Fully opaque input without a
mask, empty masks, and size mismatches fail. No RGB white-point removal is used
in this preparation, so white foreground parts are retained.

## Setup

Use an isolated environment and install the dependencies of the backend you
select; do not install all three model stacks into a working environment.
`requirements-reconstruction.txt` lists shared CPU/extraction dependencies,
not a complete replacement for each model's official requirements.

- [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- VGGT / SAM2 / Zero123++: existing project setup.

Model code and weights stay outside this repository. No weights are vendored.
The HPC test environment uses a TripoSR checkout with unused rembg/torchmcubes
imports made lazy and an optional local DINO config; surface extraction uses
scikit-image. The geometry-only Hunyuan checkout omits the unused
postprocessors import to avoid installing pymeshlab. These are dependency
adjustments, not changes to neural network weights. Standard official installs
can use the adapters without these adjustments.

Copy `configs/reconstruction.example.yaml` to
`configs/reconstruction.local.yaml` and set your own model paths. That local
file is ignored by Git. Paths may be absolute or `~` paths. The existing VGGT
pipeline config is still resolved with the project's established rules.

```bash
python scripts/reconstruct_asset.py --list-methods
python scripts/reconstruct_asset.py --method triposr --color \
  --image /path/to/photo.jpg --mask /path/to/selected/mask.png \
  --config configs/reconstruction.local.yaml --out outputs/subject-triposr
python scripts/reconstruct_asset.py --method hunyuan \
  --image /path/to/photo.jpg --mask /path/to/selected/mask.png \
  --config configs/reconstruction.local.yaml --out outputs/subject-hunyuan
python scripts/reconstruct_asset.py --method vggt \
  --image /path/to/photo.jpg --mask /path/to/selected/mask.png \
  --config configs/reconstruction.local.yaml --out outputs/subject-vggt
```

On HPC run model inference in a Slurm GPU allocation. The small CPU tests do
not download/load models:

```bash
python -m pytest -q tests/test_reconstruction_methods.py
```

Python callers use
`reconstruct_asset(method, image_path, mask_path, out_dir, options)` from
`src.reconstruction_methods`. Register another method in `METHODS` and
`BACKEND_KEYS`, and add its module with
`reconstruct(image_rgba, out_path, config) -> (points, colors_or_none, details)`.

## Output contract and limits

Each run requires a new directory and saves `foreground.png`, `pointcloud.ply`
and `run.json` (method, input hashes, effective options, source hashes,
normalization and timings). Failures are recorded and raised; there is no
success fallback to another backend. The PLY is read back to check XYZ/RGB.

Defaults: 120,000 points, gray RGB=180, bbox-centered uniform scaling to longest
side 2. Native axis conventions are retained: rotate separately when comparing
models. With `gray: false` or CLI `--color`, VGGT retains observed colors and
TripoSR queries predicted RGB in native model coordinates before normalization.
Hunyuan remains geometry-only and returns gray. The shared default remains
`gray: true` for reproducibility; the example TripoSR method overrides it to false.

TripoSR also saves `mesh.ply` with triangle faces and vertex RGB, plus an
offline-capable `viewer.html`. The mesh uses exactly the point cloud's output
normalization; `mesh_native.ply` retains the intermediate native coordinates.
No UV atlas or image texture is baked. Back/hidden-surface colors are model
predictions, not observations. Old gray files are unchanged and need a new run
to obtain predicted RGB.

Serve the output folder locally with `python -m http.server 8001 --bind 127.0.0.1`
and open `viewer.html`, or open the HTML directly and choose `mesh.ply` using
the file input. The viewer uses local WebGL2 without a CDN, supports rotation,
zoom, a gray comparison, optional surface lighting, and current-view PNG export.
It is a TripoSR Z-up viewer; display converts the native axes to Y-up without
rewriting downloaded geometry. The website's artifact ZIP includes mesh,
point cloud and viewer after successful reconstruction. GPU models are still
required for inference, but browsing an exported asset does not need HPC.

TripoSR/Hunyuan points are triangle-area samples. VGGT samples existing
observations; when fewer observations exist, repeated samples are explicitly
reported. Equal point count does not make these sampling processes identical,
and normalization does not preserve physical size. The VGGT adapter does not
attribute cached views to the supplied image: ensure reused views belong to
the selected subject. View files are referenced in the backend result.

Old `vggt_headless.py`, web routes, experimental comparison scripts, and
storyboard commands remain separate. The legacy `app.py` reconstruction page retains its VGGT behavior; backend
selection is available in the separate P6 storyboard page.
