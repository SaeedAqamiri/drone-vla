# sat2aerial

Satellite → aerial reference synthesis and map-matching bake-off for GPS-denied drone navigation.

The point is not to prove that generated images "look good". The point is to measure whether a generated
aerial reference is **metrically useful**: does it still match the source satellite tile (no hallucinated
geometry), and can a real drone frame be matched to it?

## Why this exists

The original idea — translate satellite imagery into aerial imagery, then match — moves the domain gap
instead of removing it. Generative models are optimized for perceptual realism, not survey-grade geometry,
so the dangerous failure mode is a confident but wrong match. This harness makes that failure mode visible:

- `consistency inliers`: generated image matched back to its source satellite tile. Low values mean the
  model invented or moved structures.
- `inliers` / `inlier_ratio`: generated reference matched to a **real** drone frame. Compared against the
  raw-satellite baseline, this is the number that matters.
- `aligned_edge_iou`: edge agreement after warping the generated image back to the satellite plane.
- `lab_mean_dist` / `hist_corr`: appearance statistics only; never use them alone.

## Quick start (offline, no model download)

```bash
env_flight/bin/python -m sat2aerial selfcheck --out /tmp/sat2aerial_selfcheck
env_flight/bin/python -m sat2aerial demo --out outputs/sat2aerial/demo --sites 4 --clean
```

`demo` creates procedural city scenes, runs four generation variants with the `dummy` backend, computes all
metrics, and writes `report.md`, `results.csv`, `summary.json` and per-site montages. The dummy backend is a
procedural warp — it validates the harness, not FLUX.

## Dataset layout

Any of these work:

```text
data/pairs/<site_id>/satellite.png
data/pairs/<site_id>/aerial_00.png
data/pairs/<site_id>/aerial_01.png
```

```text
data/pairs/manifest.csv
# site,satellite,aerial
# site_01,images/sat1.png,images/air1.png;images/air2.png
```

```text
data/pairs/satellite/<stem>.png
data/pairs/aerial/<stem>_00.png
```

The files must be **co-located pairs**: the satellite tile and the real drone frame(s) show the same place.
Public candidates: University-1652, SUES-200, DenseUAV, UAV-VisLoc.

## Running the bake-off

```bash
# Local FLUX 3 Image Klein through ComfyUI (works on 8GB; see below)
env_flight/bin/python -m sat2aerial bakeoff \
  --pairs data/pairs --out outputs/sat2aerial/bakeoff_comfy \
  --backend comfy --variants nadir,oblique30,nadir_style,oblique30_style \
  --style-refs 1 --seeds 3 --size 768 --max-pairs 20

# Remote FLUX 3 Image Klein
export BFL_API_KEY=...
env_flight/bin/python -m sat2aerial bakeoff \
  --pairs data/pairs --out outputs/sat2aerial/bakeoff_bfl \
  --backend bfl --variants nadir,oblique30,nadir_style,oblique30_style \
  --style-refs 1 --seeds 3 --size 768 --max-pairs 20
```

Use `--seeds 3` (or more). Generation is seed-sensitive: on the same satellite tile one seed preserved the
city layout and another invented a different one, so a single seed is not a valid comparison.

The style reference for a site is always taken from **other** sites, so the target drone frame never leaks
into the generation.

## Backends

| backend | needs | notes |
|---|---|---|
| `dummy` | nothing | procedural warp; smoke test only |
| `comfy` | ComfyUI + int8 files | **working 8GB path**; ~60-85 s / 768px image on RTX 4060 |
| `bfl` | `BFL_API_KEY` | FLUX 3 Image Klein API, base64 references, ~$0.009/image |
| `flux3` | gated weights + recent diffusers | local fp8r/bf16, needs >8GB VRAM in practice |

### Local FLUX 3 Klein on 8GB (ComfyUI)

The int8 checkpoints live in the separate gated repo
[`flux-3-image-klein-comfy`](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy). Accept
its license, then:

```bash
hf download black-forest-labs/flux-3-image-klein-comfy \
  diffusion_models/flux3-image-klein-int8-convrot.safetensors \
  text_encoders/qwen3vl_4b_int8_convrot.safetensors \
  vae/flux2-vae.safetensors \
  --local-dir ComfyUI/models

cd ComfyUI
../env_comfy/bin/python main.py --lowvram --disable-auto-launch \
  --listen 127.0.0.1 --port 8188 --preview-method none
```

Then run any command with `--backend comfy`. The backend stages the satellite tile and style references
into `ComfyUI/input/`, builds a `ReferenceLatent` graph, polls `/history`, and copies the result.

Notes:

- The first generation loads ~10GB of weights from disk and takes longer; later calls reuse them.
- The prompt upsampler (a 9B model) is not used. Prompts from `build_prompt` are already detailed; BFL
  recommends the upsampler and quality would likely improve on a larger GPU.
- `flux3` (Diffusers) is kept for larger GPUs. The FP8 transformer is 6.27GB, but Qwen3-VL is 8.9GB and
  HF reports ~16.3GB peak at 1024px without offload.

## Metrics and success rule

A match succeeds when `inliers >= 15`, `inlier_ratio >= 0.15`, **and** `coverage >= 0.03` (configurable).
Coverage is the convex-hull area of the inlier points; without it, a line of edge matches along a road or
river is accepted as a confident false positive. Matching uses SIFT on CLAHE-normalized Canny edges by
default, FLANN + ratio test, then RANSAC homography.

## Pilot results

Synthetic city scenes, FLUX 3 Klein int8, 2 sites × 2 variants × 2 seeds:

- `nadir` without a style reference: success 0.25, mean 13 inliers, seed range 7–22.
- `nadir_style`: success 1.00, mean 142 inliers, seed range 100–190.
- Raw satellite baseline: 175 inliers, success 1.00.

Real University-1652, 5 building pairs (satellite 512×512, drone view is oblique):

| variant | success | inliers | consistency inliers | edge IoU | SSIM |
|---|---|---|---|---|---|
| baseline satellite | 0.00 | 9.4 | - | - | - |
| `nadir` (no style) | 0.00 | 7.4 | 437.8 | 0.885 | 0.879 |
| `oblique30_style` | 0.00 | 8.6 | 141.0 | 0.344 | 0.292 |

Neither variant beats the raw satellite baseline, and none succeeds. The generated image stays **top-down**:
the oblique prompt does not create the 3D viewpoint change, and the style reference makes the geometry less
faithful without improving matchability.

Conclusion so far: 2D satellite→aerial translation can preserve layout and produce a plausible aerial look
(especially with a style reference), but it does not create the parallax and occlusion of an oblique drone
view. For oblique frames, use a DSM/3D representation and render, or restrict the mission to near-nadir
flight. For near-nadir frames, a style reference is essential; without it the model often re-invents the
layout.

Interpretation:

- Generated model beats the satellite baseline on `inliers` → the synthesis is genuinely helping matching.
- High `inliers` but low `consistency inliers` → the model is generating plausible imagery of the wrong place.
- High `hist_corr` and low everything else → pretty but useless.

## Tests

```bash
env_flight/bin/python -m unittest discover -s tests -v
```
