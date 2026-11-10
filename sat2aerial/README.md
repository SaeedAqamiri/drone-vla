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

## Where do the test images come from? (`flythrough` + `localize`)

Matching against satellite imagery needs the camera image and the reference map to show the
**same place at a comparable scale**. A generic 3D world (AirSim city, Gazebo `baylands`) does not
have that: there is no orthophoto of it, so any match score is measuring the sim2real gap, not
localization. Two commands fix that without building a digital twin.

### 1. Render a 2.5D flythrough over a real map

`orthofly` treats the orthophoto itself as the ground plane and renders a pinhole camera with a
real altitude, FOV, yaw, pitch and roll. The camera-ground map is a homography, so each frame is
one `warpPerspective` and the pose is exact by construction.

```bash
# real map
env_flight/bin/python -m sat2aerial flythrough \
  --ortho site.tif --out outputs/orthofly/site \
  --frames 120 --size 640 --alt 80 --hfov 60 \
  --path snake --length 400 --legs 3 --leg-spacing 80 \
  --ref-size 512 --ref-jitter 40 --pairs-dir outputs/orthofly/site/pairs

# nothing to download: procedural city, 1 km at 0.5 m/px
env_flight/bin/python -m sat2aerial flythrough --procedural \
  --out outputs/orthofly/demo --frames 30 --alt 130 --ref-jitter 40 \
  --pairs-dir outputs/orthofly/demo/pairs --clean
```

Outputs: `frames/*.png`, `poses.csv` (lat/lon, ENU metres, altitude, yaw, pitch, roll, GSD,
reference-crop centre), `meta.json`, `preview.png`, and optionally a

```
pairs/satellite/frame_0000.png   # reference window
pairs/aerial/frame_0000_00.png   # camera frame
```

pair layout that `bakeoff`/`localize` already understand.

Difficulty knobs: `--alt` (sets GSD and therefore the scale gap to the map), `--pitch`
(0 = nadir), `--yaw-jitter`, `--alt-ramp`, `--blur`, `--noise`, `--jpeg`, `--vignette`,
`--exposure-ms` (motion blur grows with `--speed`).

### 2. Score the metric error

`localize` matches every frame to its reference window, inverts the homography to a ground
position, and compares it to the truth. It reports the matcher's own gate
(`inliers >= 15`, `ratio >= 0.15`, `coverage >= 0.03`) alongside the metric error, because the
interesting number is **accepted matches that are wrong**.

```bash
env_flight/bin/python -m sat2aerial localize --run outputs/orthofly/demo --match-mode gray
# success 0.77 | gated median 0.254 m | gated p90 0.512 m | gross_passed 0 | inliers 28
```

Procedural demo (30 frames, 300 m corridor, 0.47 m/px camera, 0.5 m/px map, ±40 m reference
jitter) — a smoke test, not a claim about the algorithm:

| detector / mode | success | gated median | gated p90 | gross >5 m passed |
|---|---|---|---|---|
| sift / `gray` | 0.77 | 0.25 m | 0.51 m | **0** |
| orb / `gray` | 0.47 | 0.47 m | 25.4 m | 3 |
| sift / `edges` | 0.13 | 78.4 m | 156 m | 4 |

Two things this already showed:

* **`edges` mode is wrong for nadir map matching.** Canny keeps roads and block borders, so the
  inliers are collinear and the homography is unconstrained along the road → coverage ≈ 0 and
  errors of tens to hundreds of metres. Use `gray` for this task.
* **The gross failures are the gate's job.** Frames over feature-poor roofs produced 4–28 inliers
  with `coverage ≈ 0` and 20–250 m errors; every one of them fails the existing
  `inliers/ratio/coverage` rule, so `gross_passed == 0`. If you disable coverage, you get confident
  wrong fixes — which is exactly the failure mode `cons_inliers` was added for.

Known limits: the ground is a plane, so there is no parallax, no occlusion and no building side
wall. Keep the mission near-nadir here, and test oblique/3D behaviour in a Gazebo world built from
the same map (`scripts/make_geo_world.py`, next step).

### 2b. A real orthophoto in two minutes (NAIP, public domain)

UAV-VisLoc is the right benchmark but it is 16 GB+ behind a throttled link. For real-imagery
testing without any dataset, pull a window out of a NAIP COG on Microsoft Planetary Computer —
public domain, 0.3-0.6 m/px, no auth, and the COG means a 1.2 km crop is ~50 MB instead of 1.8 GB:

```bash
# any US bbox; this is downtown Boulder, CO
python - <<'PY'
import json, urllib.request, rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
bbox = (-105.276, 40.010, -105.262, 40.021)
item = json.load(urllib.request.urlopen(
    "https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip/items"
    f"?bbox={','.join(map(str, bbox))}&limit=1"))["features"][0]
href = item["assets"]["image"]["href"]
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
    with rasterio.open(href) as ds:
        tx = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        cx, cy = tx.transform((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
        win = from_bounds(cx-600, cy-600, cx+600, cy+600, ds.transform)
        data, transform, profile = ds.read(window=win), ds.window_transform(win), ds.profile.copy()
profile.update(width=data.shape[2], height=data.shape[1], transform=transform, compress="deflate")
with rasterio.open("data/naip/site.tif", "w", **profile) as dst:
    dst.write(data)
PY
```

Measured from Iran: ~370 KB/s, 4000×4000 px (1.2 km at 0.3 m/px) in 49 s. Then the whole stack runs
on it:

```bash
python -m sat2aerial flythrough --ortho data/naip/site.tif --out outputs/orthofly/naip \
  --frames 30 --size 512 --alt 100 --path snake --length 250 --legs 2 --leg-spacing 120 \
  --ref-jitter 40 --ref-size 768 --pairs-dir outputs/orthofly/naip/pairs
python -m sat2aerial localize --run outputs/orthofly/naip --match-mode gray
# success 1.00 | gated median 0.22 m | p90 0.33 m | gross_passed 0 | inliers 734

python -m sat2aerial closure --ortho data/naip/site.tif --out outputs/closure/naip \
  --path snake --length 200 --legs 2 --leg-spacing 100 --alt 100 --size 512
# 3/3 waypoints, 200/200 fixes, median error 0.18 m, final distance 3.3 m
```

A fixed bug worth knowing about: before 2026-11-06 `_local_affine` did not re-base its constant
term, so on **georeferenced** rasters every rendered frame was centred at `2 * pose`. `from_array`
maps and poses at the origin were unaffected, which is why the procedural tests passed. The
symptom is a position error that grows linearly with the distance from the map origin and a fitted
`scale` that looks fine. `test_render_is_centred_on_the_pose_for_a_geotiff` is the regression test.

### 3. Gazebo world from the same map (closed loop, real physics)

`orthofly` is 2.5D: no parallax, no occlusion, no closed loop. For that, generate a Gazebo
world whose ground plane *is* the orthophoto and whose origin is pinned to a WGS84 coordinate:

```bash
env_flight/bin/python scripts/make_geo_world.py \
  --ortho site.tif --name site1 --out outputs/geo \
  --extent-m 800 --tile-m 200 --tile-px 1024 \
  --center-lat 47.397 --center-lon 8.546 \
  --buildings data/site1_buildings.geojson     # or --osm

source outputs/geo/site1/env.sh
gz sim -r outputs/geo/site1/worlds/site1.sdf
```

World convention is **ENU**: `x = East`, `y = North`, `z = Up`, and `(0, 0)` is
`--center-lat/--center-lon`. That is the same frame as `poses.csv`, so a position from
`localize` can be fed straight into `VISION_POSITION_ESTIMATE` later.

Ground tiles are `--tile-m` squares at `--tile-px` pixels (so the ground GSD is
`tile_m / tile_px`) with a 2 px overlap to hide the filtering seam. Buildings are OSM/GeoJSON
footprints reduced to their convex hull and extruded to `height` (or `building:levels * 3 m`).

**Always verify a new world before flying it.** A flipped tile or a swapped axis still renders
fine but is fatal for matching:

```bash
env_flight/bin/python scripts/check_geo_world.py --run outputs/geo/site1 --alt 260
# {"inliers": 502, "coverage": 0.905, "scale": 0.99971, "rotation_deg": -0.004,
#  "offset_m": {"east": 0.29, "north": -0.28, "total": 0.41}, "passed": true}
```

It writes a static downward `probe_cam`, runs Gazebo headless, captures one frame, renders the
*expected* frame from the same GeoTIFF at the same pose, and matches the two. Scale must be 1,
rotation 0, offset a few metres at most. If it fails, `--flip-v` on `make_geo_world` is the usual
suspect (the default orientation is correct: OBJ `v=0` is south, and a spawn heading of 0° gives a
north-up image).

Measured on a synthetic 400 m world (0.195 m/px ground, 512×384 camera at 260 m): 502 inliers,
coverage 0.90, scale 0.99971, rotation −0.004°, 0.41 m offset. That is a georeferencing check, not
a localization benchmark — the reference is derived from the same raster by construction.

Known approximations: flat ground (no DEM relief), convex-hull prisms (no complex footprints),
single building height, and one PBR material per building. Textures scale with `--tile-px`, so keep
the ground GSD within ~2× of the camera GSD or matching degrades. On llvmpipe, prefer 200 m/1024 px
tiles (16 tiles = 16 MB of texture for an 800 m world) over one giant map.

### 3b. Live closed loop on real imagery (`run_live.py`)

The loop above is offline. To fly it, the autopilot has to be the real one, so the flight runs on
ArduPilot SITL in a **split simulation**:

| what | who |
|---|---|
| physics, EKF, arming checks, GUIDED | ArduPilot SITL (`--model quad`, its own built-in quad) |
| the picture | Gazebo, rendering *only* the orthophoto ground |
| the bridge | `gz_pose_bridge.py`: teleport a static downward camera to the vehicle's true pose |

ArduPilot HITL is archived and Plane-only, so there is no hardware-in-the-loop option here.

The stock ArduPilot Gazebo model (`iris` / `iris_down_cam`) cannot take off inside this build: it
gets its lift from `gz-sim-lift-drag-system` blade elements, and the vehicle plugin only sets joint
*angular velocity*, which does not produce enough thrust on Gazebo Harmonic 8.15 (see
`STATUS_2026-11-06.md` §8). The `x500_apm` model works (§8.1) — it keeps the PX4 motor model and
pushes motor speeds from `SERVO_OUTPUT_RAW` over MAVLink — and flying inside Gazebo is therefore
possible, just with a second physics loop and a motor bridge to keep tuned.

The split is a deliberate trade-off for vision work, not an impossibility: the renderer stays a
pure sensor, the vehicle can climb to 100 m and fly 240 m legs without retuning a physics model,
and the localization problem is decoupled from the flight-dynamics problem. The price is that the
camera is teleported rather than rigidly attached (no lens distortion, no camera vibration, and a
~1 frame timing skew), and the IMU/baro/mag come from SITL's numerical sensors rather than from
Gazebo bodies.

```bash
# one command: gz + frame pipe + SITL + pose bridge + vision loop, and it cleans up after itself
/usr/bin/python3 scripts/run_live.py --run outputs/geo/boulder --alt 100 \
    --waypoints '80,0|80,-80|0,-80' --speed 4 --reach 8 --rate 4

# PASS after ~2 min:
#   waypoints_reached 3/3   fixes 251/251   fix_rate 1.0   clock_synced true
#   error_m: median 0.472   p90 1.158      (map match vs the simulator's true pose)
```

It writes `outputs/geo/<name>/live/{report.json,fixes.csv,logs/*.log}`. The per-frame CSV has the
matched position, ArduPilot's own estimate, its velocity, the EKF health flags, the active setpoint
and the true position, so a failed run explains itself.

**The five bugs worth remembering** (each one produced a plausible-looking but wrong flight):

1. `VISO_TYPE` defaults to `0` (None) and is `@RebootRequired`. Until it is `1`, ArduPilot parses
   `VISION_POSITION_ESTIMATE` and throws it away: no EKF position → no HOME → `Arm: AHRS: waiting
   for home`. It has to be in SITL's `--defaults` file, not a live `PARAM_SET`; `run_live.py`
   generates that file from `sat2aerial.closure.VISION_PARAMS`.
2. The pre-arm check walks **every** EKF source set, including every `VELZ`. With `GPS1_TYPE=0`
   (4.7 name for `GPS_TYPE`; `ARMING_CHECK` is now `ARMING_SKIPCHK`), a leftover GPS reference in
   `EK3_SRC2_*`/`EK3_SRC3_*` fails with `EK3 sources require GPS`.
3. `recv_match(type=[...])` filters by message **name**. Passing ids (`[30, 32, 164]`) matches
   nothing, silently, forever.
4. **Teleporting the camera to the EKF's own estimate makes the vision channel an echo.** The
   image then shows what the EKF already believes, map matching can only confirm it, and the
   estimate is free to drift — measured: the loop "flew" to −230 m at 12 m/s with 100 % of fixes
   succeeding. The camera must sit at the vehicle's *true* pose, which SITL exposes as `SIMSTATE`
   (int32 lat/lng, no float32 loss). `--pose-source ekf` reproduces the echo on purpose.
5. GUIDED position targets fly at ArduPilot's own `WP_SPD` (default 10 m/s), so `--speed` has to
   reach the autopilot; `run_live.py` writes `WP_SPD` into the defaults file. (The waypoint
   controller's speed only matters offline.)

Timestamps: `VISION_POSITION_ESTIMATE.usec` must be in a clock the EKF can subtract from its own.
Unix epoch happens to work in SITL, but the loop asks for `SYSTEM_TIME`, anchors to ArduPilot's
boot time, and reports `clock_synced` — check it before trusting a run.

Honest caveats on the live numbers: the error is measured against `SIMSTATE` sampled when the
message arrives, while the frame was rendered ~0.2–0.5 s earlier, so at 4 m/s it is an **upper
bound** (≈1 m of it can be timing skew). The clean accuracy figure remains the offline replay
(median 0.22 m). There is also a startup transient: the first real fix can be ~35 m from the
bootstrap point, which the EKF absorbs as a position reset with a lane switch.

### 4. Use real flights instead of a simulator

`scripts/import_uav_visloc.py` converts a UAV-VisLoc-style flight into the same `pairs/` +
`poses.csv` + `meta.json` layout, so `localize` reports metric error on **real imagery** with the
dataset's own GPS as truth:

```bash
env_flight/bin/python scripts/import_uav_visloc.py \
  --src data/UAV-VisLoc --out data/uav_visloc/01 --ref-meters 400 --ref-size 1024
env_flight/bin/python -m sat2aerial localize --run data/uav_visloc/01 --match-mode gray
```

Two layouts are understood. v1 is one GeoTIFF per flight plus a CSV of lat/lon/height per frame.
v2 (the Zenodo release) is `NN/satellite-tile/satelliteNN_R_C.jpg` chips — the names are their
**pixel origins** — plus `NN/UAV-image/` and two coordinate CSVs. The v2 quirks below are
measured from the data, not assumed:

* the chip grid steps exactly 135 px in both axes (phase correlation: dx=+134.7, dy=+135.0), so
  the mosaic is rebuilt from the names, never from the CSV;
* the latitude column is broken: the north step is 1.30x the east step for a square 135 px grid
  (1/cos(40.34°) = 1.31). The longitude axis is trusted and the strip is re-centred on the track
  (`--strip-shift auto|off|<metres>`);
* EXIF is stripped, so the camera GSD is measured from consecutive frames — 0.0563 m/px on
  flight 07, i.e. a 1414 px frame covers 79.6 m;
* the satellite product is a **corridor**: 92.3 m tall against a 79.6 m frame. The reference crop
  is therefore rectangular (2.5x wider than tall, `--ref-aspect`) and cannot grow north-south.

```bash
env_flight/bin/python scripts/import_uav_visloc.py --src data/uav_visloc/raw --sites 07 \
  --out data/uav_visloc/processed/07
env_flight/bin/python -m sat2aerial localize --run data/uav_visloc/processed/07 --matcher ncc
```

Measured on flight 07 (30 frames):

| matcher | fix rate | median error | debiased median | p90 |
|---|---|---|---|---|
| `--matcher features --match-mode edges` | 0.33 | 51.2 m | — | 119.4 m |
| `--matcher ncc` | 0.87 | 11.8 m | 2.76 m | 73.4 m |

Read those numbers with three facts:

* the constant east/north offset (9.0 / 3.8 m) is a georeferencing bias between the satellite
  mosaic and the drone GPS, not matcher error; `error_m_debiased` removes it and leaves 2.76 m;
* the failures sit at the strip ends and in two mid-strip sections whose satellite content does
  not correspond (older imagery) — no matcher can fix a wrong reference;
* the drone GSD is ~4x finer than the mosaic, so SIFT's ratio test keeps almost nothing. Dense
  correlation at the mosaic's detail limit is what survives, hence `--matcher ncc` is the
  recommendation for real data. `features` remains the default because it wins in the simulator.

`--offset-x/--offset-y` shifts the reference window: with a zero offset the score is an upper bound
(matchability); with an offset comparable to your position uncertainty it becomes an absolute-fix
test. Dataset licences still apply — see the notes in `STATUS_2026-11-06.md`.

Synthesising drone-like views from satellite crops is not a shortcut available on this provider:
`cf.flux-2-klein-9b` is text-to-image only, and every image-edit model (`qwen-image-3.0-pro`,
`qwen-image-3.0`, `qwen-image-edit-plus`) is rejected by `/v1/images/edits`. A text-only model
invents geometry, which makes the output useless as a localization reference; the measured
alternative above needs no synthetic data.

#### The v1 GeoTIFF release (the 17 GB download)

The original release is `NN/drone/*.JPG` + `NN/NN.csv` + `NN/satelliteNN.tif`. Flight 07 is
3000x2000 frames at a measured **0.0571 m/px** (171 x 114 m footprint) over a satellite raster of
only 3000 x 170 px. Two facts decide the matcher:

* the raster is **anisotropic** — 0.2276 m/px east, 0.2973 m/px north — and that is real, not a
  bug: `--gsd-mode square` makes the result worse (median 4.28 m, fix 0.70 vs 1.70 m, 0.87). The
  declared transform is right; do not "correct" it;
* the drone camera is mounted **~85-90 degrees** away from the map's north-up frame, and the
  satellite corridor is only **43% of the frame height**.

| setup | fix rate | median | p90 | under 2 m |
|---|---|---|---|---|
| SIFT `gray` | 0.00 | — | — | 0.00 |
| SIFT `edges` | 0.03 | 17.0 m | 17.0 m | 0.00 |
| `ncc`, no rotation | 0.00 | — | — | 0.00 |
| **`ncc --rotate-search 360`** | **0.83** | **1.70 m** | 13.4 m | 0.43 |

So the real-data recipe is: correlation, not features, plus an explicit rotation search and a
centre-band crop (the reference is thinner than the frame). North-south is unobservable on this
flight (the corridor has no vertical slack), so the whole error is east-west. A 2-degree step over
the full circle costs ~0.33 s per frame on this CPU. One command runs the whole download:

```bash
env_flight/bin/python scripts/run_uav_visloc.py --src data/uav_visloc/raw_v1 \
  --out data/uav_visloc/processed_v1 --rotate-search auto
```

`--rotate-search auto` uses 360 degrees for v1 (camera yaw unknown) and 0 for v2 (frames already
north-up), does a 10-degree coarse pass and then refines at `--rotate-step`, and writes
`summary.csv` next to the per-flight folders.

**The result is not uniform across flights.** Every flight of the download, every 5th frame
(`--frame-stride 5`):

| flight | frames | fix rate | median | p90 | under 5 m | score |
|---|---|---|---|---|---|---|
| 01 | 164 | 0.46 | 66.1 m | 123 m | 3.7% | 0.117 |
| 02 | 215 | 0.60 | 36.7 m | 93 m | 6.5% | 0.143 |
| 03 | 154 | 0.82 | 58.0 m | 126 m | 3.9% | 0.181 |
| 04 | 148 | 0.82 | 79.3 m | 157 m | 0.7% | 0.170 |
| 05 | 95 | 0.74 | 53.0 m | 118 m | 1.1% | 0.160 |
| 06 | 69 | 0.74 | 28.0 m | 88 m | 5.8% | 0.151 |
| **07** | 6 | **1.00** | **1.02 m** | **1.97 m** | **100%** | **0.215** |
| 08 | 207 | 0.78 | 74.3 m | 183 m | 4.8% | 0.166 |
| 09 | 154 | 0.48 | 84.5 m | 190 m | 0.7% | 0.114 |
| 10 | 29 | 0.69 | 60.2 m | 97 m | 0.0% | 0.133 |
| 11 | 118 | 0.65 | 72.9 m | 198 m | 6.8% | 0.147 |

Flight 07 is the only flight whose satellite reference is a **tight corridor cut around the
flight line** (683 x 51 m, near the drone's own resolution); its 30-frame run gives 1.70 m median
(1.02 m on the every-5th subset). The other ten flights come with broad 2.5x8 km to 11x4.9 km
mosaics, and their correlation peaks sit at 0.11-0.18 — the matcher locks onto the wrong,
similar-looking parcel.

**The acquisition date gap is not the predictor.** From the dataset paper's own table, Zhuxi
(flight 06) has a **one-month** gap between the drone flight and its satellite map, yet scores
28 m, while Donghuayuan (07) has a **4.9-year** gap and scores 1 m. What separates them is the
reference geometry and local georeference: 07's map is a corridor cut around the flight line (so
there is almost no ambiguity), and its local georeference agrees with the drone track.

This is a data limitation, not a geometry bug. On flight 01 the following were swept and none
raised the peak or the accuracy: vertical scale 0.8-1.3x, Gaussian blur 1-8 px, `edges` mode, all
four flips, the reference at native GSD, and even a **global** search over the whole raster at
five scales — no peak appears near the annotated pose at any scale or rotation.

#### The truth-rank test: is the right place even identifiable?

Everything above says "the matcher picks the wrong parcel". This test asks the prior question. For
the correct scale, the **whole** correlation surface is computed and the NCC at the *correct*
alignment (crop centre) is ranked against every other offset:

| frame | NCC at the correct place | percentile among all offsets |
|---|---|---|
| 06_0001 | -0.015 | p58 of 66 875 |
| 06_0111 (wide crop) | 0.058 | p74 of 73 229 |
| 06_0221 (wide crop) | 0.136 | p57 of 66 875 |
| 01_0411 | 0.103 | p42 of 13 719 |
| 08_0516 | 0.020 | p46 of 2 169 |

**The correct place is statistically indistinguishable from a random place.** Widening the crop
from 433 x 217 m to 1000 x 500 m does not change that (p57-p74; the best peaks, 0.27-0.36, sit
85-220 m away). A global search over the whole mosaic peaks at 0.44-0.57, but 0.5-5.5 km from the
truth.

The global sweep covers every flight of the download (one mid-flight frame each, `--work-gsd 0.8`,
scales 0.8/1.0/1.25, 30-degree rotation steps). Not one frame lands within 10 m and the best is
275 m out:

| flight | best peak | best scale | offset from the truth |
|---|---|---|---|
| 01 | 0.456 | x1.25 | 810 m |
| 03 | 0.568 | x0.80 | 5 529 m |
| 04 | 0.528 | x0.80 | 2 580 m |
| 05 | 0.380 | x0.80 | 766 m |
| 06 | 0.477 | x1.00 | 667 m |
| 08 | 0.520 | x1.25 | 5 434 m |
| 09 | 0.490 | x0.80 | 2 607 m |
| 10 | 0.408 | x1.00 | 275 m |
| 11 | 0.424 | x0.80 | 1 802 m |

Note how stable some of these are: on flight 03 the top three peaks (0.568/0.546/0.541) all sit at
the *same* 5.5 km-distant village. The matcher is confident and wrong — it is matching a look-alike
place, which is exactly what a weak correlation surface does. Flight 10 shows the extreme case:
five frames spread over 350 m of the track all return **the same** lat/lon to seven decimals, i.e.
the surface has one attractor and the frame content never overrides it.

So the 28-85 m numbers are not a matcher deficiency at all: the satellite mosaic and the drone
GPS track are **not co-registered** to better than tens of metres (Google Earth imagery against raw
WGS84 GPS). Dense correlation, feature matching, rotation search, scale search and image
restyling all solve the same wrong problem until that is fixed.

**And a correction about flight 07's 1 m.** Its corridor is cut around the flight line itself:
49 m of corridor against a 189 m frame. The cross-track answer is therefore the corridor centre
*by construction*, and the 1 m is a genuine **1-D, along-track** result (±107 m of freedom on each
side of the crop), not a 2-D localization.

Operationally this says: **georeference the reference first** (RTK ground control, or register it
against one known point), then shoot it fresh over the corridor. With a registered corridor the
problem becomes the well-posed 1-D case that already works at ~1 m.

#### FLUX.3-Klein augmentation (image-conditioned restyle)

The appearance gap between a satellite orthophoto and a small-sensor drone photo is real, so the
H100's FLUX.3-Klein was used to restyle reference crops into drone-like views. The model is
image-conditioned: a base64 reference goes through `VAEEncode -> ReferenceLatent`, so it edits the
crop instead of inventing a scene.

**The prompt decides whether geometry survives.** "aerial photo taken by a fixed-wing drone…"
redraws the scene (roads and blocks move by tens of percent — useless as a map). Adding *"keep
every building, road, field boundary and tree in exactly the same place, only change the capture
style"* preserves the layout and only shifts the look (haze, softer lens, muted contrast).

Measured on six frames of flight 06 (reference 1536x768, ~13 s per image):

| metric | raw reference | FLUX-restyled |
|---|---|---|
| own-crop correlation peak (mean) | 0.25 | 0.27 (+8%) |
| 6-way retrieval, correct crop ranked first | 1/6 | 1/6 |
| `localize` fix rate | 4/6 | 5/6 |
| median error | 23.7 m | 39.3 m (24.2 m debiased) |

The appearance moves toward the drone view, but the matching decision does not change. That is
consistent with the truth-rank test above: when the reference is not co-registered with the flight,
no amount of restyling creates the missing correspondence. Restyling is worth keeping for the case
where registration *is* good — it costs one pass over the map, cached on disk.

```bash
# restyle a run's reference crops into a sibling run
env_flight/bin/python scripts/flux_augment_refs.py \
    --run data/uav_visloc/processed_v1/06 --out data/uav_visloc/processed_v1/06_flux \
    --limit 6 --stride 11 --steps 12

# ask whether the restyle discriminates the right place
env_flight/bin/python scripts/flux_retrieval_eval.py \
    --run data/uav_visloc/processed_v1/06_raw6 --aug data/uav_visloc/processed_v1/06_flux
```

#### Temporal gating

`sat2aerial/tracking.py` adds the layer a navigator would have anyway: a constant-velocity Kalman
filter with an innovation gate, plus a RANSAC line fit as a curve-tolerant consistency check.
`scripts/filter_uav_visloc.py` applies it to a recorded `results.csv`:

```bash
env_flight/bin/python scripts/filter_uav_visloc.py --root data/uav_visloc/processed_v1
```

| flight | matcher fixes | consistent trajectory | after filtering |
|---|---|---|---|
| 07 | 100% within 1.2 m | 100% inliers | 1.19 m median, 2.07 m p90 |
| 01-06, 08-11 | 28-85 m median | 0-10% inliers | no consensus to filter |

The filter works and is unit-tested (it rejects 120 m outliers, coasts through missing samples,
and recovers the line), but it cannot rescue the other flights: their per-frame fixes are not
merely noisy, they are *inconsistent*. The step magnitude is right (the matcher follows the
flight) while the lateral/along-track position jumps 20-100 m between frames, so no temporal
model can pick the right branch without the truth.

### 4b. How many frames does a sortie produce?

`scripts/plan_capture.py` turns speed, altitude, lens and overlap into frame counts, frame rate
and storage. Defaults (63°x46° FOV, 42 MP, 35 mm, 70/80% overlap, 200 m/s, 10 km):

| altitude | footprint across x along | drone GSD | frames @70% / @80% | fps @70% / @80% | storage |
|---|---|---|---|---|---|
| 100 m | 122.6 x 84.9 m | 1.3 cm | 394 / 590 | 7.9 / 11.8 | 8.6 GB |
| 200 m | 245.1 x 169.8 m | 2.6 cm | 198 / 296 | 3.9 / 5.9 | 4.3 GB |
| 300 m | 367.7 x 254.7 m | 3.9 cm | 132 / 198 | 2.6 / 3.9 | 2.9 GB |

At 100 m the drone sees the ground ~23x finer than a 0.3 m/px satellite map; at 300 m it is still
~8x. That ratio, not the altitude itself, is what decides whether sparse features survive.
Motion blur also bites: at 200 m/s and 1.3 cm/px, a 1 px smear allows only ~0.65 ms exposure.

## Tests

```bash
env_flight/bin/python -m unittest discover -s tests -v
```
