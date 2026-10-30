# Proposed notebook review — pending approval

`notes_commit` could not save: the opencode workspace root is `/` and writing `/.note.yaml`
returns `EACCES: permission denied` (root-owned `/`, no passwordless sudo). The content below is
the exact proposed notebook payload, kept in the repo for review.

To apply: run opencode from `/home/arman/drone-vla` (workspace root becomes writable), or run
`sudo touch /.note.yaml && sudo chown arman:arman /.note.yaml`, then ask me to re-run `notes_commit`.

---

## Folder summaries

### navlab
FastAPI/SQLite RTL web platform for GPS-denied UAV navigation experiments. It covers
missions/waypoints/geofence, satellite map upload, generation jobs (dummy/ComfyUI/BFL), SIFT
matching and run reports, PX4 SITL start/stop with Gazebo camera and ground-truth bridges, and a
map-keyframe EV localizer. Current state: GPS-denied map-EV with baro height is validated in SITL;
offboard control remains gated after a fly-away.

### sat2aerial
Satellite-to-aerial generation and map-matching bake-off. Pluggable backends (dummy, ComfyUI FLUX 3
int8, BFL API, local flux3), SIFT/FLANN/RANSAC matching with coverage-gated success, synthetic
multi-seed and real UAV-VisLoc/University-1652 pilots. Findings: 2D translation preserves top-down
layout but not oblique viewpoint; a style reference is essential near nadir; coverage prevents
collinear false positives.

### scripts
Operational scripts for PX4/Gazebo: flight and telemetry logging, frame capture, low-latency camera
JPEG bridge, Gazebo pose/truth bridge with NED yaw, visual keyframe-map recording, localizer holdout
evaluation, UAV-VisLoc pair extraction, and hardware BOM Excel generation.

### tests
16 unittest tests covering sat2aerial dataset/prompts/matching/metrics/sizing and navlab
geo/corridor/tiles; all pass.

---

## File entries

### navlab/sim.py
SimManager owns the PX4 SITL lifecycle and MAVSDK control. It starts
`make px4_sitl gz_x500_mono_cam_down` with HEADLESS=1 and PX4_GZ_WORLD=baylands, home set to the
mission start. Camera and truth are provided by `gz_latest_frame` and `gz_truth_bridge`. `_ev_loop`
sends `VISION_POSITION_ESTIMATE` via `drone.mavlink_direct.send_message`; the validated setup is
`EKF2_EV_CTRL=1` (horizontal), `EKF2_HGT_REF=0` (baro, reboot-required), `EKF2_GPS_CTRL=0`,
`COM_ARM_WO_GPS=1`. `use_offboard` is False after a SITL fly-away; `_flight` uses `goto_location` as
the proven control path. PX4 stdout is read into a deque with stdin=PIPE to avoid prompt spam/disk fill.

### navlab/localizer.py
`MapLocalizer.__init__` loads `manifest.json` into `frames`, `small_matrix`, `keypoints`,
`descriptors`, and `east/north` arrays. `locate(frame_path, prior_east, prior_north, radius,
last_keyframe)` shortlists by a 32x32 descriptor and keyframe window, then runs
SIFT+BFMatcher+findHomography and requires inliers>=12 and coverage>=0.02. Holdout evaluation gives
98.9% match, mean 0.56 m, p95 1.8 m, ~17 ms.

### navlab/web.py
FastAPI app with 23 routes and 9 Jinja/HTMX/Leaflet templates. Key handlers include `sim_start`,
`sim_stop`, `sim_fly`, `sim_ev_start`, `sim_gps_disable`, `sim_param`, `quick_run`,
`start_generation`, and `start_matching`. Routes use `app.add_api_route` rather than decorators
(decorator text was being redacted in this environment).

### navlab/geo.py
Functions: `route_points_multi`, `corridor_geometry_multi` (AEQD+shapely buffer and polygon),
`route_summary_multi`, `tile_bounds_multi`, `distance_to_polyline_m`, `local_ned`, and
`mission_route_points`. `tests/test_navlab_geo.py` covers corridor/tiles. Used by
`web._ensure_route`, `jobs._run_generation`, and the software geofence.

### navlab/jobs.py
Single background worker queue for generation and matching. `_run_generation` crops map tiles, calls
`sat2aerial.backends`, and stores `GeneratedTile`; `_run_matching` calls
`navlab.matching.best_match` and stores `Run` metrics with overlays.

### navlab/generation.py
`get_backend`/`generate_reference` build prompts and dispatch dummy/ComfyUI/BFL backends;
`comfy_available` checks the local server. FLUX 3 Klein BF16 does not fit comfortably on 8GB; int8
ComfyUI is the working local path and BFL is the API path.

### navlab/models.py
SQLModel schema: `Mission` (lat/lon/altitude and `meta.waypoints`), `MapSource`, `Job`,
`GeneratedTile`, and `Run`. Datetimes must be timezone-aware; naive `datetime.utcnow()` is rejected
by SQLAlchemy 2.1.

### navlab/camera.py
`CameraStream.start` launches `/usr/bin/python3 scripts/gz_latest_frame.py`, `status` reads
latest.jpg/json, and `stop` kills the process group. It auto-retries from `SimManager.status`.

### navlab/truth.py
`TruthBridge.start` launches `gz_truth_bridge.py`; `read` parses truth.json (x,y,z,NED yaw);
`status` reports age/running. Used by SimManager truth-EV and map-EV validation.

### navlab/ROADMAP.md
Status table plus sections A-H. Records completed UI/SITL/camera/waypoints and GPS-denied map-EV
with baro; marks offboard gated (`use_offboard=False`), and lists remaining fusion, real
camera/HITL, 4090 internal service, data/benchmark, UI/ops, and field safety.

### navlab/README.md
Sections for run instructions, pages, workflow, backends, SITL notes, limitations, and the Excel BOM
link. Documents the `dummy`/`comfy`/`bfl` backends and the `/media/hardware_bom.xlsx` download.

### scripts/gz_latest_frame.py
Gazebo camera bridge. With system Python it requires
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, auto-discovers the camera image topic via
`gz topic -l`, and atomically writes latest.jpg/json at 5fps.

### scripts/gz_truth_bridge.py
Subscribes to `/world/<world>/pose/info` (`pose_v_pb2.Pose_V`), filters `x500_mono_cam_down_0`,
converts the quaternion to NED yaw, and atomically writes truth.json at 20Hz. Replaces slow and
timeout-prone `gz model -p` polling.

### scripts/record_map.py
Reads `outputs/navlab/camera/latest.jpg` and `outputs/navlab/truth/truth.json`, dedupes by camera
mtime, copies keyframes, and writes a manifest with `z0` and per-frame x/y/z for `MapLocalizer`.

### scripts/eval_localizer.py
`make_holdout` puts odd frames in the map and even frames in live; `main` sweeps radius, calls
`MapLocalizer.locate`, and reports match rate/mean/median/p95/latency. Verified 98.9% match, p95
below 2 m.

### scripts/make_hardware_xlsx.py
`main()` writes `outputs/navlab/hardware_bom.xlsx` with Summary/Drone_BOM/GPU_Station/Available/Notes
sheets, RTL layout, and editable USD/toman formulas. `navlab/templates/hardware.html` links it at
`/media/hardware_bom.xlsx`.

### scripts/uav_visloc_pairs.py
Builds paired satellite crops from UAV-VisLoc GeoTIFF/CSV/drone frames with offset, footprint, CRS,
and edge handling; outputs a manifest consumed by sat2aerial matching experiments.

### sat2aerial/matching.py
`match_images` builds CLAHE/Canny or gray inputs, uses `_detector` (SIFT), `_matcher` (FLANN), a
ratio test, `findHomography` RANSAC, and `_coverage` convex hull; success requires `min_inliers`,
`min_ratio`, and `min_coverage=0.03`. `draw_matches` renders inlier overlays.

### sat2aerial/backends/comfy.py
`ComfyBackend.generate` stages references, builds a ReferenceLatent graph, submits `/prompt`, polls
`/history`, and downloads via `/view`; measured ~56-85 s per 768px image on the RTX 4060. Uses int8
FLUX 3 Klein files under `ComfyUI/models`.

### sat2aerial/bakeoff.py
`run_bakeoff` supports variants/seeds, generates references, matches to real aerial frames, and
writes `report.md`/`results.csv` with inliers/ratio/coverage/consistency/edge IoU/SSIM plus seed
variance.

### sat2aerial/sizing.py
`SizingInput` and `compute()` calculate footprint, GSD, corridor tiles, mission frames, generation
hours/cost, and matching load; `format_report()` renders them. At 200 m footprint is ~296x197 m and
5000 km at 0.5 m GSD needs ~38k tiles.

### sat2aerial/README.md
Documents the `sat2aerial` CLI (`generate`, `bakeoff`, `demo`, `sizing`, `selfcheck`), `run_bakeoff`,
`build_prompt`, and `match_images` with `min_coverage`; backends `dummy`, `comfy`, `bfl`. Pilot
results: synthetic style-ref success vs no-style, real oblique negative, and 8GB/4090 constraints.

### README.md
Top-level project map: PX4/Gazebo/Isaac, FLUX action dataset, NavLab, sat2aerial, and hardware notes.

### NOTES.md
Chronological engineering log of environment traps, PX4/Gazebo, FLUX quantization, NavLab, SITL EV,
map-EV, and failures/fixes (sections 12.x and 13.x).

---

## Relations

1. `navlab/web.py → navlab/sim.py` — `sim_start`, `sim_stop`, `sim_fly`, `sim_ev_start`,
   `sim_gps_disable`, and `sim_param` call SimManager methods.
2. `navlab/sim.py → navlab/localizer.py` — `SimManager._ev_loop` calls `self.localizer.locate` when
   `ev_source == "map"`.
3. `navlab/sim.py → navlab/truth.py → scripts/gz_truth_bridge.py` — `_truth_pose` reads
   `self.truth.read()`; `TruthBridge` launches the bridge.
4. `navlab/sim.py → navlab/camera.py → scripts/gz_latest_frame.py` — `CameraStream` launches the
   bridge; `_ev_loop` reads `CAMERA_LATEST`.
5. `scripts/record_map.py → scripts/gz_latest_frame.py` — recorder reads `latest.jpg`.
6. `scripts/record_map.py → scripts/gz_truth_bridge.py` — recorder pairs frames with `truth.json`.
7. `scripts/eval_localizer.py → navlab/localizer.py` — `main` creates `MapLocalizer` and calls
   `locate` on holdout frames.
8. `navlab/jobs.py → navlab/generation.py → sat2aerial/backends/comfy.py` — `_run_generation` →
   `generate_reference` → `ComfyBackend.generate`.
9. `navlab/jobs.py → navlab/matching.py → sat2aerial/matching.py` — `_run_matching` →
   `best_match`/`match_pair` → `match_images`.
10. `scripts/make_hardware_xlsx.py → navlab/templates/hardware.html` — `main()` writes the BOM;
    the template links it at `/media/hardware_bom.xlsx`.
11. `navlab/ROADMAP.md → NOTES.md` — status/roadmap mirrors sections 12.x/13.x of the engineering log.
