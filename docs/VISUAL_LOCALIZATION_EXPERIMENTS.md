# Satellite and aerial image matching experiments

This is research software. The existing results do not establish that the
prototype can safely navigate an aircraft. No autopilot integration, estimator
covariance calibration, integrity monitoring, or real-flight validation is
provided. The module requires an initial position and does not perform global
relocalization.

## Experiment assumptions

The scripts use the `UAV_VisLoc_dataset.zip` dataset under
`data/raw/uav_visloc/UAV_VisLoc_dataset`. `scripts/prepare_visloc_local.py`
constructs a **fixed** map bounding box from the labeled evaluation segment and
its margin. This is an offline, corridor-assisted experiment, not a test of
finding an unknown location on an entire map. The first two labeled positions
are used to initialize position and velocity. Subsequent labels are used for
scoring only unless `--oracle-motion` is set. Oracle motion derives each
inter-frame displacement from ground truth and is a diagnostic upper-bound
experiment, not an actual inertial/visual odometry system.

The benchmark has **no default aircraft speed or flight altitude**. Time and
image observations come from the dataset; velocity initialization is inferred
from the first two positions. Separately, `Prior.update` has a default
80 m/s gate for accepting a *new velocity estimate*; estimates above that gate
are not used to update velocity. This is not an assumed flight speed and was
not validated as a flight limit. Other image-generation CLIs have their own
prompt altitude defaults, distinct from the real-image benchmark. For the
`06_after_turn` segment, the median
finite-difference ground speed between consecutive labeled frames with intervals
under 10 seconds is about 20 m/s, and the median recorded altitude is 839 m.
These quantities are descriptive only. The altitude reference (especially
height above ground) is not established, and the dataset timestamps are not
always reliable. Dataset 06 includes a timestamp reversal after frame 0092 and
large gaps around turns. Dataset 07 timestamps required a parser fix.

The matching results below use a generous 30 m error threshold, one-frame
acceptance, and a maximum of 8 candidate tiles; the threshold is only a metric,
not a flight requirement. Truth labels and reference maps themselves may have
systematic error. An accepted fix can be wrong. Errors below are relative to
available labels.

| Data segment | Reference and setting | Fixes within 30 m | Median accepted error | Notes |
| --- | --- | ---: | ---: | --- |
| 01 | raw, same 24 source tiles, LoFTR, 0/180° fallback | 12 / 25 | 19.9 m | 14 accepted; two >30 m |
| 01 | FLUX, validated, same 24 source tiles and settings | 10 / 25 | 22.7 m | 14 accepted; four >30 m |
| 02 | raw, LoFTR, 0/180° fallback | 19 / 25 | 12.7 m | 21 accepted; two >30 m |
| 04 | raw, LoFTR, 0/180° fallback | 17 / 25 | 24.0 m | 24 accepted; seven >30 m |
| 05 | raw, LoFTR, 0/90/180/270° fallback | 10 / 25 | 25.8 m | 12 accepted; two >30 m; 0/25 with only 0/180° |
| 06_straight | raw, LoFTR | 18 / 25 | 13.4 m | fixed corridor, two-position start |
| 06_straight | FLUX, validated, LoFTR | 22 / 25 | 13.4 m | same 60 raw source tiles; exploratory |
| 06_after_turn | raw, LoFTR, 0/180° fallback | 27 / 30 | 11.0 m | 28 accepted; one >30 m |
| 06_holdout40 | raw, same 42 source tiles, LoFTR, 0/180°, oracle motion | 28 / 30 | 11.7 m | oracle motion, includes large time gap |
| 06_holdout40 | FLUX, validated, same oracle | 28 / 30 | 12.6 m | 42 validated source tiles |
| 07 | raw, LoFTR, 0/90/180/270° fallback | 0 / 25 | unavailable | candidate coverage 100%; no fix |
| 08 | raw, LoFTR, 0/180° fallback | 0 / 25 | unavailable | candidate coverage 100%; no fix |
| 09 | raw, LoFTR, 0/90/180/270° fallback | 11 / 25 | 13.0 m | all accepted below 30 m; 50 s max gap |
| 10 | raw, LoFTR, 0/90/180/270° fallback | 3 / 25 | 19.8 m | candidate coverage 64%; trajectory prior later diverges |
| 11 contiguous | raw, LoFTR, 0/90/180/270° fallback | 21 / 25 | 15.3 m | 25 accepted; four >30 m |

Results are local ignored artifacts under `outputs/navlab/local_study/`.
The 06 FLUX result is not proof of general improvement: the number of frames
is small, accepted trajectories influence later candidate selection, and the
holdout comparison on the same 42 source tiles found 28/30 fixes within 30 m
for both versions and median accepted errors of 11.7 m raw versus 12.6 m
FLUX; both use ground-truth-derived oracle motion and the same configuration.
The same frames were accepted, but sequential priors may still differ. On
one segment, 60 of 66 generated references passed the geometric check; on
another, 42 of 60 passed. The check uses SIFT correspondence and registration;
it cannot certify preservation of local detail or correctness in all areas.
On 01, 24 of 30 generated tiles passed validation; using those same source
tiles, candidate truth-center coverage was 25/25 for both versions. Despite
the same coverage, FLUX had fewer fixes within 30 m and more accepted errors
over 30 m than raw. All these comparisons are short trajectory experiments,
not evidence of fleet-wide or flight performance.

In 04, accepted-fix residuals correlate strongly with recorded pitch/roll.
An exploratory offline linear residual fit on the first ten *accepted labeled*
frames gives a median 9.7 m and p95 19.2 m on the following 14 accepted frames,
versus 19.5 m and 54.0 m before correction. This excludes rejected frames,
uses ground-truth labels for fitting, and is a short, same-trajectory test;
it is not an online result. The same fitted correction worsens median error on
independent regions 01, 02 and 06. A yaw-aware variant also worsens 06, despite
helping some other samples. Camera orientation conventions, true AGL and
calibration are unknown. Neither correction is deployed. Reproduction and
feature definitions: `scripts/analyze_attitude_error.py` and ignored output
`outputs/navlab/local_study/04/attitude_offline_diagnostic/`.

Dataset 09 stores its satellite map in four files. `scripts/build_visloc_vrt.py`
can build an absolute-path VRT without copying their pixel data; pass that VRT
to `scripts/prepare_visloc_local.py --satellite`. The VRT depends on the source
paths remaining in place.

An offline visual inspection of flight 08 suggests a material scene limitation:
seven sampled camera images across the tested segment show mostly open water,
and the label-covered reference tiles also show water. All 25 labeled centers
are on water in the reference image. Wave and reflection textures need not be
repeatable between satellite and camera acquisition times. No stationary
landmark is apparent in those camera samples. The approximate footprint and
the unverified height/FOV do not establish the true camera coverage; nor does
this observation completely exclude registration errors or orientation
problems. Failure on these frames should not be generalized to all land scenes
or treated as something an image generator can reliably repair.

For an exploratory FLUX experiment on this segment, 10 of 32 generated tiles
passed the same geometric validation. These 10 tiles cover the truth center of
only 4 of 25 frames. With the same 10 source tiles, both raw and FLUX yielded
zero accepted fixes; the same result holds for those four covered frames.
This experiment cannot compare full-route performance because the remaining
21 truth centers were outside the validated reference subset. The 10/32
validation rate also depends on the chosen feature matcher and thresholds;
it is not a direct measurement of the fraction of warped images.

## Reproduction

Run the automated checks with:

```sh
env_flight/bin/python -m unittest discover -s tests -q
```

An existing prepared segment can be evaluated as follows:

```sh
env_flight/bin/python scripts/run_local_benchmark.py \
  --study outputs/navlab/local_study/06_after_turn \
  --out outputs/navlab/local_study/06_after_turn/reproduce \
  --matcher loftr --max-dim 768 --online --rotations 0,180
```

Use `--references` to supply a geometry-validated manifest and
`scripts/validate_generated_refs.py --help` for validation options. Never
assume that an image generator preserves coordinates. `--camera-gsd` is an
optional supplied estimate in metres per original camera pixel; it is not
computed from the label or default altitude. `--oracle-motion` is explicitly
labeled as a diagnostic in run metadata and must not be presented as an actual
motion-estimation result.

## Interface

`navlab.lab.navigator.VisualNavigator` accepts a georeferenced `TileStore`, an
explicit initial `Prior`, and BGR frames or image paths with strictly increasing
timestamps. The caller can supply an east/north relative motion increment and
its uncertainty for each interval. `NavigationResult` includes acceptance,
reason, candidate IDs, position when accepted, and match diagnostics. The
prior sigma is a search heuristic, **not** a calibrated covariance. A rejected
frame does not produce a position measurement. Users must independently handle
latency, erroneous fixes, sensor fusion, and failure modes. Generated tile
manifests without a successful geometric check are rejected by this interface.
The older batch runner can still load unchecked references, so its outputs must
be treated accordingly.

Optional LoRetta adapter: install its upstream repository and weights outside
tracked project files, follow both LoRetta's and its underlying model's
license terms, then configure `PYTHONPATH` and `NAVLAB_LORETTA_CHECKPOINT`.
In a small ground-truth-selected diagnostic using the nearest reference tile
for three images from flight 08, no LoRetta match passed acceptance. Two
outputs had about 1,000 densely sampled inliers concentrated in less than
1.2% of the image; their derived errors were about 185 m and 209 m. Dense
inliers are correlated, so raw counts are not directly comparable with LoFTR.
These three examples are insufficient to establish comparative performance.

The optional REDI-Match adapter uses the official `outdoor.pth` checkpoint,
set with `NAVLAB_REDIMATCH_CHECKPOINT`; the source checkout is external. A
preliminary five-frame laptop-GPU run is **invalid**: a follow-up diagnostic
found repeated CUDA out-of-memory errors, which the old report did not retain.
Do not interpret those exceptions as image-matching failures. The navigator
now rejects an incomplete candidate search as `matcher_error` and the online
benchmark records execution errors in its frame CSV. The batch runner aborts
on matcher execution errors. Further comparison requires a rerun with
sufficient memory and error-free candidate evaluation.

On H100 the official REDI checkpoint completed 33 offline pairs, three from
each of 11 regions, with no inference exceptions. Each reference tile was
selected using the labeled position: this is an oracle pair test, not online
navigation. With the current basic geometric thresholds, 15 of 33 pairs were
accepted; nine accepted errors were <=30 m and six were >30 m. LoFTR on the
same pairs (with four right-angle rotations) accepted 17, of which 12 were <=30 m
and five >30 m. Matching these few selected pairs does not imply comparative
flight or end-to-end performance. Details are in
`outputs/navlab/redi_pair_study/{pairs,redi_results,loftr_results}.json`.

## Measured-attitude subset and timing interpretation

`scripts/prepare_visloc_local.py --first-level-deg 3 --count 12` selects the
first 12 consecutive source frames with *measured* absolute Omega and Kappa
below three degrees, no more than ten seconds between adjacent timestamps.
The first two seed position and velocity; the ten subsequent frames are
evaluated. Camera tilt and airframe-to-camera extrinsics remain unknown.
Selection does not use match outcomes, but the fixed map ROI still uses the
segment's ground truth offline, and the tracks are short and correlated.
Region 07 has no usable angle fields and is not a measured-level sample.
The resulting logs and per-frame results are under `outputs/navlab/level_study`.

With the same four-rotation LoFTR configuration on regions 01–06, 09 and 10,
80 subsequent frames were evaluated: 46 accepted, of which 40 were within 30 m
of the original labels and six were not. Region 09 ended with at least 34 s
without a new accepted fix; the actual gap is right-censored by the segment
ending. This is a small, convenience-selected observational sample, not a
flight-time availability estimate.

As a diagnostic, for region 02 five of ten subsequent frames were accepted;
three were within 30 m and two were not, with maximum 85.76 m. The largest
interval without an accepted fix within the sampled sequence was 26 s. These
figures do not establish the absolute camera-position error because UAV-VisLoc
labels may be offset or timestamp-misaligned. Nor do they establish 1 Hz
coverage: the original query frames are several seconds apart.

Speed cannot be determined from a positioning-error percentile alone.
Consider only an idealized position-hold model with no propagation: if a fix
has a *guaranteed* error <=20 m, a 30 m budget and a fixed gap of 1 s,
the sufficient bound from the triangle inequality is v <=10 m/s. The same
assumptions with a 26 s gap give <=0.38 m/s. These examples are not flight
limits: neither the assumed 20 m bound nor a worst-case gap has been proven,
and delay, false fixes, turns and control response are excluded. An actual
inertial/visual motion estimate can change the answer and must be validated
for the complete duration and uncertainty growth.

## Additional public research and cross-region pair diagnostic

- The official [VistaLoc-Set](https://github.com/Sea-hy/VistaLoc-Set) authors
  note offsets in some original UAV-VisLoc labels and manually adjust their
  image-center annotations. This does not establish accurate aircraft-center
  position, so evaluation reference points must be compared explicitly.
- [OrthoLoC](https://github.com/deepscenario/OrthoLoC) provides calibrated
  images and DOP/DSM for investigating geometry; its license is CC BY-NC-SA
  4.0. `data/raw/ortholoc_samples/highway_rural.npz` is one downloaded demo
  sample, not a trajectory benchmark.
- [OrthoTrack](https://github.com/cvg/orthotrack) uses map keyframes, DSM,
  optical flow and calibrated pose; its code is CC BY-NC 4.0 and its stated
  frame and map requirements do not match this sparse dataset automatically.
- The November 2026 [GIM-DKM and particle-filter study](https://link.springer.com/article/10.1186/s43020-026-00232-w)
  reports results with a known start and revised labels. Its numbers are not
  directly comparable with these original labels or an end-to-end flight.

GIM-LoFTR's official 50 h RGB checkpoint was run on H100 on the same 33
truth-selected pairs described above, with four right-angle fallback rotations.
Twelve were accepted, eight <=30 m and four >30 m; no inference exceptions.
This is not the GIM-DKM variant in the cited paper. The server used a private
CUDA MPS pipe and an isolated Kornia compatibility setup; the unused optional
`kornia_rs` image-IO import was stubbed. Results are in
`outputs/navlab/redi_pair_study/gim_loftr_results.json`. Running all pairs
requires the external official checkout and checkpoint plus `yacs`.

GIM-DKM 100 h used the author's full Lightning checkpoint (its SHA-256 was
checked against the author's Hugging Face `x-linked-etag` before loading),
the same truth-selected pairs, and the same basic homography acceptance gate.
On the server it accepted 24/33, with 16 <=30 m and eight >30 m. One accepted
water-image match had 421.5 m label error and only 3.17% image coverage;
another match had 69.1 m error despite 91% coverage. A post-hoc 5% coverage
threshold would remove two large errors in these same samples, but this is not
independent validation and would not remove the 69.1 m error. Full results:
`outputs/navlab/redi_pair_study/gim_dkm_results.json`. PyTorch reported an
allocation failure warning during the run but recovered; there were no
matcher execution exceptions. Pair runtimes and accuracy do not include
candidate search, motion propagation, or position-filtering overhead.

## Online local-search study with GIM-DKM

The experimental `RemoteGIMMatcher` runs the officially weighted GIM-DKM on
server 117 over a private SSH session while candidate selection, prior updates
and retrospective evaluation remain local. It transfers lossless, resized
images and caches them by content; it terminates the session after a protocol
error so a late reply cannot become the answer for another image. This is a
benchmark transport, not a flight API. Per-frame timings include SSH transfer
and all candidate matches, but exclude model initialization. The host shares
its GPU with other processes and reported a recovered PyTorch allocation
warning. Images are sparse, so timing on these frames does not establish 1 Hz
online throughput or queue stability.

On 80 frames across the eight measured-attitude sequences above, GIM-DKM with
768 px inputs, eight candidates, four right-angle fallback rotations and a
known start accepted 69; 58 had original-label error <=30 m and 11 >30 m.
Two accepted errors in region 10 were 195.7 m and 282.1 m. These are short,
non-independent samples and ROI-assisted; accepted-position label errors may
include reference/ground-footprint biases. The earlier 40/80 LoFTR figure used
1024 px, so it is not a controlled model comparison.

For a controlled 768 px comparison on regions 02, 04, 06 and 09, each with
ten later frames, the same map ROI, candidate limits, fallback rotations and
initialization, LoFTR accepted 30/40 (25 <=30 m, five >30 m); GIM-DKM accepted
35/40 (29 <=30 m, six >30 m). They ran on different devices and GIM included
SSH, so their runtime numbers are not a same-hardware comparison. Different
matches change the closed-loop prior, and the label/ROI limitations still apply.

A post-hoc rank-only analysis found that a tile containing the reference label
would be among the first two sorted tiles in 80/80 frames on the *eight-tile*
run's priors. However, a new closed-loop run with two candidates in region 09
accepted positions 126.8 m and 434.9 m from the labels. Candidate containment
of the image center is not sufficient to prove that the image footprint is
covered, that the match is valid, or that the trajectory remains unchanged.
The two-candidate run in region 02 had the same 9/10 <=30 m as the eight-tile
run; its measured p95 per-frame time was 2.11 s rather than 6.65 s. Ten frames
are not enough to establish a reliable latency percentile or flight rate.

The `--min-coverage .10` option is an exploratory gate motivated by errors
observed on these same examples. In region 01, the re-run had seven <=30 m
fixes and one >30 m, with maximum error 37.8 m, compared with seven <=30 m and
three >30 m without that gate, maximum 85.4 m. It must be tested on independent
flights and does not remove high-coverage geometric/label biases.

`--visual-propagation` enables an experimental image-to-image SIFT tracker,
anchored on the latest accepted image-to-map homography. It supplies only a
relative ground-plane motion increment to the prior. It has no calibrated IMU,
camera pose, or terrain model and must not be treated as VIO. On each of the
four tested 10-frame segments, it found nine frame-to-frame increments per
segment; mean increment errors relative to consecutive original labels were
15.1 m, 18.9 m, 15.7 m and 9.0 m for 02, 04, 06 and 09 respectively. With
those sequences and LoFTR at 768 px, it did not change the numbers of accepted
or <=30 m fixes.
The label differences are retrospective only and not fed back into tracking.

`RunnerConfig.max_speed_mps` and `--max-speed-mps` configure the threshold for
using inferred speed to update the search prior; their default of 80 m/s is
not an aircraft capability or a certified flight-speed limit. The image-motion
tracker uses the same configurable threshold when enabled.

## Metric 3-D geometry proof of concept

The experimental `navlab.lab.pnp.estimate_pose` solves the calibrated camera
pose from matched image pixels and corresponding 3-D map points. The separate
`scripts/benchmark_ortholoc_pnp.py` uses an already-selected orthophoto and its
co-registered DSM from the CC BY-NC-SA OrthoLoC demo, plus that sample's camera
intrinsics. It does not search for the right region, handle camera-to-airframe
extrinsics, fuse IMU or provide a navigation confidence bound. Results use the
sample's supplied extrinsics only for retrospective evaluation.

On all 14 available demo NPZ files, SIFT + PnP produced six accepted poses;
their horizontal errors ranged from 0.15 m to 2.48 m. The GIM-DKM + PnP run
accepted 12, with horizontal errors from 0.12 m to 2.79 m. One oblique urban
sample produced 0.98 m horizontal error with GIM-DKM, while SIFT failed. The
two GIM failures were the highway-rural xDOP and xDOPDSM samples. These are
hand-picked demo files and some are alternate representations of the same
scene, so they are not 14 independent trials. Their altitude, orthophoto
resolution and guaranteed map overlap differ from the user's UAV-VisLoc data;
this comparison does not establish performance at 500–4000 m, GNSS-denied
continuous navigation or on the user's hardware. The outputs are under
`outputs/navlab/ortholoc`, and the sample license restricts commercial use.

An additional deterministic evaluation selected five evenly spaced R samples
from each of the three OrthoLoC `test_outPlace` sites (L08, L50, L51), then
paired each with the xDOP sample of the same index. All 15 pairs were checked:
query image, intrinsics, reference extrinsics and DSM were identical, and the
orthophoto differed. With the same supplied map crop and DSM, GIM-DKM + PnP
had 15/15 accepted and <=5 m in the same-domain version, and 13/15 accepted
and <=5 m with xDOP; the 13 cross-domain accepted errors ranged from 0.12 m
to 1.78 m. SIFT + PnP had 14/15 <=5 m same-domain and 6/15 <=5 m with xDOP;
one other xDOP fix was accepted with 11.35 m horizontal error. These are 15
correlated samples from three sites, with selected map crops, calibrated
cameras and known DSM, not a continuous flight or a 500–4000 m evaluation.
The selection and complete paired results are recorded in
`outputs/navlab/ortholoc/{outplace_selection,outplace_pairs}.json`.

As a resolution sensitivity diagnostic, the same 14 query images were
downsampled with their intrinsics and PnP reprojection threshold scaled
accordingly. SIFT + PnP accepted six at 50% width/height and two at 25%,
versus six at native size. The world points and camera positions were not
changed: this is image-resolution degradation, not a physical simulation of
flying at a greater altitude or a test at a different ground footprint.

The original UAV-VisLoc files examined locally do not provide a co-registered
DSM or camera calibration in the directories searched. Acquiring or deriving
those inputs, verifying vertical datums and image/camera-to-aircraft geometry,
and running on independent test trajectories are prerequisites for testing
whether this 3-D approach improves the main dataset.

## Subsequent disjoint segments and initial-offset diagnostic

With the previously selected two-candidate, 10% coverage rule, the first ten
query frames of the next qualifying contiguous level-attitude runs gave:

| Region | Accepted | <=30 m | >30 m | Maximum accepted error |
| --- | ---: | ---: | ---: | ---: |
| 01 | 7 | 1 | 6 | 46.6 m |
| 02 | 10 | 10 | 0 | 15.8 m |
| 03 | 10 | 10 | 0 | 26.4 m |
| 06 | 1 | 0 | 1 | 239.9 m |
| 09 | 9 | 9 | 0 | 23.8 m |

These later segments do not overlap the previous selected windows, but use
the same source regions/maps; they are not independent new geographic sites.
The ROI is still selected offline using the evaluation segment's reference
labels. In region 06, the large false fix occurred after several rejected
frames while the uncertainty/search area grew; this is not a calibrated
integrity test. Original-image timestamp spacing and starting velocity should
be considered before attributing the cause to any single component.

`scripts/analyze_seed_offset.py` tries to estimate a constant offset from only
the two known startup positions and compares existing accepted outputs after
the fact. It is not a corrected closed-loop run: 01 and 06 lacked two usable
startup matches, 02 and 09 had inconsistent initial offsets, and for 03 the
retrospective accepted-error median worsened from 16.2 m to 20.4 m. A simple
fixed correction therefore is not justified by these five diagnostics.

The region 06 later segment also has irregular 2–3 s timestamp intervals.
For the same eight evaluation frames, estimating the initial velocity from
five known pre-outage positions instead of the last two reduced mean prior
error from 189.6 m to 169.6 m, but both runs accepted the same erroneous
approximately 240 m fix and had no <=30 m fixes. The means include errors
after that bad fix. This is not a flight test or proof of timing-error cause.

After observing that error, an exploratory jump gate with `--max-jump 30` and
`--jump-sigma-scale 1` rejected that fix and produced zero fixes over the
entire 10-frame region 06 segment; it did not change the previously measured
results in regions 01–03. This threshold was chosen after looking at this very
error, and prior sigma is a heuristic, not a calibrated covariance. It must
not be interpreted as an independent reliability measurement.

## Adaptive crops and map-only visual anchors

For the same five 10-frame level-holdout segments, a prior-centred square crop
with a fixed exploratory 450 m horizontal camera footprint, 2x heuristic sigma,
3 m/s sigma growth since the previous prior timestamp, and a minimum 95% valid
map fraction produced 33 accepted fixes, 31 within 30 m, two above 30 m, and
none above 50 m. The original two-tile run gave 37 accepted, 30 within 30 m,
seven above 30 m, including a 239.9 m discrepancy. The 95% crop policy excluded
six frames in region 06 at the map boundary; using a 70% minimum yielded an
accepted 239.6 m discrepancy. These results are exploratory, spatially and
temporally correlated, not calibrated integrity evidence; the footprint was
chosen after observing some of these data. Mean crop area in the conservative
run was about 0.41–0.54 km² depending on region, not necessarily smaller than
the union of two overlapping 500 m tiles. With a 1x sigma margin, regions 02,
03, and 09 had 29/30 within 30 m, matching the fixed-tile count, at about
0.35–0.36 km² mean crop area. These are not held-out tuning results.

The adaptive module can use a supplied camera horizontal FOV and true AGL
height, but evaluation used the explicit fixed footprint because those inputs
are not calibrated in this dataset. Partial-map mode checks the projected centre
is in an available map pixel; it does not validate every match against the map
mask. Rejection on missing map information is not a successful localization.

A visual-anchor diagnostic selects SIFT corners from the map alone and counts
spatially separated areas supported by GIM correspondences. The 4-anchor gate
was selected retrospectively; it is not a semantic landmark detector and its
correlated matches are not independent observations. On 04, a closed-loop gate
left results at 6/10 within 30 m with four larger discrepancies; on 10 it reduced
within-30-m fixes from 6/10 to 3/10, without accepted errors above 30 m in
either run. The gate is disabled by default.

The local results viewer (`scripts/localization_review.py`) indexes saved
frames.csv reports and OrthoLoC sample reports. Candidate IDs are reconstructed
only when recorded priors can be replayed consistently, otherwise only the
recorded best candidate is shown. Lines are recomputed on demand, not a record
of original sampled points; OrthoLoC lines are tentative correspondences, not
saved PnP inliers.


## Dataset coordinate semantics

The [UAV-VisLoc authors](https://github.com/IntelliSensing/UAV-VisLoc) describe
latitude/longitude as the centre of the drone image and describe their dataset
as containing some coordinate errors. The local `README_dataset.txt` makes the
same image-centre claim. This is stronger evidence than an assumption that the
coordinates necessarily denote the aircraft position. It does not establish the
accuracy of any particular frame, the camera calibration, height datum, map
registration, or temporal synchronization. Thus the 06 discrepancy needs an
independent registration/metadata audit, not a blanket claim of either a bad
matcher or an aircraft-to-image-centre offset.

Compact machine-readable results accompanying this report are archived under
[`docs/evidence/visual_localization`](evidence/visual_localization/README.md).

## Oracle label-alignment diagnostic (not a flight test)

Using the true current label solely to select up to two candidate map tiles,
with a 10% visual coverage threshold and no prior jump gate, the later level
segments yielded median matched image-centre discrepancies of 37.1 m (01,
7/10 matched), 10.8 m (02, 10/10), 16.2 m (03, 10/10), 287.5 m (06, 8/10),
and 17.5 m (09, 9/10). True coordinates determine candidate selection; this
is an oracle diagnostic and cannot measure GPS-denied navigation or arrival.

A further retrospective nearest-neighbour search over up to 12 labelled
frames in either direction found that `06_0174`, `06_0175`, and `06_0178`
lay 4.5, 15.2, and 9.4 m from labels `06_0180`, `06_0181`, and `06_0184`,
respectively: six rows later and 14–15 s later according to the recorded dates.
Other frames do not share a consistent six-row offset. For each matched frame
in the four comparison segments (01, 02, 03, 09), the selected nearest label
was the image's own label. Neighbour searching uses future labels and takes the
minimum over as many as 25 candidates, so the distances are optimistic and do
not establish a clock error. No correction was applied. Reproducible evidence:
`docs/evidence/visual_localization/label_alignment/` and the scripts
`audit_visloc_label_alignment.py`, `analyze_label_neighbors.py`.

A strict retrospective proxy in `scripts/summarize_segment_proxy.py` asks
whether all ten frames in each selected segment had an accepted fix within
30 m: two of five pass. This is not an arrival probability. The segments are
short, selected and correlated, and no aircraft/INS/flight-control mission has
been evaluated end to end.
