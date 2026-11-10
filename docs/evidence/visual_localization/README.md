# Evidence snapshot: GPS-denied visual localization

This directory preserves compact, per-frame CSVs, run configurations,
summaries, selected query metadata, crop diagnostics, and landmark/OrthoLoC
diagnostics corresponding to the reports in
[GPS_DENIED_STATUS_FA.md](../../GPS_DENIED_STATUS_FA.md) and
[VISUAL_LOCALIZATION_EXPERIMENTS.md](../../VISUAL_LOCALIZATION_EXPERIMENTS.md).

- `level_holdout/`: the five selected later UAV-VisLoc segments, two-tile
  baselines, adaptive-crop variants, and one exploratory jump-gate variant.
- `level_study/`: selected experiments on regions 04, 05 and 10, including
  adaptive crops and the closed-loop visual-anchor gate.
- `decision_examples/`: measured footprints and retrospective landmark support.
- `ortholoc/`: selection and paired results for the OrthoLoC out-of-place test.
- `label_alignment/`: oracle-only diagnostics of label/image consistency; future labels are used in the nearest-neighbor analysis, so this is not navigation.

The numerical outputs preserve the settings and paths recorded at the time;
those paths describe the original local environment, not portable locations.
The source photographs, maps, generated imagery, third-party checkpoints,
map tiles and large datasets are **not included**. In particular, UAV-VisLoc
satellite maps are sourced from Google Maps according to the dataset authors;
use the original dataset and comply with its source terms. The local viewer
indexes `outputs/navlab/` and needs its corresponding imagery on disk; this
snapshot preserves the results even on a clean clone.

This is a research snapshot, not a flight authorization or a statistically
validated probability of mission success. Frames from one flight are correlated.
The 450 m footprint and some acceptance gates were chosen after observing some
of these data, so the comparisons are exploratory. See the reports for
methodology and failure cases.
