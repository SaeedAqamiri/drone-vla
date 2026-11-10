from __future__ import annotations

import math
import statistics
from typing import Any

from .types import FrameResult


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def _clean(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]


def compute_summary(
    frames: list[FrameResult],
    config: Any | None = None,
) -> dict[str, Any]:
    total = len(frames)
    attempted = [frame for frame in frames if frame.attempted]
    accepted = [frame for frame in frames if frame.accepted]
    matched = [frame for frame in frames if frame.matched]
    errors = _clean([frame.error_m for frame in accepted])
    prior_errors = _clean([frame.prior_error_m for frame in attempted])
    latencies = [frame.latency_ms for frame in attempted]
    max_error = float(getattr(config, "max_error_m", 30.0))

    correct = [error for error in errors if error <= max_error]
    wrong = [error for error in errors if error > max_error]
    coverage = [frame.candidate_truth_covered for frame in attempted
                if frame.candidate_truth_covered is not None]
    timestamps = [frame.timestamp for frame in frames]
    accepted_times = [frame.timestamp for frame in accepted]
    labelled_attempts = [frame for frame in attempted if frame.truth_east is not None and frame.truth_north is not None]
    boundaries = ([timestamps[0]] + accepted_times + [timestamps[-1]]) if timestamps else []
    longest_gap = max((b - a for a, b in zip(boundaries, boundaries[1:])), default=None)
    first_fix = (
        accepted_times[0] - timestamps[0] if accepted_times and timestamps else None
    )
    fix_intervals = [
        accepted_times[index] - accepted_times[index - 1]
        for index in range(1, len(accepted_times))
    ]

    dropout_start = getattr(config, "dropout_start", None)
    dropout_length = int(getattr(config, "dropout_length", 0) or 0)
    recovery_s = None
    if dropout_start is not None and dropout_length > 0:
        end_index = dropout_start + dropout_length
        after = [frame for frame in accepted if frame.index >= end_index]
        if after and 0 <= end_index < total:
            recovery_s = after[0].timestamp - frames[end_index].timestamp

    return {
        "frames": total,
        "attempted": len(attempted),
        "matched": len(matched),
        "accepted": len(accepted),
        "rejected": len(attempted) - len(accepted),
        "match_rate": len(matched) / max(len(attempted), 1),
        "fix_rate": len(accepted) / max(total, 1),
        "accepted_rate": len(accepted) / max(len(attempted), 1),
        # A fix without ground truth is unlabelled, not a false fix. Also,
        # precision is undefined when no accepted fix has a known error.
        "ground_truth_accepted": len(errors),
        "correct_fixes": len(correct),
        "false_fixes": len(wrong),
        "candidate_truth_coverage": sum(coverage) / len(coverage) if coverage else None,
        "labelled_attempts": len(labelled_attempts),
        "fix_within_5m": sum(f.error_m is not None and f.error_m <= 5 for f in accepted) / len(labelled_attempts) if labelled_attempts else None,
        "fix_within_10m": sum(f.error_m is not None and f.error_m <= 10 for f in accepted) / len(labelled_attempts) if labelled_attempts else None,
        "fix_within_20m": sum(f.error_m is not None and f.error_m <= 20 for f in accepted) / len(labelled_attempts) if labelled_attempts else None,
        "fix_within_30m": sum(f.error_m is not None and f.error_m <= 30 for f in accepted) / len(labelled_attempts) if labelled_attempts else None,
        "precision": len(correct) / len(errors) if errors else None,
        "false_fix_rate": len(wrong) / len(errors) if errors else None,
        "mean_error_m": statistics.mean(errors) if errors else None,
        "median_error_m": statistics.median(errors) if errors else None,
        "p95_error_m": _percentile(errors, 0.95),
        "max_error_m": max(errors) if errors else None,
        "mean_prior_error_m": statistics.mean(prior_errors) if prior_errors else None,
        "p95_prior_error_m": _percentile(prior_errors, 0.95),
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "mean_candidates": statistics.mean(
            [frame.candidates for frame in attempted]
        )
        if attempted
        else 0.0,
        "time_to_first_fix_s": first_fix,
        "mean_fix_interval_s": statistics.mean(fix_intervals)
        if fix_intervals
        else None,
        "longest_gap_without_fix_s": longest_gap,
        "recovery_s": recovery_s,
        "max_error_threshold_m": max_error,
        "failure_counts": _failure_counts(frames),
    }


def _failure_counts(frames: list[FrameResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        if frame.accepted:
            continue
        counts[frame.reason] = counts.get(frame.reason, 0) + 1
    return counts


def format_summary(summary: dict[str, Any]) -> str:
    def number(key: str, digits: int = 2) -> str:
        value = summary.get(key)
        if value is None:
            return "-"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return "-"
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        f"frames={summary['frames']} attempted={summary['attempted']} "
        f"accepted={summary['accepted']} matched={summary['matched']}",
        f"fix_rate={number('fix_rate', 3)} accepted_rate={number('accepted_rate', 3)} "
        f"precision={number('precision', 3)} false_fix_rate={number('false_fix_rate', 3)}",
        f"error mean={number('mean_error_m')} m median={number('median_error_m')} m "
        f"p95={number('p95_error_m')} m max={number('max_error_m')} m",
        f"prior_error mean={number('mean_prior_error_m')} m "
        f"p95={number('p95_prior_error_m')} m",
        f"latency mean={number('mean_latency_ms')} ms p95={number('p95_latency_ms')} ms "
        f"candidates={number('mean_candidates')}",
        f"first_fix={number('time_to_first_fix_s')} s "
        f"fix_interval={number('mean_fix_interval_s')} s "
        f"recovery={number('recovery_s')} s",
    ]
    return "\n".join(lines)
