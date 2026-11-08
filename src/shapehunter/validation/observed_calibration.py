"""Holdout-based threshold calibration for user-supplied labelled observations."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.preprocess import prepare_lightcurve
from shapehunter.search import run_bls
from shapehunter.validation.population import ReferenceTarget, evaluate_population


def _curve_digest(curve: LightCurve) -> str:
    """Fingerprint the time and flux arrays to catch exact duplicate inputs."""
    digest = hashlib.sha256()
    for value in (curve.time, curve.flux):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(array.tobytes())
    return digest.hexdigest()


def observed_screening_score(curve: LightCurve) -> float:
    """Return the pipeline BLS screening score, or zero for a rejected fit."""
    clean = prepare_lightcurve(curve)
    if len(clean) < 30 or clean.baseline <= 1.0:
        raise ValueError("light curve is too short for the pipeline search")
    result = run_bls(clean)
    if (not np.isfinite(result.period) or result.period <= 0 or
            not np.isfinite(result.duration) or not 0 < result.duration < result.period / 2 or
            not np.isfinite(result.power) or not np.isfinite(result.snr)):
        return 0.0
    return float(max(0.0, min(result.power, result.snr)))


def calibrate_observed_population(
    training_controls: Sequence[ReferenceTarget],
    holdout_targets: Sequence[ReferenceTarget],
    target_false_alarm_rate: float = 0.05,
    period_tolerance: float = 0.01,
) -> dict:
    """Fit a threshold on labelled controls and evaluate a disjoint holdout.

    Input labels, independent provenance and representativeness are the caller's
    responsibility; this function cannot verify them. Training must contain
    enough distinct negative controls for the order statistic, and holdout must
    contain positive and negative targets. Identical arrays or IDs are rejected
    across both sets; different segments from one star still require external
    identity checks. Training errors abort. Holdout errors are recorded and
    prevent the numerical false-alarm goal from being reported as met.
    """
    training, holdout = list(training_controls), list(holdout_targets)
    if not 0 < target_false_alarm_rate < 1 or not 0 < period_tolerance < 1:
        raise ValueError("rates and tolerance must lie strictly between 0 and 1")
    if not training or any(t.label != "negative" or t.expected_period is not None for t in training):
        raise ValueError("training must contain labelled negative controls only")
    if not any(t.label == "positive" for t in holdout) or not any(t.label == "negative" for t in holdout):
        raise ValueError("holdout requires both positive and negative labels")
    targets = training + holdout
    ids = [t.target_id.strip().casefold() for t in targets]
    if any(not name for name in ids) or len(ids) != len(set(ids)):
        raise ValueError("target IDs must be nonempty and unique across both splits")
    if any(not t.reference_source.strip() for t in targets):
        raise ValueError("each target needs an independent reference source")
    digests = [_curve_digest(t.curve) for t in targets]
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate light-curve arrays across targets or splits")
    rank = math.ceil((len(training) + 1) * (1 - target_false_alarm_rate))
    if rank > len(training):
        raise ValueError("not enough training controls to resolve the target rate")
    scores = []
    for target, digest in zip(training, digests):
        # Errors abort calibration, rather than being removed from its sample.
        value = observed_screening_score(target.curve)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid screening score for {target.target_id}")
        scores.append({"target_id": target.target_id, "reference_source": target.reference_source,
                       "curve_sha256": digest, "score": float(value)})
    threshold = float(np.nextafter(sorted(row["score"] for row in scores)[rank - 1], np.inf))
    holdout_result = evaluate_population(holdout, min_power=threshold, period_tolerance=period_tolerance)
    false_alarm = holdout_result["false_alarm"]
    interval = false_alarm["interval_95"]
    goal_met = (holdout_result["excluded_or_error"] == 0 and interval is not None
                and interval[1] <= target_false_alarm_rate)
    return {
        "threshold": threshold, "target_false_alarm_rate": target_false_alarm_rate,
        "training_rank": rank, "training_controls": scores,
        "holdout_curve_sha256": {t.target_id: digest for t, digest in zip(holdout, digests[len(training):])},
        "holdout": holdout_result, "numeric_false_alarm_goal_met": bool(goal_met),
        "note": "Numeric holdout results do not establish survey representativeness or verified labels; no default threshold is changed.",
    }
