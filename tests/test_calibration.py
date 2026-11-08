"""Synthetic calibration, holdout isolation and shape-confusion checks."""

import numpy as np
import pytest

from shapehunter.core import Candidate
from shapehunter.validation import binomial_interval, calibrate_threshold, evaluate_holdout, shape_confusion
from shapehunter.validation import calibration, shapes


def test_threshold_rank_and_independent_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen threshold is evaluated only on disjoint trial seeds."""
    monkeypatch.setattr(calibration, "screening_score", lambda seed: float(seed))
    cal = calibrate_threshold([1, 2, 3], target_rate=0.5)
    assert cal["threshold"] > 2 and cal["threshold"] < 3
    seen = []
    def search(curve, min_power):
        seen.append((curve.meta["seed"], min_power))
        return [Candidate(period=3.5, t0=2, duration=0.1, depth=0.01, snr=20, status="PASS")] if "injected" in curve.meta else []
    monkeypatch.setattr(calibration, "search_star", search)
    outcome = evaluate_holdout(cal, [4], [5])
    assert outcome["false_alarm_fraction"] == 0
    assert outcome["recovery_fraction"] == 1
    assert outcome["false_alarm_interval_95"][1] > 0.9
    assert all(value == cal["threshold"] for _, value in seen)
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_holdout(cal, [1], [5])
    with pytest.raises(ValueError, match="not enough"):
        calibrate_threshold([1], target_rate=0.05)


def test_exact_binomial_interval() -> None:
    """Zero detections have a positive upper bound, not proof of zero risk."""
    low, high = binomial_interval(0, 20)
    assert low == 0
    assert high == pytest.approx(1 - 0.025 ** (1 / 20))


def test_shape_confusion_counts_predictions_and_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Classifications and missing period matches populate distinct bins."""
    def search(curve, min_power):
        if curve.meta["seed"] == 2:
            return []
        return [Candidate(period=3.5, t0=2, duration=0.1, depth=0.01, snr=20, family="sphere")]
    monkeypatch.setattr(shapes, "search_star", search)
    result = shape_confusion([1, 2], families=["sphere", "ringed"])
    assert result["matrix"]["sphere"]["sphere"] == 1
    assert result["matrix"]["ringed"]["sphere"] == 1
    assert result["matrix"]["ringed"]["missed"] == 1
    assert len(result["trials"]) == 4
