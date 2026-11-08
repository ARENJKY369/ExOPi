"""Offline checks for disjoint observed-population calibration."""

import pytest

from shapehunter.data import generate_lightcurve
from shapehunter.validation import ReferenceTarget, calibrate_observed_population
from shapehunter.validation import observed_calibration


def test_calibration_uses_disjoint_labelled_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frozen training threshold is passed to the independent holdout."""
    controls = [ReferenceTarget(f"n{i}", generate_lightcurve(seed=i, n_points=100), "negative", "synthetic controls") for i in range(3)]
    holdout = [ReferenceTarget("n4", generate_lightcurve(seed=4, n_points=100), "negative", "synthetic controls"),
               ReferenceTarget("p5", generate_lightcurve(seed=5, n_points=100), "positive", "synthetic labels", 0.8)]
    monkeypatch.setattr(observed_calibration, "observed_screening_score", lambda curve: float(curve.meta["seed"] + 1))
    calls = []
    def evaluate(targets, min_power, period_tolerance):
        calls.append(min_power)
        return {"false_alarm": {"interval_95": (0.0, 0.9)}, "excluded_or_error": 0}
    monkeypatch.setattr(observed_calibration, "evaluate_population", evaluate)
    result = calibrate_observed_population(controls, holdout, target_false_alarm_rate=0.5)
    assert result["training_rank"] == 2
    assert result["threshold"] > 2
    assert calls == [result["threshold"]]
    assert result["numeric_false_alarm_goal_met"] is False
    assert len(result["holdout_curve_sha256"]) == 2
    assert controls[0].target_id in [row["target_id"] for row in result["training_controls"]]


def test_calibration_rejects_data_leakage_and_insufficient_sample() -> None:
    """Reused stars or arrays and under-resolved target rates cannot be calibrated."""
    control = ReferenceTarget("n1", generate_lightcurve(seed=1, n_points=100), "negative", "synthetic")
    copied = ReferenceTarget("different-name", control.curve.copy(), "negative", "synthetic")
    positive = ReferenceTarget("p2", generate_lightcurve(seed=2, n_points=100), "positive", "synthetic", 0.8)
    with pytest.raises(ValueError, match="duplicate light-curve"):
        calibrate_observed_population([control], [copied, positive], target_false_alarm_rate=0.5)
    negative = ReferenceTarget("n3", generate_lightcurve(seed=3, n_points=100), "negative", "synthetic")
    with pytest.raises(ValueError, match="not enough training"):
        calibrate_observed_population([control], [negative, positive], target_false_alarm_rate=0.05)
