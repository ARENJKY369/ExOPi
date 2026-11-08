"""Deterministic offline checks for the proxy-only calibration workflow."""

import csv

import numpy as np
import pytest

from scripts import calibrate_kepler_proxy as calibration


def test_disjoint_proxy_calibration(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training, held-out negatives, and excluded rows never overlap."""
    path = tmp_path / "synthetic.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["LABEL"] + [f"FLUX.{i}" for i in range(12)])
        for index in range(15):
            writer.writerow([2 if index == 0 else 1] + [index + j for j in range(12)])
    monkeypatch.setattr(calibration, "score_row", lambda values, cadence: float(values[0] + 1))
    result = calibration.calibrate(path, [1, 2], training_size=6, holdout_size=4, target_rate=0.5, seed=10)
    train = {row["row_index"] for row in result["training_scores"]}
    holdout = {row["row_index"] for row in result["label_1_holdout"]}
    assert not train & holdout
    assert not (train | holdout) & {1, 2}
    assert result["label_1_summary"]["evaluated"] == 4
    assert result["label_2_summary"]["evaluated"] == 1
    assert result == calibration.calibrate(path, [1, 2], training_size=6, holdout_size=4, target_rate=0.5, seed=10)
    with pytest.raises(ValueError, match="label-1"):
        calibration.calibrate(path, [0], training_size=6, holdout_size=4, target_rate=0.5)


def test_training_error_is_not_silently_ignored(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid training samples abort rather than biasing the threshold."""
    path = tmp_path / "synthetic.csv"
    path.write_text("LABEL," + ",".join(f"FLUX.{i}" for i in range(12)) + "\n" +
                    "\n".join(",".join([str(label)] + ["1"] * 12) for label in [2, 1, 1, 1, 1, 1]))
    def fail(values: np.ndarray, cadence: float) -> float:
        raise ValueError("bad data")
    monkeypatch.setattr(calibration, "score_row", fail)
    with pytest.raises(ValueError, match="bad data"):
        calibration.calibrate(path, [], training_size=3, holdout_size=2, target_rate=0.5)
