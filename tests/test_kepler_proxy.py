"""Offline tests of the Kepler-label exploratory benchmark adapter."""

import csv

import numpy as np
import pytest

from scripts import benchmark_kepler_labels as benchmark


def test_proxy_sampling_and_recording(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stratified sampling is repeatable and uses the supplied assumed cadence."""
    path = tmp_path / "synthetic.csv"
    rng = np.random.default_rng(21)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["LABEL"] + [f"FLUX.{i}" for i in range(30)])
        for label in [2, 1, 1, 1]:
            writer.writerow([label] + list(rng.normal(size=30)))
    observed = []
    def search(curve):
        observed.append(curve.time[1])
        return []
    monkeypatch.setattr(benchmark, "search_star", search)
    first = benchmark.evaluate(path, negatives=2, seed=3, cadence=0.02)
    second = benchmark.evaluate(path, negatives=2, seed=3, cadence=0.02)
    assert first == second
    assert first["summary"]["2"]["evaluated"] == 1
    assert first["summary"]["1"]["evaluated"] == 2
    assert observed == [0.02] * 6
    assert first["summary"]["1"]["interval_95"][1] > 0
    with pytest.raises(ValueError, match="not enough"):
        benchmark.evaluate(path, negatives=4)
