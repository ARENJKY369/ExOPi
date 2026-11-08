"""End-to-end offline CLI and results database checks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from shapehunter.core import Candidate
from shapehunter.pipeline.results_db import load_candidates, store_candidates


@pytest.mark.parametrize("noise", [False, True])
def test_synthetic_cli(tmp_path: Path, noise: bool) -> None:
    """The CLI recovers the planet and returns no result on seeded noise."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    command = [sys.executable, "-m", "shapehunter.pipeline.run_star", "--synthetic", "--seed", "7", "--out", str(tmp_path)]
    if noise:
        command.append("--no-injection")
    subprocess.run(command, check=True, env=env, capture_output=True, text=True)
    data = json.loads((tmp_path / "candidates.json").read_text())
    assert (tmp_path / "report.txt").exists()
    if noise:
        assert data == []
    else:
        assert len(data) == 1
        assert data[0]["period"] == pytest.approx(3.5, abs=0.01)
        assert {"period", "t0", "duration", "depth", "snr", "family", "params", "vetting", "status"} <= data[0].keys()


def test_database_replaces_run(tmp_path: Path) -> None:
    """SQLite round trips candidates and replacement clears stale results."""
    path = tmp_path / "results.sqlite"
    candidate = Candidate(period=3.5, t0=2, duration=0.14, depth=0.01, snr=30)
    store_candidates(path, "a", [candidate])
    assert load_candidates(path, "a")[0].period == 3.5
    store_candidates(path, "a", [])
    assert load_candidates(path, "a") == []


def test_batch_writes_database(tmp_path: Path) -> None:
    """Batch mode processes saved inputs into indexed output directories."""
    from shapehunter.data import generate_lightcurve, save_lightcurve
    from shapehunter.pipeline.batch import run_batch

    source = tmp_path / "star.npz"
    save_lightcurve(generate_lightcurve(seed=7), source)
    output = tmp_path / "batch"
    run_batch([source], output)
    assert json.loads((output / "0000-star" / "candidates.json").read_text()) == []
    assert load_candidates(output / "results.sqlite", "0000-star") == []


def test_scheduler_runs_only_due_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The caller-driven scheduler runs due batches and returns future jobs."""
    from datetime import datetime, timedelta
    from shapehunter.pipeline import scheduler

    now = datetime(2026, 1, 1)
    due = scheduler.ScheduledBatch(now, (tmp_path / "in.npz",), tmp_path / "out")
    future = scheduler.ScheduledBatch(now + timedelta(days=1), (), tmp_path / "later")
    calls = []
    monkeypatch.setattr(scheduler, "run_batch", lambda inputs, output: calls.append((inputs, output)))
    assert scheduler.run_due([future, due], now) == [future]
    assert calls == [([tmp_path / "in.npz"], tmp_path / "out")]


def test_iterative_search_recovers_two_planets() -> None:
    """Masking the first transit allows recovery of an independent second one."""
    from shapehunter.data import generate_lightcurve, simulate_planet_transit
    from shapehunter.pipeline.run_star import search_star

    curve = generate_lightcurve(seed=7)
    curve = simulate_planet_transit(curve, period=3.5, t0=2, rp=0.1)
    curve = simulate_planet_transit(curve, period=5.2, t0=1, rp=0.08)
    candidates = search_star(curve, max_candidates=2)
    assert len(candidates) == 2
    periods = [candidate.period for candidate in candidates]
    assert any(abs(period - 3.5) < 0.03 for period in periods)
    assert any(abs(period - 5.2) < 0.05 for period in periods)
    assert "sphere" in candidates[0].meta["family_chi2"]


def test_multi_candidate_noise_does_not_create_signal() -> None:
    """Iterative mode still emits no candidate for the seeded noise control."""
    from shapehunter.data import generate_lightcurve
    from shapehunter.pipeline.run_star import search_star

    assert search_star(generate_lightcurve(seed=7), max_candidates=2) == []


def test_nearly_degenerate_shapes_are_marked_ambiguous() -> None:
    """An oblate injection indistinguishable from a sphere is not claimed definite."""
    from shapehunter.data import generate_lightcurve, simulate_planet_transit
    from shapehunter.pipeline.run_star import search_star

    curve = simulate_planet_transit(generate_lightcurve(seed=401), family="oblate")
    candidate = search_star(curve)[0]
    assert candidate.meta["shape_classification"] == "ambiguous"
    assert candidate.meta["shape_score_gap"] < 10


def test_oblate_sphere_alternatives_are_visible() -> None:
    """A near-degenerate oblate injection reports both plausible shapes."""
    from shapehunter.data import generate_lightcurve, simulate_planet_transit
    from shapehunter.pipeline.run_star import search_star

    candidate = search_star(simulate_planet_transit(generate_lightcurve(seed=401), family="oblate"))[0]
    assert {"sphere", "oblate"} <= set(candidate.meta["plausible_shapes"])
    assert candidate.meta["shape_classification"] == "ambiguous"
    assert candidate.meta["oblate_sphere_decision"] == "inconclusive"


def test_search_rejects_invalid_refined_transit_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pathological long BLS fit is rejected instead of crashing matching."""
    from types import SimpleNamespace
    from shapehunter.data import generate_lightcurve
    from shapehunter.pipeline import run_star

    monkeypatch.setattr(run_star, "run_bls", lambda curve: SimpleNamespace(power=20., snr=20., period=1., duration=0.6))
    assert run_star.search_star(generate_lightcurve(seed=3), max_candidates=1) == []


def test_report_warns_screening_not_calibrated(tmp_path: Path) -> None:
    """Candidate reports do not present an empirical threshold as calibrated."""
    from shapehunter.report.summary import write_report

    candidate = Candidate(period=3.5, t0=2, duration=0.1, depth=0.01, snr=10,
                          meta={"screening_calibration": "uncalibrated"})
    report = write_report([candidate], tmp_path / "report.txt").read_text()
    assert "not calibrated for real observations" in report


def test_zero_flattening_does_not_claim_distinction() -> None:
    """A spherical projection is explicitly inconclusive against a sphere."""
    from shapehunter.data import generate_lightcurve, simulate_planet_transit
    from shapehunter.pipeline.run_star import search_star

    curve = simulate_planet_transit(generate_lightcurve(seed=7), family="oblate", params={"flattening": 0.0})
    candidate = search_star(curve)[0]
    assert candidate.meta["oblate_sphere_decision"] == "inconclusive"
    assert {"sphere", "oblate"} <= set(candidate.meta["plausible_shapes"])
