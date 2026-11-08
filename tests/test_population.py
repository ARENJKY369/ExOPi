"""Target-level evaluation tests use synthetic curves and offline labels only."""

import pytest

from shapehunter.core import Candidate
from shapehunter.data import generate_lightcurve
from shapehunter.validation import ReferenceTarget, evaluate_population
from shapehunter.validation import population


def test_population_reports_detection_vetting_and_false_alarms(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed-vetting detection is a recovery and a negative false alarm."""
    curves = [generate_lightcurve(seed=i, n_points=400) for i in range(4)]
    targets = [ReferenceTarget("p1", curves[0], "positive", "test catalogue", 3.5),
               ReferenceTarget("p2", curves[1], "positive", "test catalogue", 3.5),
               ReferenceTarget("n1", curves[2], "negative", "test controls"),
               ReferenceTarget("short", curves[3], "positive", "test catalogue", 100)]
    def search(curve, min_power):
        if curve.meta["seed"] == 1:
            return []
        return [Candidate(period=3.5, t0=2, duration=0.1, depth=0.01, snr=12, status="FAIL")]
    monkeypatch.setattr(population, "search_star", search)
    result = evaluate_population(targets)
    assert result["recovery"]["successes"] == 1
    assert result["recovery"]["trials"] == 2
    assert result["vetted_recovery"]["successes"] == 0
    assert result["false_alarm"]["successes"] == 1
    assert result["excluded_or_error"] == 1
    assert result["targets"][-1]["status"] == "excluded"


def test_duplicate_targets_and_missing_provenance_are_rejected() -> None:
    """Repeated stars cannot inflate a target-level sample."""
    curve = generate_lightcurve(seed=7, n_points=100)
    target = ReferenceTarget("id", curve, "negative", "test")
    with pytest.raises(ValueError, match="unique"):
        evaluate_population([target, target])
    with pytest.raises(ValueError, match="source"):
        evaluate_population([ReferenceTarget("id", curve, "negative", "")])


def test_manifest_tracks_input_hash_and_provenance(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local manifest produces a reproducible result without network access."""
    import json
    from scripts.validate_population import evaluate_manifest

    data = tmp_path / "synthetic.csv"
    data.write_text("time,flux\n" + "\n".join(f"{i},1" for i in range(20)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"target_id": "s1", "path": "synthetic.csv", "format": "flux_csv",
                                     "data_source": "synthetic test", "reference_source": "synthetic test",
                                     "label": "negative"}]))
    monkeypatch.setattr(population, "search_star", lambda curve, min_power: [])
    result = evaluate_manifest(manifest)
    assert result["false_alarm"]["trials"] == 1
    assert result["false_alarm"]["successes"] == 0
    assert len(result["files"][0]["sha256"]) == 64
    assert result["files"][0]["data_source"] == "synthetic test"
