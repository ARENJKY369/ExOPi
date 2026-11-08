"""Vetting checks on deterministic synthetic planet and binary systems."""

import numpy as np
import pytest

from shapehunter.core import Candidate
from shapehunter.data import generate_lightcurve, simulate_eclipsing_binary, simulate_planet_transit
from shapehunter.preprocess import prepare_lightcurve
from shapehunter.vetting import vet_candidate


def test_clean_planet_passes_and_binary_fails() -> None:
    """A clean planet passes; a system with a secondary eclipse fails."""
    planet = prepare_lightcurve(simulate_planet_transit(generate_lightcurve(seed=7)))
    p = Candidate(period=3.5, t0=2.0, duration=0.14, depth=0.01, snr=50)
    planet_result = vet_candidate(planet, p)
    assert planet_result["status"] == "PASS"
    assert planet_result["checks"]["known_objects"]["status"] == "unknown"
    binary = prepare_lightcurve(simulate_eclipsing_binary(seed=11))
    eb = Candidate(period=2.4, t0=0.7, duration=0.18, depth=0.06, snr=50)
    result = vet_candidate(binary, eb)
    assert result["status"] == "FAIL"
    assert "secondary_eclipse" in result["failed"]


def test_systematics_and_known_ephemerides() -> None:
    """Supplied flags and catalogue information are evaluated explicitly."""
    planet = prepare_lightcurve(simulate_planet_transit(generate_lightcurve(seed=7)))
    p = Candidate(period=3.5, t0=2, duration=0.14, depth=0.01, snr=50)
    result = vet_candidate(planet, p, known_objects=[{"period": 3.5, "t0": 2.0}], systematics_mask=np.ones(len(planet), dtype=bool))
    assert result["checks"]["known_objects"]["value"] == 1
    assert result["checks"]["systematics"]["status"] == "fail"
    with pytest.raises(ValueError):
        vet_candidate(planet, p, systematics_mask=np.ones(2, dtype=bool))


def test_missing_checks_mark_vetting_incomplete() -> None:
    """A PASS with unavailable ancillary checks is explicitly incomplete."""
    planet = prepare_lightcurve(simulate_planet_transit(generate_lightcurve(seed=7)))
    result = vet_candidate(planet, Candidate(period=3.5, t0=2, duration=0.14, depth=0.01, snr=50))
    assert result["status"] == "PASS"
    assert result["completeness"] == "incomplete"
    assert {"known_objects", "systematics"} <= set(result["unknown"])


@pytest.mark.parametrize("seed", [5, 11, 12])
def test_odd_even_handles_uneven_ingress_sampling(seed: int) -> None:
    """Uneven sampling of an unchanged transit does not mimic odd/even depths."""
    from shapehunter.search import run_bls

    curve = prepare_lightcurve(simulate_planet_transit(generate_lightcurve(seed=seed)))
    peak = run_bls(curve)
    candidate = Candidate(period=peak.period, t0=peak.t0, duration=peak.duration, depth=peak.depth, snr=peak.snr)
    assert vet_candidate(curve, candidate)["checks"]["odd_even"]["status"] == "pass"


def test_odd_even_detects_alternating_depths() -> None:
    """Genuinely different odd and even eclipses are rejected."""
    from shapehunter.core import LightCurve

    time = np.arange(0, 50, 0.01)
    phase = (time - 1 + 1.5) % 3 - 1.5
    epoch = np.rint((time - 1) / 3).astype(int)
    flux = 1 - np.where(np.abs(phase) < 0.08, np.where(epoch % 2, 0.02, 0.01), 0.0)
    flux += np.random.default_rng(8).normal(0, 0.0003, len(time))
    result = vet_candidate(LightCurve(time, flux, np.full(len(time), 0.0003)), Candidate(period=3, t0=1, duration=0.16, depth=0.01, snr=50))
    assert result["checks"]["odd_even"]["status"] == "fail"
