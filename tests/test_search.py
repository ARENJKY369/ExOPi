"""Offline checks for transit search and shape-model comparison."""

import numpy as np
import pytest

from shapehunter.data import generate_lightcurve, simulate_planet_transit
from shapehunter.preprocess import prepare_lightcurve
from shapehunter.search import (
    Template, TemplateBank, compare_families, find_single_events, matched_filter,
    run_bls,
)


def test_bls_recovers_seeded_planet() -> None:
    """Frequency resolution is sufficient to recover the planted period."""
    curve = prepare_lightcurve(simulate_planet_transit(generate_lightcurve(seed=7)))
    result = run_bls(curve)
    assert result.period == pytest.approx(3.5, abs=0.01)
    assert result.t0 == pytest.approx(2.0, abs=0.03)
    assert result.snr > 20
    assert result.depth > 0.005


def test_bank_renders_every_family() -> None:
    """Every family is rendered to a finite normalised profile."""
    bank = TemplateBank.build(radii=(0.1,), n_phase=41, grid=101)
    assert len(bank) == 6
    for template in bank:
        assert template.peak_depth > 0
        assert np.isfinite(template.shape).all()
        assert np.max(template.shape) == pytest.approx(1.0)
        assert template.evaluate(np.array([-1.0, 1.0])).tolist() == [0.0, 0.0]


def test_matched_filter_and_bic_select_shape() -> None:
    """A strong asymmetric template wins against a symmetric spherical null."""
    phi = np.linspace(-0.5, 0.5, 101)
    sphere = Template("sphere", {}, phi, np.maximum(1 - 4 * phi**2, 0), 0.01)
    tail = Template("comet_tail", {}, phi, np.maximum(1 - 2 * np.abs(phi + 0.2), 0), 0.01)
    bank = TemplateBank([sphere, tail])
    time = np.linspace(0, 20, 2001)
    phase = ((time - 1 + 1.5) % 3 - 1.5) / 0.6
    flux = 1 - 0.01 * tail.evaluate(phase)
    err = np.full(time.size, 0.0002)
    match = matched_filter(time, flux, err, 3, 1, 0.6, tail)
    assert match.amplitude == pytest.approx(0.01)
    comparison = compare_families(bank, (time, flux), 3, 1, 0.6, flux_err=err)
    assert comparison.family == "comet_tail"
    assert comparison.delta_bic > 10
    comparison = compare_families(bank, (time, 1 - 0.01 * sphere.evaluate(phase)), 3, 1, 0.6, flux_err=err)
    assert comparison.family == "sphere"


def test_single_event_finder_detects_dip() -> None:
    """An isolated dip is found near its known mid-time."""
    rng = np.random.default_rng(4)
    time = np.arange(1000) * 0.01
    flux = 1 + rng.normal(0, 0.0002, time.size)
    flux[np.abs(time - 5) < 0.05] -= 0.01
    events = find_single_events((time, flux), flux_err=np.full(time.size, 0.0002), durations=[0.1], min_snr=8)
    assert len(events) == 1
    assert events[0].t_mid == pytest.approx(5, abs=0.02)
    assert events[0].depth > 0.008


def test_raycast_non_sphere_can_beat_sphere() -> None:
    """A seeded raycast tail gives evidence against a spherical template."""
    curve = prepare_lightcurve(
        simulate_planet_transit(generate_lightcurve(seed=7), family="comet_tail")
    )
    result = run_bls(curve)
    bank = TemplateBank.build(radii=(0.1,), n_phase=61, grid=101)
    comparison = compare_families(bank, curve, result.period, result.t0, result.duration)
    assert comparison.is_non_spherical
    assert comparison.delta_bic > 10


def test_bic_requires_a_spherical_null() -> None:
    """A bank without the comparison model cannot claim a BIC preference."""
    phi = np.linspace(-0.5, 0.5, 11)
    template = Template("ringed", {}, phi, 1 - 2 * np.abs(phi), 0.01)
    time = np.linspace(0, 5, 101)
    with pytest.raises(ValueError, match="sphere"):
        compare_families(TemplateBank([template]), (time, np.ones_like(time)), 2, 1, 0.2)


def test_optimized_shape_matching_recovers_asymmetric_tail() -> None:
    """Shape-specific timing and width avoid misclassifying a raycast tail."""
    curve = prepare_lightcurve(simulate_planet_transit(generate_lightcurve(seed=401), family="comet_tail"))
    peak = run_bls(curve)
    bank = TemplateBank.build(radii=(0.07, 0.10, 0.12), n_phase=81, grid=101)
    comparison = compare_families(bank, curve, peak.period, peak.t0, peak.duration, optimize_ephemeris=True)
    assert comparison.family == "comet_tail"
    assert comparison.is_non_spherical


def test_bls_rejects_only_invalid_trial_durations() -> None:
    """Invalid duration requests are not replaced by an unrelated search."""
    curve = generate_lightcurve(seed=7)
    with pytest.raises(ValueError, match="trial duration"):
        run_bls(curve, period_min=0.5, durations=[1.0, -1.0])
