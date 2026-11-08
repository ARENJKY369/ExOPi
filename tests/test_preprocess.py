"""Check one-sided flare removal and transit preservation."""

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.data import generate_lightcurve, simulate_planet_transit
from shapehunter.preprocess import clip_flares, find_flares, prepare_lightcurve


def test_flares_are_one_sided() -> None:
    """A bright spike is clipped while an equally deep dip survives."""
    rng = np.random.default_rng(42)
    time = np.arange(500) * 0.02
    flux = 1.0 + rng.normal(0.0, 0.0002, time.size)
    flux[150] += 0.05
    flux[350] -= 0.05
    lc = LightCurve(time=time, flux=flux, flux_err=np.full(time.size, 0.0002))
    flags = find_flares(time, flux)
    assert flags[150]
    assert not flags[350]
    clipped = clip_flares(lc)
    assert abs(clipped.flux[150] - 1.0) < 0.002
    assert clipped.flux[350] == flux[350]
    assert lc.flux[150] == flux[150]


def test_preparation_preserves_injected_transit() -> None:
    """Detrending suppresses slow trends without erasing a planet transit."""
    raw = simulate_planet_transit(generate_lightcurve(seed=7), period=3.5, t0=2.0)
    clean = prepare_lightcurve(raw)
    offset = (clean.time - 2.0 + 1.75) % 3.5 - 1.75
    inside = np.abs(offset) < 0.025
    outside = np.abs(offset) > 0.2
    assert inside.sum() > 10
    assert np.median(clean.flux[outside]) - np.median(clean.flux[inside]) > 0.008
    assert np.std(clean.flux[outside]) < 0.001


def test_preparation_propagates_multiplicative_uncertainty() -> None:
    """Scaling absolute flux and errors leaves prepared data unchanged."""
    original = generate_lightcurve(seed=7)
    scaled = original.copy()
    scaled.flux *= 1000.0
    scaled.flux_err *= 1000.0
    first = prepare_lightcurve(original)
    second = prepare_lightcurve(scaled)
    np.testing.assert_allclose(second.time, first.time)
    np.testing.assert_allclose(second.flux, first.flux, rtol=1e-11)
    np.testing.assert_allclose(second.flux_err, first.flux_err, rtol=1e-11)
