"""Deterministic tests of dynamics, fitting, and transit timing."""

import numpy as np
import pytest

from shapehunter.core import LightCurve
from shapehunter.data.synthetic import trapezoid_eclipse
from shapehunter.fitting.least_squares import fit_trapezoid
from shapehunter.fitting.models import circle_overlap_area, trapezoid_flux
from shapehunter.physics import extract_ttv, total_energy, velocity_verlet


def test_verlet_energy_over_100_orbits() -> None:
    """A barycentric circular binary conserves energy over 100 orbits."""
    masses = np.array([1.0, 0.001])
    omega = np.sqrt(masses.sum())
    positions = np.array([[-masses[1] / masses.sum(), 0], [masses[0] / masses.sum(), 0]])
    velocities = np.column_stack((np.zeros(2), omega * positions[:, 0]))
    period = 2 * np.pi / omega
    xs, vs = velocity_verlet(positions, velocities, masses, period / 500, 50000)
    initial = total_energy(xs[0], vs[0], masses)
    energies = np.array([total_energy(xs[i], vs[i], masses) for i in range(0, len(xs), 100)])
    assert np.max(np.abs((energies - initial) / initial)) < 1e-6


def test_circle_overlap_partial_and_vectorized() -> None:
    """Two unit circles at unit separation have the known lens area."""
    areas = circle_overlap_area(np.array([0.0, 1.0, 2.0]), 1.0)
    np.testing.assert_allclose(areas, [np.pi, 2*np.pi/3 - np.sqrt(3)/2, 0.0])


def test_trapezoid_fit_with_fixed_period() -> None:
    """Fixed-period least squares recovers a noiseless transit."""
    time = np.linspace(0, 20, 4001)
    flux = trapezoid_flux(time, 3.5, 2.0, 0.01, 0.2, 0.3)
    fit = fit_trapezoid((time, flux), 3.5, 2.01, 0.21, 0.009, fit_period=False)
    assert fit.success
    assert fit.params["period"] == 3.5
    assert fit.params["depth"] == pytest.approx(0.01, rel=0.01)
    assert fit.params["t0"] == pytest.approx(2, abs=0.001)


def test_ttv_extracts_injected_offsets() -> None:
    """Independent mid-times recover known alternating transit offsets."""
    time = np.arange(0, 16, 0.01)
    flux = np.ones_like(time)
    for epoch in range(5):
        offset = 0.02 if epoch % 2 else -0.02
        flux -= trapezoid_eclipse(time - (1 + 3*epoch + offset), 0.01, 0.3, 0.3)
    curve = LightCurve(time, flux, np.full(time.size, 0.0002))
    events = extract_ttv(curve, 3, 1, 0.3, 0.01)
    assert len(events) == 5
    for event in events:
        assert event["oc"] == pytest.approx(0.02 if event["epoch"] % 2 else -0.02, abs=0.002)
