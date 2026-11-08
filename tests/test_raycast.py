"""Physical and numerical checks for the stellar transit raycaster."""

import numpy as np
import pytest

from shapehunter.shapes.limb_darkening import analytic_center_depth
from shapehunter.shapes.raycast import simulate_transit


def test_center_depth_matches_small_planet_analytic_limit() -> None:
    """A small central planet agrees with the analytic depth within 3%."""
    rp = 0.1
    flux = simulate_transit(np.array([0.0]), np.array([0.0]), np.array([0.0]), params={"rp": rp})
    expected = analytic_center_depth(rp, (0.4, 0.25))
    assert 1.0 - flux[0] == pytest.approx(expected, rel=0.03)


def test_outside_star_is_exactly_unity() -> None:
    """Positions well outside the stellar disc do not block any light."""
    times = np.arange(4.0)
    flux = simulate_transit(times, np.array([-3.0, 3.0, 0.0, 0.0]), np.array([0.0, 0.0, 3.0, -3.0]))
    np.testing.assert_array_equal(flux, np.ones(4))


def test_sphere_transit_is_symmetric_and_bounded() -> None:
    """A centered spherical transit is symmetric and physically bounded."""
    x = np.linspace(-1.4, 1.4, 21)
    flux = simulate_transit(np.arange(x.size), x, np.zeros_like(x), grid=201)
    np.testing.assert_allclose(flux, flux[::-1], atol=1e-12)
    assert np.all((flux >= 0.0) & (flux <= 1.0))
    assert flux[x.size // 2] < 1.0
