"""Exercise every registered occulter through the public raycaster."""

import numpy as np
import pytest

from shapehunter.shapes import FAMILIES, simulate_transit
from shapehunter.shapes.families import default_params, get_family


NAMES = ("sphere", "ringed", "comet_tail", "binary_planet", "oblate", "dust_cloud")


def test_registry_has_six_named_families() -> None:
    """The public registry exposes all required models."""
    assert set(FAMILIES) == set(NAMES)
    with pytest.raises(KeyError):
        get_family("unknown")


@pytest.mark.parametrize("family", NAMES)
def test_family_renders_bounded_nonzero_transit(family: str) -> None:
    """Each family blocks light centrally and is transparent far away."""
    params = default_params(family, rp=0.1)
    flux = simulate_transit(
        np.array([0.0, 1.0, 2.0]),
        np.array([-5.0, 0.0, 5.0]),
        np.zeros(3),
        family=family,
        params=params,
        grid=201,
    )
    assert flux.shape == (3,)
    assert np.all(np.isfinite(flux))
    assert 0.0 <= flux[1] < 1.0
    assert flux[0] == flux[2] == 1.0


def test_distinct_shapes_produce_distinct_depths() -> None:
    """Ringed and oblate defaults are not aliases of the sphere."""
    def depth(name: str) -> float:
        return 1.0 - simulate_transit(
            np.zeros(1), np.zeros(1), np.zeros(1), family=name,
            params=default_params(name, rp=0.1), grid=201,
        )[0]

    spherical = depth("sphere")
    assert depth("ringed") > spherical
    assert depth("oblate") < spherical


def test_unflattened_oblate_is_physically_identical_to_sphere() -> None:
    """Zero projected flattening is unidentifiable from a spherical occulter."""
    x = np.linspace(-1.2, 1.2, 13)
    args = (np.zeros_like(x), x, np.zeros_like(x))
    sphere = simulate_transit(*args, family="sphere", params={"rp": 0.1}, grid=101)
    oblate = simulate_transit(*args, family="oblate", params={"rp": 0.1, "flattening": 0.0}, grid=101)
    np.testing.assert_array_equal(sphere, oblate)
