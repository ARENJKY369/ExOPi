"""Registry of non-spherical occulter families.

Every family module exposes:

``NAME``            stable string identifier used in reports
``DEFAULT_PARAMS``  dict of shape parameters *excluding* ``rp``
``PARAM_DOC``       human readable parameter description
``extent(params)``  maximum occulter extent in stellar radii
``alpha(dx, dy, params)``  opacity in [0, 1] for offsets from the planet centre
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from shapehunter.shapes.families import (
    binary_planet,
    comet_tail,
    dust_cloud,
    oblate,
    ringed,
    sphere,
)

FAMILIES: Dict[str, Any] = {
    sphere.NAME: sphere,
    ringed.NAME: ringed,
    comet_tail.NAME: comet_tail,
    binary_planet.NAME: binary_planet,
    oblate.NAME: oblate,
    dust_cloud.NAME: dust_cloud,
}


def list_families() -> List[str]:
    """Return the registered family names in deterministic order."""
    return list(FAMILIES)


def get_family(name: str) -> Any:
    """Look up a family module by name (raises ``KeyError`` when unknown)."""
    try:
        return FAMILIES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"unknown shape family {name!r}; available: {sorted(FAMILIES)}"
        ) from exc


def default_params(name: str, rp: float = 0.1) -> Dict[str, float]:
    """Return a full parameter dictionary for ``name`` with planet radius ``rp``."""
    family = get_family(name)
    params: Dict[str, float] = dict(family.DEFAULT_PARAMS)
    params["rp"] = float(rp)
    return params


def build_template(
    name: str,
    params: Mapping[str, float] | None = None,
    n_phase: int = 101,
    grid: int = 201,
    u: Iterable[float] = (0.4, 0.25),
) -> Dict[str, np.ndarray]:
    """Render a normalised transit template for one family.

    Returns a dict with ``phase`` (0..1, mid-transit at 0), ``flux`` and the
    ``params`` actually used. The template is generated with the planet moving
    along the x axis from ``-2`` to ``+2`` stellar radii, which is enough to
    cover every family's contact points.
    """
    from shapehunter.shapes.raycast import simulate_transit

    full = default_params(name, params["rp"]) if params else default_params(name)
    if params:
        full.update({k: float(v) for k, v in params.items()})
    family = get_family(name)
    half_width = float(family.extent(full)) + 1.05
    half_width = max(half_width, 1.2)
    x = np.linspace(-half_width, half_width, int(n_phase))
    flux = simulate_transit(np.zeros_like(x), x, np.zeros_like(x), family=name, params=full, u=tuple(u), grid=grid)
    return {"phase": x / (4.0 * half_width) + 0.5, "x": x, "flux": np.asarray(flux), "params": full}
