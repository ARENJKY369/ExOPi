"""Family 6/6: a dusty, partially transparent cloud around a small core."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shapehunter.shapes.families.base import disk, param

NAME = "dust_cloud"
DEFAULT_PARAMS: dict[str, float] = {
    "cloud_radius": 2.5,
    "cloud_opacity": 0.3,
}
PARAM_DOC = (
    "rp: opaque core radius; cloud_radius: Gaussian scale radius of the dust in "
    "units of rp; cloud_opacity: peak dust opacity"
)


def extent(params: Mapping[str, float]) -> float:
    """The Gaussian halo is truncated at three scale radii."""
    rp = max(param(params, "rp", 0.1), 0.0)
    if param(params, "cloud_opacity", 0.3) <= 0.0:
        return rp
    return max(rp, 3.0 * param(params, "cloud_radius", 2.5) * rp)


def alpha(dx: np.ndarray, dy: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Opaque core plus a truncated Gaussian dust halo."""
    rp = max(param(params, "rp", 0.1), 0.0)
    core = disk(dx, dy, rp)
    opacity = float(np.clip(param(params, "cloud_opacity", 0.3), 0.0, 1.0))
    if opacity <= 0.0 or rp <= 0.0:
        return core
    scale = max(param(params, "cloud_radius", 2.5) * rp, 1e-6)
    rr = np.hypot(dx, dy)
    halo = opacity * np.exp(-0.5 * (rr / scale) ** 2)
    halo = np.where(rr <= 3.0 * scale, halo, 0.0)
    return np.maximum(core, np.clip(halo, 0.0, 1.0))
