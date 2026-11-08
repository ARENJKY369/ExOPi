"""Family 4/6: two planets sharing a barycentre (a close binary planet)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shapehunter.shapes.families.base import disk, param

NAME = "binary_planet"
DEFAULT_PARAMS: dict[str, float] = {
    "separation": 3.5,
    "rp2": 0.0,
    "binary_angle": 0.0,
}
PARAM_DOC = (
    "rp: primary radius; rp2: secondary radius (0 means 0.6*rp); "
    "separation: centre-to-centre distance in units of rp; "
    "binary_angle: orientation of the pair in radians"
)


def _radii(params: Mapping[str, float]) -> tuple[float, float]:
    rp = max(param(params, "rp", 0.1), 0.0)
    rp2 = param(params, "rp2", 0.0)
    if rp2 <= 0.0:
        rp2 = 0.6 * rp
    return rp, max(rp2, 0.0)


def extent(params: Mapping[str, float]) -> float:
    """Barycentric half-separation plus the larger companion."""
    rp, rp2 = _radii(params)
    sep = max(param(params, "separation", 3.5), 0.0) * rp
    return float(0.5 * sep + max(rp, rp2))


def alpha(dx: np.ndarray, dy: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Union of the two opaque discs."""
    rp, rp2 = _radii(params)
    sep = max(param(params, "separation", 3.5), 0.0) * rp
    angle = param(params, "binary_angle", 0.0)
    half = 0.5 * sep
    ca, sa = np.cos(angle), np.sin(angle)
    primary = disk(dx - half * ca, dy - half * sa, rp)
    secondary = disk(dx + half * ca, dy + half * sa, rp2)
    return np.maximum(primary, secondary)
