"""Family 2/6: a planet with a broad, optically thin ring system."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shapehunter.shapes.families.base import disk, param

NAME = "ringed"
DEFAULT_PARAMS: dict[str, float] = {
    "ring_inner": 1.4,
    "ring_outer": 2.3,
    "ring_opacity": 0.5,
    "ring_tilt": 0.6,
}
PARAM_DOC = (
    "rp: planet radius; ring_inner/ring_outer: radii in units of rp; "
    "ring_opacity: 0-1 optical depth of the ring; ring_tilt: cosine of the "
    "ring inclination (1 = face-on circle, 0.2 = nearly edge-on)"
)


def extent(params: Mapping[str, float]) -> float:
    """Rings dominate the occulting area when they are opaque enough."""
    rp = max(param(params, "rp", 0.1), 0.0)
    outer = param(params, "ring_outer", 2.3)
    opacity = param(params, "ring_opacity", 0.5)
    if opacity <= 0.0:
        return rp
    return max(rp, outer * rp)


def alpha(dx: np.ndarray, dy: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Opaque sphere plus a tilted annulus of partial opacity."""
    rp = max(param(params, "rp", 0.1), 0.0)
    core = disk(dx, dy, rp)
    opacity = float(np.clip(param(params, "ring_opacity", 0.5), 0.0, 1.0))
    if opacity <= 0.0 or rp <= 0.0:
        return core
    tilt = float(np.clip(param(params, "ring_tilt", 0.6), 0.05, 1.0))
    inner = param(params, "ring_inner", 1.4) * rp
    outer = param(params, "ring_outer", 2.3) * rp
    rr = np.hypot(dx, dy / tilt)
    ring = ((rr >= inner) & (rr <= outer)).astype(float) * opacity
    return np.maximum(core, ring)
