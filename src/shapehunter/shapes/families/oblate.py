"""Family 5/6: a rotationally flattened (oblate) body."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shapehunter.shapes.families.base import ellipse, param

NAME = "oblate"
DEFAULT_PARAMS: dict[str, float] = {
    "flattening": 0.25,
    "obliquity": 0.0,
}
PARAM_DOC = (
    "rp: equatorial radius; flattening: (a-b)/a in [0, 0.9]; "
    "obliquity: rotation angle of the projected ellipse in radians"
)


def extent(params: Mapping[str, float]) -> float:
    """The equatorial radius bounds the projected body."""
    return max(param(params, "rp", 0.1), 0.0)


def alpha(dx: np.ndarray, dy: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Opaque ellipse with semi-minor axis ``rp*(1-flattening)``."""
    rp = max(param(params, "rp", 0.1), 0.0)
    flattening = float(np.clip(param(params, "flattening", 0.25), 0.0, 0.9))
    angle = param(params, "obliquity", 0.0)
    return ellipse(dx, dy, rp, rp * (1.0 - flattening), angle)
