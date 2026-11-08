"""Family 3/6: a comet-like body with an extended dust tail."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shapehunter.shapes.families.base import disk, param

NAME = "comet_tail"
DEFAULT_PARAMS: dict[str, float] = {
    "tail_length": 8.0,
    "tail_width": 1.2,
    "tail_opacity": 0.45,
    "tail_angle": np.pi,
}
PARAM_DOC = (
    "rp: nucleus radius; tail_length: tail length in units of rp; "
    "tail_width: Gaussian cross-section width in units of rp; "
    "tail_opacity: peak opacity at the nucleus; tail_angle: direction of the "
    "tail in radians (pi points towards -x, i.e. trailing the planet)"
)


def extent(params: Mapping[str, float]) -> float:
    """The tail sets the footprint of this family."""
    rp = max(param(params, "rp", 0.1), 0.0)
    length = param(params, "tail_length", 8.0) * rp
    width = param(params, "tail_width", 1.2) * rp
    if param(params, "tail_opacity", 0.45) <= 0.0:
        return rp
    return max(rp, float(np.hypot(length, 2.0 * width)))


def alpha(dx: np.ndarray, dy: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Opaque nucleus plus an exponentially fading, tapered tail."""
    rp = max(param(params, "rp", 0.1), 0.0)
    core = disk(dx, dy, rp)
    opacity = float(np.clip(param(params, "tail_opacity", 0.45), 0.0, 1.0))
    if opacity <= 0.0 or rp <= 0.0:
        return core
    length = max(param(params, "tail_length", 8.0) * rp, 1e-6)
    width = max(param(params, "tail_width", 1.2) * rp, 1e-6)
    angle = param(params, "tail_angle", float(np.pi))
    ca, sa = np.cos(angle), np.sin(angle)
    along = dx * ca + dy * sa
    cross = -dx * sa + dy * ca
    live = (along > 0.0) & (along < length)
    taper = np.where(live, 1.0 - along / length, 0.0)
    profile = (
        opacity
        * np.exp(-along / (0.35 * length))
        * taper
        * np.exp(-0.5 * (cross / width) ** 2)
    )
    return np.maximum(core, np.clip(profile, 0.0, 1.0))
