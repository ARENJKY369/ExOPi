"""Family 1/6: a perfect sphere (the null hypothesis of every BLS/TLS search)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shapehunter.shapes.families.base import disk, param

NAME = "sphere"
DEFAULT_PARAMS: dict[str, float] = {}
PARAM_DOC = "rp: planet radius in stellar radii"


def extent(params: Mapping[str, float]) -> float:
    """Maximum occulter extent in stellar radii."""
    return max(param(params, "rp", 0.1), 0.0)


def alpha(dx: np.ndarray, dy: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Opacity of an opaque disc centred on the planet position."""
    return disk(dx, dy, extent(params))
