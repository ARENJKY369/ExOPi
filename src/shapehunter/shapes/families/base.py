"""Shared helpers for the shape families."""

from __future__ import annotations

from typing import Mapping

import numpy as np

__all__ = ["param", "disk", "ellipse", "rotate", "smoothstep"]


def param(params: Mapping[str, float], key: str, default: float) -> float:
    """Read a float parameter, falling back to ``default``."""
    value = params.get(key, default)
    return float(default if value is None else value)


def rotate(dx: np.ndarray, dy: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate offsets by ``angle`` radians (counter-clockwise)."""
    if angle == 0.0:
        return dx, dy
    ca, sa = np.cos(angle), np.sin(angle)
    return dx * ca + dy * sa, -dx * sa + dy * ca


def disk(dx: np.ndarray, dy: np.ndarray, radius: float) -> np.ndarray:
    """Hard-edged opaque disc of radius ``radius``."""
    if radius <= 0.0:
        return np.zeros_like(dx, dtype=float)
    return (dx * dx + dy * dy <= radius * radius).astype(float)


def ellipse(
    dx: np.ndarray,
    dy: np.ndarray,
    semi_major: float,
    semi_minor: float,
    angle: float = 0.0,
) -> np.ndarray:
    """Hard-edged opaque ellipse."""
    if semi_major <= 0.0 or semi_minor <= 0.0:
        return np.zeros_like(dx, dtype=float)
    xr, yr = rotate(dx, dy, angle)
    inside = (xr / semi_major) ** 2 + (yr / semi_minor) ** 2 <= 1.0
    return inside.astype(float)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Classic smoothstep ramp from 0 at ``edge0`` to 1 at ``edge1``."""
    if edge1 == edge0:
        return (x >= edge1).astype(float)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
