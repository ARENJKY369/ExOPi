"""Analytic transit shapes used for fitting and model comparison."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from shapehunter.data.synthetic import trapezoid_eclipse
from shapehunter.shapes.limb_darkening import integrated_flux
from shapehunter.shapes.raycast import sky_velocity
from shapehunter.utils import wrap_phase

__all__ = [
    "trapezoid_flux",
    "box_flux",
    "circle_overlap_area",
    "uniform_sphere_flux",
    "depth_to_radius",
    "radius_to_depth",
    "projected_separation",
]


def _offset(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Signed time offset from the nearest mid-transit."""
    t = np.asarray(time, dtype=float)
    return wrap_phase((t - t0) / period) * period


def trapezoid_flux(
    time: Sequence[float] | np.ndarray,
    period: float,
    t0: float,
    depth: float,
    duration: float,
    ingress_fraction: float = 0.2,
) -> np.ndarray:
    """Periodic trapezoid transit model."""
    dt = _offset(time, period, t0)
    return 1.0 - trapezoid_eclipse(dt, depth, duration, ingress_fraction)


def box_flux(
    time: Sequence[float] | np.ndarray,
    period: float,
    t0: float,
    depth: float,
    duration: float,
) -> np.ndarray:
    """Periodic box transit model."""
    return trapezoid_flux(time, period, t0, depth, duration, 0.0)


def circle_overlap_area(r: np.ndarray | float, rp: float) -> np.ndarray:
    """Area of the overlap of a unit disc and a disc of radius ``rp``.

    ``r`` is the centre separation. The unit disc is the star, so the result is
    in units where the stellar radius is 1.
    """
    rr = np.asarray(r, dtype=float)
    rp = float(rp)
    if rp <= 0.0:
        return np.zeros_like(rr)
    area = np.zeros_like(rr)
    # Planet fully inside the stellar disc: overlap is the planet area.
    inside = rr <= abs(1.0 - rp)
    area = np.where(inside, np.pi * min(rp, 1.0) ** 2, area)
    # Partial overlap.
    partial = (rr > abs(1.0 - rp)) & (rr < 1.0 + rp)
    if np.any(partial):
        rp_safe = np.clip(rr[partial], abs(1.0 - rp) + 1e-12, 1.0 + rp - 1e-12)
        k0 = np.arccos(np.clip((rp_safe**2 + rp**2 - 1.0) / (2.0 * rp_safe * rp), -1.0, 1.0))
        k1 = np.arccos(np.clip((rp_safe**2 + 1.0 - rp**2) / (2.0 * rp_safe), -1.0, 1.0))
        term = k0 * rp**2 + k1 - 0.5 * np.sqrt(
            np.clip(
                (-rp_safe + rp + 1.0) * (rp_safe + rp - 1.0) * (rp_safe - rp + 1.0) * (rp_safe + rp + 1.0),
                0.0,
                None,
            )
        )
        area[partial] = term
    return np.clip(area, 0.0, np.pi * min(rp, 1.0) ** 2)


def projected_separation(time: np.ndarray, period: float, t0: float, a_over_rstar: float, impact_parameter: float) -> np.ndarray:
    """Sky-projected star-planet separation in stellar radii."""
    b = float(np.clip(impact_parameter, 0.0, 0.999))
    v = sky_velocity(period, a_over_rstar, b)
    dt = _offset(time, period, t0)
    return np.hypot(v * dt, b)


def uniform_sphere_flux(
    time: Sequence[float] | np.ndarray,
    period: float,
    t0: float,
    rp: float,
    a_over_rstar: float,
    impact_parameter: float = 0.0,
    u: Sequence[float] = (0.4, 0.25),
) -> np.ndarray:
    """Uniform-disc (no LD gradient) sphere transit, LD-corrected in depth.

    This is the fast analytic model used to seed fits; the raycaster
    (:func:`shapehunter.shapes.raycast.simulate_transit`) remains the reference
    forward model for non-spherical shapes.
    """
    r = projected_separation(np.asarray(time, dtype=float), period, t0, a_over_rstar, impact_parameter)
    area = circle_overlap_area(r, rp)
    blocked = area / (np.pi * integrated_flux(u))
    return 1.0 - np.clip(blocked, 0.0, 1.0)


def depth_to_radius(depth: float, u: Sequence[float] = (0.4, 0.25)) -> float:
    """Invert ``depth = rp^2 / (1 - u1/3 - u2/6)``."""
    return float(np.sqrt(max(float(depth), 0.0) * integrated_flux(u)))


def radius_to_depth(rp: float, u: Sequence[float] = (0.4, 0.25)) -> float:
    """Forward model for the central transit depth of a sphere."""
    return float(rp) ** 2 / integrated_flux(u)
