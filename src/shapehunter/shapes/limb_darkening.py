"""Quadratic limb darkening for a pixelised stellar disc.

The stellar surface brightness is described by the standard quadratic law

.. math:: I(\\mu) = 1 - u_1 (1-\\mu) - u_2 (1-\\mu)^2

with ``mu = sqrt(1 - r^2)`` the cosine of the angle between the surface normal
and the line of sight, and ``r`` the projected radius in stellar radii.
``I(0) = 1`` at disc centre; the intensity is *not* normalised by the
disc-integrated flux, so the total flux of the star is
``pi * (1 - u1/3 - u2/6)``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "check_coefficients",
    "intensity_at_radius",
    "intensity_at_mu",
    "integrated_flux",
    "analytic_center_depth",
    "analytic_depth_curve",
]


def check_coefficients(u: Sequence[float] | None) -> Tuple[float, float]:
    """Validate and unpack a quadratic limb-darkening pair."""
    if u is None:
        return 0.0, 0.0
    vals = tuple(float(v) for v in u)
    if len(vals) != 2:
        raise ValueError("quadratic limb darkening needs exactly two coefficients")
    return vals[0], vals[1]


def intensity_at_mu(mu: np.ndarray | float, u: Sequence[float] = (0.4, 0.25)) -> np.ndarray:
    """Surface brightness ``I(mu)`` normalised so that ``I(1) = 1``."""
    u1, u2 = check_coefficients(u)
    mu_arr = np.clip(np.asarray(mu, dtype=float), 0.0, 1.0)
    one_minus = 1.0 - mu_arr
    return 1.0 - u1 * one_minus - u2 * one_minus**2


def intensity_at_radius(r: np.ndarray | float, u: Sequence[float] = (0.4, 0.25)) -> np.ndarray:
    """Surface brightness at projected radius ``r`` (stellar radii)."""
    r_arr = np.asarray(r, dtype=float)
    inside = r_arr <= 1.0
    mu = np.sqrt(np.clip(1.0 - r_arr**2, 0.0, None))
    flux = intensity_at_mu(mu, u)
    return np.where(inside, flux, 0.0)


def integrated_flux(u: Sequence[float] = (0.4, 0.25)) -> float:
    """Analytic disc-integrated flux divided by pi (i.e. mean surface brightness)."""
    u1, u2 = check_coefficients(u)
    return float(1.0 - u1 / 3.0 - u2 / 6.0)


def analytic_center_depth(rp: float, u: Sequence[float] = (0.4, 0.25)) -> float:
    """Transit depth of an opaque sphere crossing the disc centre.

    For a small planet the blocked flux is ``pi * rp^2 * I(0)`` while the
    stellar flux is ``pi * (1 - u1/3 - u2/6)``, hence
    ``depth = rp^2 / (1 - u1/3 - u2/6)``.
    """
    rp = float(rp)
    if rp < 0:
        raise ValueError("planet radius must be non-negative")
    return rp**2 / integrated_flux(u)


def analytic_depth_curve(r: np.ndarray | float, rp: float, u: Sequence[float] = (0.4, 0.25)) -> np.ndarray:
    """Opaque-sphere transit depth ignoring limb darkening gradients.

    ``r`` is the sky-projected centre separation in stellar radii. The curve is
    an idealised flat-bottomed approximation used only for sanity checks and
    for seeding least-squares fits; the raycaster is the ground truth.
    """
    r_arr = np.asarray(r, dtype=float)
    depth = analytic_center_depth(rp, u)
    return np.where(r_arr <= 1.0 - rp, depth, np.where(r_arr >= 1.0 + rp, 0.0, depth / 2.0))
