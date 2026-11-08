"""Pixel-grid raycasting of occulters across a limb-darkened stellar disc.

The star is rendered once on a ``grid x grid`` pixel grid centred at (0, 0)
with unit radius. Each pixel stores the mean surface brightness over its area
(sub-sampled), so the sum over all pixels is the disc-integrated flux. For a
given planet position the occulter's opacity field ``alpha(dx, dy)`` is
sub-sampled inside the pixel block that the occulter can touch, and the flux is

    flux = 1 - sum(stellar_brightness * alpha) / sum(stellar_brightness)

which is exactly ``1.0`` when nothing overlaps the disc.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from shapehunter.shapes.families import default_params, get_family
from shapehunter.shapes.limb_darkening import check_coefficients

__all__ = [
    "simulate_transit",
    "transit_flux_curve",
    "sky_velocity",
    "transit_duration",
    "a_over_rstar_from_duration",
    "DEFAULT_GRID",
]

DEFAULT_GRID = 401
DEFAULT_SUBPIXEL = 4
_STAR_SUBPIXEL = 3


@lru_cache(maxsize=128)
def _star_pixels(grid: int, u1: float, u2: float) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return ``(pixel_weight, total_flux, pixel_centres)`` for the star.

    ``pixel_weight[i, j]`` is the *area-averaged* surface brightness of pixel
    ``(i, j)``; pixels outside the disc are zero.
    """
    n = int(grid)
    if n < 16:
        raise ValueError("grid must be at least 16 pixels across")
    sub = _STAR_SUBPIXEL
    samples = np.arange(n * sub) + 0.5
    coords = samples / (n * sub) * 2.0 - 1.0
    xs, ys = np.meshgrid(coords, coords, indexing="xy")
    rr = np.hypot(xs, ys)
    mu = np.sqrt(np.clip(1.0 - rr**2, 0.0, None))
    intensity = np.where(rr <= 1.0, 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2, 0.0)
    weight = intensity.reshape(n, sub, n, sub).mean(axis=(1, 3))
    total = float(weight.sum())
    if total <= 0.0:  # pragma: no cover - defensive
        raise ValueError("degenerate limb darkening: stellar flux is zero")
    centres = (np.arange(n) + 0.5) * 2.0 / n - 1.0
    return weight, total, centres


def _merge_params(family: str, params: Mapping[str, float] | None) -> dict[str, float]:
    merged = default_params(family, (params or {}).get("rp", 0.1))
    if params:
        for key, value in params.items():
            merged[str(key)] = float(value)
    return merged


def _blocked_flux(
    weight: np.ndarray,
    centres: np.ndarray,
    x: float,
    y: float,
    radius: float,
    fam: Any,
    params: Mapping[str, float],
    subpixel: int,
) -> float:
    """Stellar flux hidden by the occulter centred at ``(x, y)``."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return 0.0
    if x - radius > 1.0 or x + radius < -1.0 or y - radius > 1.0 or y + radius < -1.0:
        return 0.0
    n = weight.shape[0]
    pixw = 2.0 / n
    i0 = max(int(np.floor((x - radius + 1.0) / pixw)), 0)
    i1 = min(int(np.ceil((x + radius + 1.0) / pixw)), n)
    j0 = max(int(np.floor((y - radius + 1.0) / pixw)), 0)
    j1 = min(int(np.ceil((y + radius + 1.0) / pixw)), n)
    if i1 <= i0 or j1 <= j0:
        return 0.0
    block = weight[i0:i1, j0:j1]
    cx = centres[i0:i1]
    cy = centres[j0:j1]
    offs = (np.arange(subpixel) + 0.5) / subpixel - 0.5
    dx = cx[:, None, None, None] + offs[None, None, None, :] * pixw - x
    dy = cy[None, :, None, None] + offs[None, None, :, None] * pixw - y
    dx, dy = np.broadcast_arrays(dx, dy)
    alpha = np.asarray(fam.alpha(dx, dy, params), dtype=float)
    if alpha.shape != dx.shape:  # pragma: no cover - defensive
        alpha = np.broadcast_to(alpha, dx.shape)
    alpha = np.clip(alpha, 0.0, 1.0)
    coverage = alpha.mean(axis=(2, 3))
    return float(np.sum(block * coverage))


def simulate_transit(
    times: Sequence[float] | np.ndarray,
    x: Sequence[float] | float,
    y: Sequence[float] | float,
    family: str = "sphere",
    params: Mapping[str, float] | None = None,
    u: Sequence[float] = (0.4, 0.25),
    grid: int = DEFAULT_GRID,
    subpixel: int = DEFAULT_SUBPIXEL,
) -> np.ndarray:
    """Render a transit light curve from sky positions.

    Parameters
    ----------
    times:
        Observation times (any units); only used to define the output length.
    x, y:
        Sky-projected occulter position in stellar radii, with the star at
        ``(0, 0)`` and radius ``1``. Scalars broadcast over ``times``.
    family:
        One of the keys of :data:`shapehunter.shapes.families.FAMILIES`.
    params:
        Shape parameters; missing ones fall back to the family defaults.
        ``params['rp']`` is the planet radius in stellar radii.
    u:
        Quadratic limb-darkening coefficients.
    grid:
        Pixel resolution of the stellar disc (``grid x grid``).
    subpixel:
        Sub-samples per pixel and per axis for the occulter.

    Returns
    -------
    numpy.ndarray
        Relative flux, exactly ``1.0`` out of transit.
    """
    t = np.atleast_1d(np.asarray(times, dtype=float))
    xa = np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)), t.shape)
    ya = np.broadcast_to(np.atleast_1d(np.asarray(y, dtype=float)), t.shape)
    n_sub = int(max(1, subpixel))
    fam = get_family(family)
    p = _merge_params(family, params)
    radius = float(fam.extent(p))
    u1, u2 = check_coefficients(u)
    weight, total, centres = _star_pixels(int(grid), u1, u2)
    flux = np.ones_like(t)
    # Fast path: nothing is in transit at all.
    outside = (
        (xa - radius > 1.0) | (xa + radius < -1.0) | (ya - radius > 1.0) | (ya + radius < -1.0)
    )
    for k in range(t.size):
        if outside[k]:
            continue
        blocked = _blocked_flux(weight, centres, xa[k], ya[k], radius, fam, p, n_sub)
        if blocked > 0.0:
            flux[k] = max(0.0, 1.0 - blocked / total)
    return flux


def sky_velocity(period: float, a_over_rstar: float, impact_parameter: float = 0.0) -> float:
    """Sky-projected orbital speed in stellar radii per unit time."""
    b = float(np.clip(impact_parameter, 0.0, 0.999))
    return 2.0 * np.pi * a_over_rstar * np.sqrt(1.0 - b * b) / period


def transit_duration(
    period: float,
    a_over_rstar: float,
    radius: float,
    impact_parameter: float = 0.0,
) -> float:
    """Total (first-to-fourth contact) transit duration.

    Uses the straight-chord approximation, which is accurate to better than a
    percent for ``a/Rstar >~ 4``.
    """
    b = float(np.clip(impact_parameter, 0.0, 0.999))
    chord = np.sqrt(max((1.0 + radius) ** 2 - b * b, 1e-12))
    return 2.0 * chord / sky_velocity(period, a_over_rstar, b)


def a_over_rstar_from_duration(
    period: float,
    duration: float,
    radius: float,
    impact_parameter: float = 0.0,
) -> float:
    """Invert :func:`transit_duration` for the scaled semi-major axis."""
    b = float(np.clip(impact_parameter, 0.0, 0.999))
    chord = np.sqrt(max((1.0 + radius) ** 2 - b * b, 1e-12))
    return float(period * chord / (np.pi * max(duration, 1e-12) * np.sqrt(1.0 - b * b)))


def transit_flux_curve(
    times: Sequence[float] | np.ndarray,
    period: float,
    t0: float,
    a_over_rstar: float,
    family: str = "sphere",
    params: Mapping[str, float] | None = None,
    impact_parameter: float = 0.0,
    u: Sequence[float] = (0.4, 0.25),
    grid: int = DEFAULT_GRID,
    subpixel: int = DEFAULT_SUBPIXEL,
) -> np.ndarray:
    """Raycast a full transit light curve from orbital elements.

    The star is at the origin with radius one, the orbit is circular and the
    sky-projected separation is ``x = v * dt``, ``y = b`` with ``v`` from
    :func:`sky_velocity`.
    """
    t = np.atleast_1d(np.asarray(times, dtype=float))
    b = float(np.clip(impact_parameter, 0.0, 0.999))
    v = sky_velocity(period, a_over_rstar, b)
    dt = np.mod(t - t0 + 0.5 * period, period) - 0.5 * period
    x = v * dt
    y = np.full_like(x, b)
    return simulate_transit(t, x, y, family=family, params=params, u=u, grid=grid, subpixel=subpixel)
