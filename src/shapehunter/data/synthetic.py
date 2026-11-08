"""Deterministic synthetic light curves: noise, stellar variability, flares,
planet injection and eclipsing binaries.

Everything in this module is seeded; the same seed always yields the same
arrays, which keeps the whole test suite reproducible and offline.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from shapehunter.shapes.raycast import transit_flux_curve
from shapehunter.core import LightCurve
from shapehunter.utils import wrap_phase

__all__ = [
    "LightCurve",
    "DEFAULT_CADENCE",
    "generate_lightcurve",
    "inject_transit",
    "simulate_planet_transit",
    "simulate_eclipsing_binary",
    "trapezoid_eclipse",
]

#: 30 minute cadence in days (Kepler long cadence).
DEFAULT_CADENCE = 30.0 * 60.0 / 86400.0


def trapezoid_eclipse(phase: np.ndarray, depth: float, duration: float, ingress: float) -> np.ndarray:
    """Symmetric trapezoid eclipse profile as a function of phase offset.

    ``duration`` is the total first-to-last contact width and ``ingress`` the
    fraction of it spent in ingress/egress (0 = box, 1 = pure triangle).
    """
    ph = np.abs(np.asarray(phase, dtype=float))
    half = 0.5 * float(duration)
    ingress = float(np.clip(ingress, 0.0, 1.0))
    flat = half * (1.0 - ingress)
    ramp = max(half - flat, 1e-12)
    inside = ph <= half
    frac = np.where(ph <= flat, 1.0, np.clip((half - ph) / ramp, 0.0, 1.0))
    return np.where(inside, float(depth) * frac, 0.0)


def generate_lightcurve(
    n_points: int = 2400,
    cadence: float = DEFAULT_CADENCE,
    noise: float = 3e-4,
    seed: int = 0,
    trend_amplitude: float = 2e-3,
    flare_rate: float = 0.08,
    gap_fraction: float = 0.03,
    centroid_scatter: float = 2e-3,
    include_centroids: bool = True,
) -> LightCurve:
    """Build a quiet-star light curve with gaps, trends and a few flares.

    Parameters
    ----------
    n_points:
        Number of accepted (post-gap) cadences.
    noise:
        Per-point Gaussian white noise (relative flux).
    trend_amplitude:
        Amplitude of the slow (stellar rotation-like) variability.
    flare_rate:
        Expected number of upward flares per day.
    gap_fraction:
        Fraction of the baseline removed by random gaps.
    """
    if n_points < 10:
        raise ValueError("n_points must be at least 10")
    rng = np.random.default_rng(int(seed))
    target = int(n_points / (1.0 - float(gap_fraction)))
    time = np.sort(rng.uniform(0.0, target * cadence, target))
    time = time + rng.normal(0.0, 0.02 * cadence, target)
    time = np.sort(time)
    # Keep roughly `n_points` samples.
    keep = np.zeros(time.size, dtype=bool)
    idx = rng.choice(time.size, size=min(n_points, time.size), replace=False)
    keep[idx] = True
    time = time[keep]
    baseline = float(time.max() - time.min())

    trend = 1.0 + trend_amplitude * (
        np.sin(2.0 * np.pi * time / 27.0) + 0.5 * np.cos(2.0 * np.pi * time / 11.3)
    )
    flux = trend * (1.0 + rng.normal(0.0, noise, time.size))
    flux_err = np.full(time.size, float(noise))

    n_flares = int(rng.poisson(max(flare_rate, 0.0) * baseline))
    flare_meta: list[dict[str, float]] = []
    for _ in range(n_flares):
        centre = rng.uniform(time.min(), time.max())
        amplitude = float(rng.uniform(8.0, 25.0) * noise)
        width = float(rng.uniform(0.01, 0.05))
        bump = amplitude * np.exp(-0.5 * ((time - centre) / width) ** 2)
        flux = flux + bump
        flare_meta.append({"time": float(centre), "amplitude": amplitude, "width": width})

    cx = cy = None
    if include_centroids:
        cx = 0.5 + rng.normal(0.0, centroid_scatter, time.size)
        cy = 0.5 + rng.normal(0.0, centroid_scatter, time.size)

    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        centroid_x=cx,
        centroid_y=cy,
        meta={
            "seed": int(seed),
            "cadence": float(cadence),
            "noise": float(noise),
            "baseline": baseline,
            "trend_amplitude": float(trend_amplitude),
            "flares": flare_meta,
            "is_eclipsing_binary": False,
            "target": f"synthetic-{int(seed)}",
        },
    )


def inject_transit(
    time: Sequence[float] | np.ndarray,
    flux: Sequence[float] | np.ndarray,
    period: float = 3.5,
    t0: float = 2.0,
    rp: float = 0.1,
    family: str = "sphere",
    params: Optional[Mapping[str, float]] = None,
    u: Sequence[float] = (0.4, 0.25),
    grid: int = 201,
    a_over_rstar: Optional[float] = None,
    impact_parameter: float = 0.0,
) -> np.ndarray:
    """Multiply a raycast transit signal into ``flux``.

    When ``a_over_rstar`` is ``None`` it is derived from a nominal 0.15 d
    duration so that injected planets have a realistic transit width.
    """
    t = np.asarray(time, dtype=float)
    shape_params: Dict[str, float] = dict(params or {})
    shape_params["rp"] = float(rp)
    if a_over_rstar is None:
        # duration ~ 0.15 d for a 3.5 d orbit; scale with sqrt(period)
        a_over_rstar = max(4.0, 9.0 * (float(period) / 3.5) ** (2.0 / 3.0))
    model = transit_flux_curve(
        t,
        period=float(period),
        t0=float(t0),
        a_over_rstar=float(a_over_rstar),
        family=family,
        params=shape_params,
        impact_parameter=float(impact_parameter),
        u=tuple(u),
        grid=int(grid),
    )
    return np.asarray(flux, dtype=float) * model


def simulate_planet_transit(
    lc: LightCurve,
    period: float = 3.5,
    t0: float = 2.0,
    rp: float = 0.1,
    family: str = "sphere",
    params: Optional[Mapping[str, float]] = None,
    u: Sequence[float] = (0.4, 0.25),
    grid: int = 201,
    a_over_rstar: Optional[float] = None,
    impact_parameter: float = 0.0,
) -> LightCurve:
    """Return a copy of ``lc`` with a transit signal injected."""
    flux = inject_transit(
        lc.time,
        lc.flux,
        period=period,
        t0=t0,
        rp=rp,
        family=family,
        params=params,
        u=u,
        grid=grid,
        a_over_rstar=a_over_rstar,
        impact_parameter=impact_parameter,
    )
    truth = {
        "period": float(period),
        "t0": float(t0),
        "rp": float(rp),
        "family": family,
        "params": dict(params or {}),
        "impact_parameter": float(impact_parameter),
    }
    out = lc.with_flux(flux)
    out.meta["injected"] = truth
    return out


def simulate_eclipsing_binary(
    n_points: int = 2400,
    cadence: float = DEFAULT_CADENCE,
    noise: float = 3e-4,
    seed: int = 11,
    period: float = 2.4,
    t0: float = 0.7,
    depth_primary: float = 0.06,
    depth_secondary: float = 0.025,
    duration: float = 0.18,
    ingress: float = 0.45,
    ellipsoidal: float = 4e-3,
    include_centroids: bool = True,
) -> LightCurve:
    """Synthetic detached eclipsing binary (a classic false positive).

    The light curve shows a deep V-shaped primary eclipse, a shallower
    secondary eclipse at phase 0.5 and out-of-eclipse ellipsoidal modulation.
    """
    base = generate_lightcurve(
        n_points=n_points,
        cadence=cadence,
        noise=noise,
        seed=seed,
        trend_amplitude=5e-4,
        flare_rate=0.0,
        include_centroids=include_centroids,
    )
    rng = np.random.default_rng(int(seed) + 991)
    time = base.time
    dt = np.mod(time - t0 + 0.5 * period, period) - 0.5 * period
    flux = np.ones_like(time)
    flux -= trapezoid_eclipse(dt, depth_primary, duration, ingress)
    phase = wrap_phase((time - t0) / period - 0.5) * period
    flux -= trapezoid_eclipse(phase, depth_secondary, duration * 0.9, ingress)
    flux += ellipsoidal * np.cos(4.0 * np.pi * (time - t0) / period)
    out = LightCurve(
        time=time,
        flux=flux + rng.normal(0.0, noise * 0.5, time.size),
        flux_err=np.full(time.size, float(noise)),
        centroid_x=base.centroid_x,
        centroid_y=base.centroid_y,
        meta=dict(base.meta),
    )
    out.meta.update(
        {
            "is_eclipsing_binary": True,
            "truth": {
                "period": float(period),
                "t0": float(t0),
                "depth_primary": float(depth_primary),
                "depth_secondary": float(depth_secondary),
                "duration": float(duration),
            },
            "target": f"synthetic-eb-{int(seed)}",
        }
    )
    return out
