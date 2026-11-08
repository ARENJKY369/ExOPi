"""Measure per-transit timing offsets from a fixed trapezoid template."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from shapehunter.core import LightCurve
from shapehunter.data.synthetic import trapezoid_eclipse


def extract_ttv(
    curve: LightCurve,
    period: float,
    t0: float,
    duration: float,
    depth: float,
    ingress_fraction: float = 0.3,
    fit_duration: bool = True,
) -> list[dict[str, float | int]]:
    """Fit each observed event's mid-time relative to a linear ephemeris.

    Returns epochs, measured mid-times, and observed-minus-calculated offsets.
    An event requires at least five points near its predicted time.
    """
    if period <= 0 or not 0 < duration < period / 2 or depth <= 0:
        raise ValueError("invalid transit parameters")
    epochs = np.unique(np.rint((curve.time - t0) / period).astype(int))
    events = []
    for epoch in epochs:
        predicted = t0 + epoch * period
        if predicted - duration < curve.time.min() or predicted + duration > curve.time.max():
            continue
        keep = np.abs(curve.time - predicted) < duration
        if keep.sum() < 5:
            continue
        time, flux, err = curve.time[keep], curve.flux[keep], curve.error()[keep]
        if np.any(~np.isfinite(flux)) or np.any(~np.isfinite(err)) or np.any(err <= 0):
            continue
        if fit_duration:
            def residual(params: np.ndarray) -> np.ndarray:
                model = 1 - trapezoid_eclipse(time - predicted - params[0], depth, params[1], ingress_fraction)
                return (flux - model) / err
            result = least_squares(
                residual,
                [0.0, duration],
                bounds=([-duration / 2, 0.5 * duration,], [duration / 2, 1.5 * duration]),
            )
        else:
            def residual(params: np.ndarray) -> np.ndarray:
                model = 1 - trapezoid_eclipse(time - predicted - params[0], depth, duration, ingress_fraction)
                return (flux - model) / err
            result = least_squares(residual, [0.0], bounds=([-duration / 2], [duration / 2]))
        improvement = np.sum(((flux - 1.0) / err)**2) - np.sum(result.fun**2)
        if result.success and improvement > 9.0:
            offset = float(result.x[0])
            events.append({"epoch": int(epoch), "time": float(predicted + offset), "oc": offset})
    return events
