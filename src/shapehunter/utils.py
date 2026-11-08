"""Small numeric helpers used across shapehunter."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

__all__ = [
    "mad_std",
    "weighted_mean",
    "weighted_rms",
    "phase_fold",
    "wrap_phase",
    "bin_phase",
    "running_median",
    "safe_float",
    "as_float_array",
]


def mad_std(values: Iterable[float]) -> float:
    """Median absolute deviation converted to a Gaussian sigma."""
    arr = as_float_array(values)
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


def weighted_mean(values: np.ndarray, errors: np.ndarray | None = None) -> float:
    """Inverse-variance weighted mean (falls back to the plain mean)."""
    arr = as_float_array(values)
    if arr.size == 0:
        return 0.0
    if errors is None:
        return float(np.mean(arr))
    err = as_float_array(errors)
    var = np.square(err)
    good = np.isfinite(arr) & np.isfinite(var) & (var > 0)
    if not np.any(good):
        return float(np.mean(arr))
    w = 1.0 / var[good]
    return float(np.sum(arr[good] * w) / np.sum(w))


def weighted_rms(values: np.ndarray, errors: np.ndarray | None = None) -> float:
    """Root mean square scatter of ``values`` around zero (or weighted)."""
    arr = as_float_array(values)
    if arr.size == 0:
        return 0.0
    if errors is None:
        return float(np.sqrt(np.mean(arr**2)))
    err = as_float_array(errors)
    good = np.isfinite(arr) & np.isfinite(err) & (err > 0)
    if not np.any(good):
        return float(np.sqrt(np.mean(arr**2)))
    return float(np.sqrt(np.mean((arr[good] / err[good]) ** 2)))


def phase_fold(time: np.ndarray, period: float, t0: float = 0.0) -> np.ndarray:
    """Fold ``time`` on ``period``; mid-transit is at phase 0."""
    t = as_float_array(time)
    if period <= 0:
        raise ValueError("period must be positive")
    return wrap_phase((t - t0) / period)


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap phases into ``[-0.5, 0.5)``."""
    ph = as_float_array(phase)
    return np.mod(ph + 0.5, 1.0) - 0.5


def bin_phase(
    phase: np.ndarray,
    flux: np.ndarray,
    n_bins: int = 40,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a folded light curve between phase -0.5 and 0.5.

    Returns ``(centres, binned_flux, binned_error)``; empty bins are NaN.
    """
    ph = as_float_array(phase)
    fl = as_float_array(flux)
    edges = np.linspace(-0.5, 0.5, int(n_bins) + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, int(n_bins) - 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    out = np.full(int(n_bins), np.nan)
    err = np.full(int(n_bins), np.nan)
    w = None if weights is None else as_float_array(weights)
    for k in range(int(n_bins)):
        sel = idx == k
        if not np.any(sel):
            continue
        vals = fl[sel]
        out[k] = np.median(vals) if w is None else weighted_mean(vals, w[sel])
        err[k] = mad_std(vals) / np.sqrt(max(sel.sum(), 1))
    return centres, out, err


def running_median(time: np.ndarray, values: np.ndarray, window: float) -> np.ndarray:
    """Running median in a window of width ``window`` (units of ``time``).

    NaN samples are ignored; windows that contain only NaNs yield NaN.
    """
    t = as_float_array(time)
    v = as_float_array(values)
    if t.size == 0:
        return v.copy()
    order = np.argsort(t)
    ts, vs = t[order], v[order]
    out = np.empty_like(vs)
    half = 0.5 * float(window)
    lo = np.searchsorted(ts, ts - half, side="left")
    hi = np.searchsorted(ts, ts + half, side="right")
    for k in range(ts.size):
        chunk = vs[lo[k]:hi[k]]
        finite = chunk[np.isfinite(chunk)]
        out[k] = np.median(finite) if finite.size else np.nan
    result = np.empty_like(out)
    result[order] = out
    return result


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert to float, returning ``default`` for non-finite input."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def as_float_array(values: Iterable[float]) -> np.ndarray:
    """Convert to a 1-D float array."""
    return np.asarray(values, dtype=float).ravel()
