"""Detrending, normalisation and clipping utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.utils import mad_std, running_median

__all__ = ["detrend", "normalize", "sigma_clip", "whiten", "running_median", "robust_sigma"]


def robust_sigma(values: np.ndarray) -> float:
    """MAD-based scatter estimate."""
    return mad_std(values)


def sigma_clip(
    time: np.ndarray,
    flux: np.ndarray,
    n_sigma: float = 4.0,
    max_iterations: int = 5,
    flux_err: Optional[np.ndarray] = None,
    window_days: float = 0.5,
    protect_transits: bool = True,
    min_run_length: int = 2,
) -> np.ndarray:
    """Boolean mask of inliers, robust against genuine transits.

    Points are compared with a local running median rather than the global
    median, so slow trends cannot masquerade as outliers. When
    ``protect_transits`` is set, only *isolated* outliers (cosmic rays, hot
    pixels) are rejected: contiguous runs of ``min_run_length`` or more
    consecutive clipped cadences are kept, because a transit looks exactly
    like a multi-cadence downward excursion.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    keep = np.ones(f.size, dtype=bool)
    if f.size < 5:
        return keep
    for _ in range(int(max(1, max_iterations))):
        trend = running_median(t, np.where(keep, f, np.nan), window_days)
        fallback = float(np.nanmedian(f))
        trend = np.where(np.isfinite(trend), trend, fallback)
        resid = f - trend
        sigma = robust_sigma(resid[keep]) if np.any(keep) else robust_sigma(resid)
        if sigma <= 0:
            break
        flagged = np.abs(resid) > n_sigma * sigma
        if protect_transits and min_run_length > 1:
            flagged = _isolated_only(flagged, min_run_length)
        new_keep = keep & ~flagged
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if flux_err is not None:
        err = np.asarray(flux_err, dtype=float)
        keep &= np.isfinite(err) & (err > 0)
    return keep & np.isfinite(t) & np.isfinite(f)


def _isolated_only(flags: np.ndarray, min_run_length: int) -> np.ndarray:
    """Keep flags that belong to short (isolated) runs only."""
    out = np.zeros_like(flags, dtype=bool)
    n = flags.size
    if n == 0:
        return out
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return out
    # Split flagged indices into runs of consecutive samples.
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    for run in np.split(idx, breaks):
        if run.size and run.size < int(min_run_length):
            out[run] = True
    return out


def detrend(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = 2.0,
    n_sigma: float = 3.0,
    iterations: int = 2,
    protect_transits: bool = True,
    n_bins_per_window: int = 6,
) -> np.ndarray:
    """Remove slow stellar/instrumental trends by dividing by a binned median.

    The trend is estimated on time bins roughly ``window_days /
    n_bins_per_window`` wide, interpolated to the observing times and divided
    out. Transit-like negative excursions are masked while the trend is
    re-estimated, so genuine transits are not absorbed into the trend.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    if t.size < 8:
        med = float(np.median(f)) if f.size else 1.0
        return f / med if med else f.copy()
    baseline = float(np.nanmax(t) - np.nanmin(t))
    width = max(float(window_days) / float(n_bins_per_window), 4.0 * float(np.median(np.diff(np.sort(t)))))
    n_bins = int(np.clip(np.ceil(baseline / width), 4, 200))
    edges = np.linspace(float(np.nanmin(t)), float(np.nanmax(t)) + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)

    keep = np.ones(t.size, dtype=bool)
    trend = np.ones(t.size, dtype=float)
    for _ in range(int(max(1, iterations))):
        centres = np.empty(n_bins)
        values = np.full(n_bins, np.nan)
        for k in range(n_bins):
            sel = idx == k
            if not np.any(sel):
                continue
            good = sel & keep
            use = good if np.count_nonzero(good) >= 3 else sel
            centres[k] = float(np.mean(t[use]))
            values[k] = float(np.median(f[use]))
        valid = np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            break
        trend = np.interp(t, centres[valid], values[valid])
        trend = np.where(np.isfinite(trend) & (trend != 0.0), trend, np.median(f))
        if not protect_transits:
            break
        resid = f / trend - 1.0
        sigma = robust_sigma(resid)
        if sigma > 0:
            depth_cut = 3.0 * sigma
            keep = resid > -max(depth_cut, float(n_sigma) * sigma)
    out = f / trend
    scale = float(np.median(out))
    return out / scale if scale else out


def normalize(flux: np.ndarray, method: str = "median") -> np.ndarray:
    """Normalise a light curve to out-of-transit flux 1.

    ``method`` is ``"median"`` (default) or ``"mean"``; the upper half of the
    distribution is used so transits do not bias the zero point.
    """
    f = np.asarray(flux, dtype=float)
    if f.size == 0:
        return f.copy()
    if method == "mean":
        ref = float(np.mean(f))
    elif method == "median":
        ref = float(np.median(f))
    elif method == "out_of_transit":
        ref = float(np.median(f[f >= np.percentile(f, 60.0)]))
    else:
        raise ValueError(f"unknown normalisation method {method!r}")
    return f / ref if ref else f.copy()


def whiten(time: np.ndarray, flux: np.ndarray, window_days: float = 0.5) -> np.ndarray:
    """High-pass filter: divide by a running median of the given window."""
    trend = running_median(np.asarray(time, dtype=float), np.asarray(flux, dtype=float), window_days)
    trend = np.where(trend != 0.0, trend, 1.0)
    out = np.asarray(flux, dtype=float) / trend
    med = float(np.median(out))
    return out / med if med else out
