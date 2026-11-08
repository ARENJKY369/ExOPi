"""Upward-only outlier (flare) clipping.

Flares are one-sided: they only ever brighten the star. Ordinary sigma
clipping would also remove genuine transits, so this module only rejects
positive excursions.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.utils import mad_std, running_median

__all__ = ["find_flares", "clip_flares"]


def find_flares(
    time: np.ndarray,
    flux: np.ndarray,
    n_sigma: float = 5.0,
    window_days: float = 0.5,
    max_iterations: int = 3,
) -> np.ndarray:
    """Indices of upward outliers relative to the local median."""
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    flags = np.zeros(f.size, dtype=bool)
    if f.size < 8:
        return flags
    for _ in range(int(max(1, max_iterations))):
        trend = running_median(t, np.where(flags, np.nan, f), window_days)
        trend = np.where(np.isfinite(trend), trend, np.nanmedian(f))
        resid = f - trend
        sigma = mad_std(resid[~flags]) if np.any(~flags) else mad_std(resid)
        if sigma <= 0:
            break
        new_flags = flags | (resid > n_sigma * sigma)
        if np.array_equal(new_flags, flags):
            break
        flags = new_flags
    return flags


def clip_flares(
    lc: LightCurve,
    n_sigma: float = 5.0,
    window_days: float = 0.5,
    max_iterations: int = 3,
    replace_with_trend: bool = True,
) -> LightCurve:
    """Return a copy of ``lc`` with upward outliers clipped.

    Clipped samples are replaced by the local trend value (default) or removed
    from the light curve when ``replace_with_trend`` is false. Transits are
    never touched because only positive residuals are flagged.
    """
    flags = find_flares(lc.time, lc.flux, n_sigma, window_days, max_iterations)
    out = lc.copy()
    if not np.any(flags):
        out.meta["n_flares_clipped"] = 0
        return out
    if replace_with_trend:
        trend = running_median(lc.time, np.where(flags, np.nan, lc.flux), window_days)
        fallback = float(np.median(lc.flux))
        trend = np.where(np.isfinite(trend), trend, fallback)
        fixed = out.flux.copy()
        fixed[flags] = trend[flags]
        out.flux = fixed
        out.meta["n_flares_clipped"] = int(np.count_nonzero(flags))
        out.meta["flares_replaced"] = True
        return out
    out.meta["n_flares_clipped"] = int(np.count_nonzero(flags))
    out.meta["flares_replaced"] = False
    return out.mask(~flags)


def flare_report(lc: LightCurve) -> List[Tuple[float, float]]:
    """List of ``(time, excess_flux)`` for detected flares."""
    flags = find_flares(lc.time, lc.flux)
    return [(float(lc.time[i]), float(lc.flux[i] - np.median(lc.flux))) for i in np.flatnonzero(flags)]
