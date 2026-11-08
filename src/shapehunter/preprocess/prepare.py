"""End-to-end pre-processing of a raw light curve."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.preprocess.detrend import detrend, normalize, sigma_clip
from shapehunter.preprocess.flares import clip_flares
from shapehunter.utils import mad_std

__all__ = ["prepare_lightcurve"]


def prepare_lightcurve(
    lc: LightCurve,
    window_days: float = 2.0,
    clip_sigma: float = 4.0,
    flare_sigma: float = 5.0,
    normalize_flux: bool = True,
) -> LightCurve:
    """Clip flares, remove gross outliers, detrend and normalise.

    Returns a new :class:`~shapehunter.core.LightCurve` whose ``meta`` records
    every step that was applied.
    """
    work = lc.sort()
    work = clip_flares(work, n_sigma=flare_sigma, window_days=window_days)
    keep = sigma_clip(work.time, work.flux, n_sigma=clip_sigma, window_days=0.5)
    work = work.mask(keep)
    if len(work) < 10:
        raise ValueError("light curve has too few points after clipping")
    flux = detrend(work.time, work.flux, window_days=window_days)
    if normalize_flux:
        flux = normalize(flux)
    out = work.with_flux(flux)
    if work.flux_err is not None:
        # Detrending and normalization are multiplicative; propagate their
        # per-cadence scale to uncertainty, treating the fitted trend as fixed.
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = flux / work.flux
        if not np.all(np.isfinite(scale)):
            raise ValueError("cannot propagate uncertainties through zero or nonfinite flux")
        out.flux_err = work.flux_err * np.abs(scale)
    out.meta["preprocess"] = {
        "window_days": float(window_days),
        "clip_sigma": float(clip_sigma),
        "flare_sigma": float(flare_sigma),
        "n_flares_clipped": int(out.meta.get("n_flares_clipped", 0)),
        "n_points": len(out),
        "scatter": float(mad_std(flux)),
    }
    return out
