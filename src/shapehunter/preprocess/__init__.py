"""Pre-processing: flare clipping, detrending and normalisation."""

from __future__ import annotations

from shapehunter.preprocess.detrend import (
    detrend,
    normalize,
    robust_sigma,
    running_median,
    sigma_clip,
    whiten,
)
from shapehunter.preprocess.flares import clip_flares, find_flares, flare_report
from shapehunter.preprocess.prepare import prepare_lightcurve

__all__ = [
    "clip_flares",
    "find_flares",
    "flare_report",
    "detrend",
    "normalize",
    "sigma_clip",
    "robust_sigma",
    "running_median",
    "whiten",
    "prepare_lightcurve",
]
