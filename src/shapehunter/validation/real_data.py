"""Explicit validation against locally supplied, independently labelled TESS data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from shapehunter.core import LightCurve
from shapehunter.pipeline.run_star import search_star


def _load_pdcsap_fits(path: str | Path, quality_column: str, mission: str) -> LightCurve:
    """Load finite quality-zero mission PDCSAP flux and errors from a FITS file.

    Time is returned in BJD when the FITS header provides BJDREFI/BJDREFF.
    The flux and errors are normalized by the median positive flux. This loader
    does not download files or infer scientific labels.
    """
    with fits.open(path, memmap=False) as hdus:
        data = hdus[1].data
        names = set(data.names)
        required = {"TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", quality_column}
        if not required <= names:
            raise ValueError("FITS table is missing a required time, flux, error, or quality column")
        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=float)
        error = np.asarray(data["PDCSAP_FLUX_ERR"], dtype=float)
        quality = np.asarray(data[quality_column])
        keep = np.isfinite(time) & np.isfinite(flux) & np.isfinite(error) & (flux > 0) & (error > 0) & (quality == 0)
        if keep.sum() < 10:
            raise ValueError("fewer than 10 usable FITS samples")
        reference = float(hdus[1].header.get("BJDREFI", 0)) + float(hdus[1].header.get("BJDREFF", 0))
        median = float(np.median(flux[keep]))
        metadata = {"target": str(hdus[0].header.get("OBJECT", "unknown")),
                    "source": f"{mission} FITS", "time_reference": reference,
                    "input": str(path), "discarded_samples": int((~keep).sum())}
        return LightCurve(time=time[keep] + reference, flux=flux[keep] / median,
                          flux_err=error[keep] / median, meta=metadata).sort()


def load_tess_fits(path: str | Path) -> LightCurve:
    """Read normalized TESS PDCSAP data with QUALITY equal to zero."""
    return _load_pdcsap_fits(path, "QUALITY", "TESS")


def load_kepler_fits(path: str | Path) -> LightCurve:
    """Read normalized Kepler PDCSAP data with SAP_QUALITY equal to zero."""
    return _load_pdcsap_fits(path, "SAP_QUALITY", "Kepler")


def load_flux_csv(path: str | Path, target: str) -> LightCurve:
    """Load time/flux CSV, estimating scatter later because errors are absent.

    Target, quality flags and absolute time reference cannot be verified from
    these two columns; the caller supplies the label and records provenance.
    """
    if not target.strip():
        raise ValueError("target label is required")
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if rows.dtype.names is None or not {"time", "flux"} <= set(rows.dtype.names):
        raise ValueError("CSV must contain time and flux columns")
    rows = np.atleast_1d(rows)
    time, flux = rows["time"], rows["flux"]
    keep = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    if keep.sum() < 10:
        raise ValueError("fewer than 10 usable CSV samples")
    median = float(np.median(flux[keep]))
    return LightCurve(time=time[keep], flux=flux[keep] / median, meta={
        "target": target, "source": "time/flux CSV", "input": str(path),
        "quality_flags": "unavailable", "uncertainties": "estimated",
        "discarded_samples": int((~keep).sum()),
    }).sort()


def validate_reference(curve: LightCurve, expected_period: float, reference_source: str,
                       min_power: float = 8.0, period_tolerance: float = 0.01) -> dict:
    """Compare detections with an independently supplied reference period.

    The reference source must identify the provenance of the label. At least
    two full reference periods of baseline are required. Harmonics are listed
    but are not counted as exact recovery.
    """
    if not reference_source.strip() or not np.isfinite(expected_period) or expected_period <= 0 or not 0 < period_tolerance < 1:
        raise ValueError("a reference source and positive period/tolerance are required")
    if curve.baseline < 2 * expected_period:
        raise ValueError("light-curve baseline must span at least two reference periods")
    candidates = search_star(curve, min_power=min_power)
    periods = [float(c.period) for c in candidates]
    recovered = any(abs(p - expected_period) <= expected_period * period_tolerance for p in periods)
    harmonics = [p for p in periods if any(abs(p - expected_period * factor) <= expected_period * factor * period_tolerance for factor in (0.5, 2.0))]
    return {"target": curve.meta.get("target", "unknown"), "reference_source": reference_source,
            "expected_period": expected_period, "period_tolerance": period_tolerance,
            "baseline": curve.baseline, "periods": periods, "recovered": recovered,
            "harmonic_periods": harmonics, "candidate_statuses": [c.status for c in candidates],
            "note": "A single reference target does not establish survey-wide reliability."}
