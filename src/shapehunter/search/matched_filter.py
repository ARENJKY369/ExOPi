"""Matched filtering of a light curve against the shape template bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.search.template_bank import Template, TemplateBank
from shapehunter.utils import phase_fold

__all__ = ["MatchResult", "matched_filter", "match_bank", "best_match"]


@dataclass
class MatchResult:
    """Outcome of correlating one template with one transit signal."""

    family: str
    label: str
    amplitude: float
    amplitude_err: float
    snr: float
    chi2: float
    n_points: int
    n_in_transit: int
    shape: np.ndarray
    phi: np.ndarray

    @property
    def depth(self) -> float:
        """Fitted transit depth (template amplitude)."""
        return float(self.amplitude)

    def to_dict(self) -> Dict[str, float | str]:
        """Serialisable summary."""
        return {
            "family": self.family,
            "label": self.label,
            "amplitude": float(self.amplitude),
            "amplitude_err": float(self.amplitude_err),
            "snr": float(self.snr),
            "chi2": float(self.chi2),
            "n_points": int(self.n_points),
            "n_in_transit": int(self.n_in_transit),
        }


def matched_filter(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    template: Template,
    min_phase_points: int = 1,
) -> MatchResult:
    """Fit a template amplitude by inverse-variance weighted correlation.

    The model is ``flux = 1 - A * template(phi)``; ``A`` is solved in closed
    form, so the template only needs to be evaluated once.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    err = np.clip(np.asarray(flux_err, dtype=float), 1e-12, None)
    dt = phase_fold(t, period, t0) * period
    phi = dt / max(float(duration), 1e-12)
    shape = template.evaluate(phi)
    active = shape > 0.0
    n_in = int(np.count_nonzero(active))
    if n_in < int(min_phase_points):
        return MatchResult(
            family=template.family,
            label=template.label,
            amplitude=0.0,
            amplitude_err=float("nan"),
            snr=0.0,
            chi2=float("nan"),
            n_points=int(t.size),
            n_in_transit=n_in,
            shape=shape,
            phi=phi,
        )
    w = 1.0 / err**2
    s = shape
    denom = float(np.sum(w * s * s))
    if denom <= 0.0:  # pragma: no cover - defensive
        return MatchResult(template.family, template.label, 0.0, float("nan"), 0.0, float("nan"), int(t.size), n_in, shape, phi)
    depth_data = 1.0 - f
    amplitude = float(np.sum(w * s * depth_data) / denom)
    amplitude_err = float(1.0 / np.sqrt(denom))
    model = 1.0 - amplitude * s
    chi2 = float(np.sum(((f - model) / err) ** 2))
    return MatchResult(
        family=template.family,
        label=template.label,
        amplitude=amplitude,
        amplitude_err=amplitude_err,
        snr=float(amplitude / amplitude_err) if amplitude_err > 0 else 0.0,
        chi2=chi2,
        n_points=int(t.size),
        n_in_transit=n_in,
        shape=shape,
        phi=phi,
    )


def match_bank(
    bank: TemplateBank,
    lightcurve: LightCurve | tuple,
    period: float,
    t0: float,
    duration: float,
    flux_err: Optional[np.ndarray] = None,
) -> List[MatchResult]:
    """Run :func:`matched_filter` over every template in the bank."""
    if isinstance(lightcurve, LightCurve):
        time, flux = lightcurve.time, lightcurve.flux
        err = lightcurve.error() if flux_err is None else np.asarray(flux_err, dtype=float)
    else:
        time, flux = lightcurve
        err = np.full_like(np.asarray(flux, dtype=float), 1e-4) if flux_err is None else np.asarray(flux_err, dtype=float)
    return [
        matched_filter(time, flux, err, period, t0, duration, template)
        for template in bank
    ]


def best_match(results: Sequence[MatchResult]) -> MatchResult:
    """Return the result with the smallest chi-square."""
    if not results:
        raise ValueError("no match results")
    return min(results, key=lambda r: (np.inf if not np.isfinite(r.chi2) else r.chi2))
