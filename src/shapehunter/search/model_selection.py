"""BIC model selection: does a non-spherical shape beat a sphere?"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.search.matched_filter import MatchResult, match_bank, matched_filter
from shapehunter.search.template_bank import TemplateBank

__all__ = ["ModelComparison", "bic", "compare_families"]


def bic(chi2: float, n_params: int, n_data: int) -> float:
    """Gaussian BIC with known variances: ``chi2 + k * ln(n)``."""
    if n_data <= 0:
        return float("inf")
    return float(chi2 + n_params * np.log(n_data))


@dataclass
class ModelComparison:
    """Result of comparing all families against the spherical null model."""

    family: str = "sphere"
    delta_bic: float = 0.0
    bic: float = float("nan")
    bic_sphere: float = float("nan")
    chi2: Dict[str, float] = field(default_factory=dict)
    best_label: str = ""
    results: Dict[str, MatchResult] = field(default_factory=dict)
    threshold: float = 10.0

    @property
    def is_non_spherical(self) -> bool:
        """True when a non-sphere model is preferred at the BIC threshold."""
        return self.family != "sphere" and self.delta_bic > self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary."""
        return {
            "family": self.family,
            "delta_bic": float(self.delta_bic),
            "bic": float(self.bic),
            "bic_sphere": float(self.bic_sphere),
            "chi2": {k: float(v) for k, v in self.chi2.items()},
            "best_label": self.best_label,
            "threshold": float(self.threshold),
            "is_non_spherical": bool(self.is_non_spherical),
        }


def compare_families(
    bank: TemplateBank,
    lightcurve: LightCurve | tuple,
    period: float,
    t0: float,
    duration: float,
    flux_err: Optional[np.ndarray] = None,
    threshold: float = 10.0,
    null_family: str = "sphere",
    optimize_ephemeris: bool = False,
) -> ModelComparison:
    """Compare every family with the sphere using the BIC.

    By default only depth is fitted; ``optimize_ephemeris`` additionally fits
    a bounded epoch offset and duration for every template. All models then
    have the same number of continuous parameters; the discrete template
    search is not accounted for by the BIC penalty. Moving away from
    the spherical null requires ``delta_bic > threshold``; because the sphere
    family includes several radius variants, the test is conservative.
    """
    if optimize_ephemeris:
        results = _optimized_matches(bank, lightcurve, period, t0, duration, flux_err)
    else:
        results = match_bank(bank, lightcurve, period, t0, duration, flux_err=flux_err)
    n_params = 3 if optimize_ephemeris else 1
    usable = [r for r in results if np.isfinite(r.chi2)]
    if not usable:
        return ModelComparison(family=null_family, threshold=threshold)
    n_data = max(usable[0].n_points, 1)
    per_family: Dict[str, MatchResult] = {}
    for result in usable:
        current = per_family.get(result.family)
        if current is None or result.chi2 < current.chi2:
            per_family[result.family] = result
    chi2 = {family: float(r.chi2) for family, r in per_family.items()}
    bic_values = {family: bic(value, n_params, n_data) for family, value in chi2.items()}
    best = min(per_family.values(), key=lambda r: r.chi2)
    null = per_family.get(null_family)
    if null is None:
        raise ValueError(f"bank must contain a usable {null_family!r} template")
    null_chi2 = float(null.chi2)
    delta = float(null_chi2 - best.chi2)
    chosen = best.family if delta > threshold else null_family
    chosen_bic = bic_values.get(chosen, float("nan"))
    return ModelComparison(
        family=chosen,
        delta_bic=delta if best.family != null_family else 0.0,
        bic=float(chosen_bic),
        bic_sphere=float(bic(null_chi2, n_params, n_data)),
        chi2=chi2,
        best_label=best.label,
        results=per_family,
        threshold=float(threshold),
    )


def _optimized_matches(
    bank: TemplateBank, lightcurve: LightCurve | tuple, period: float, t0: float,
    duration: float, flux_err: Optional[np.ndarray],
) -> List[MatchResult]:
    """Fit each shape's timing, duration, and depth without altering BLS period."""
    from scipy.optimize import least_squares
    from shapehunter.utils import phase_fold

    if period <= 0 or duration <= 0 or duration >= period / 2:
        raise ValueError("invalid ephemeris")
    if isinstance(lightcurve, LightCurve):
        time, flux = lightcurve.time, lightcurve.flux
        err = lightcurve.error() if flux_err is None else np.asarray(flux_err, dtype=float)
    else:
        time, flux = map(lambda x: np.asarray(x, dtype=float), lightcurve)
        err = np.full_like(flux, 1e-4) if flux_err is None else np.asarray(flux_err, dtype=float)
    err = np.clip(err, 1e-12, None)
    dt = phase_fold(time, period, t0) * period
    window = np.abs(dt) < 2.0 * duration
    if np.count_nonzero(window) < 5:
        return match_bank(bank, lightcurve, period, t0, duration, flux_err=flux_err)
    x, y, error = dt[window], flux[window], err[window]
    results = []
    for template in bank:
        initial = matched_filter(time, flux, err, period, t0, duration, template)
        amplitude = max(initial.amplitude, 1e-8)
        def residual(theta: np.ndarray) -> np.ndarray:
            offset, width, depth = theta
            return (y - (1.0 - depth * template.evaluate((x - offset) / width))) / error
        result = least_squares(
            residual, [0.0, duration, amplitude],
            bounds=([-0.6 * duration, 0.5 * duration, 0.0],
                    [0.6 * duration, 2.0 * duration, 5.0 * amplitude]),
            x_scale=[duration, duration, amplitude], max_nfev=150,
        )
        offset, width, _ = result.x
        match = matched_filter(time, flux, err, period, t0 + offset, width, template)
        results.append(match)
    return results
