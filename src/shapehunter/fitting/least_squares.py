"""Least-squares refinement of transit parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from shapehunter.core import LightCurve
from shapehunter.fitting.models import trapezoid_flux
from shapehunter.utils import mad_std

__all__ = ["FitResult", "fit_trapezoid", "refine_period", "estimate_scatter"]


@dataclass
class FitResult:
    """Outcome of a least-squares fit."""

    params: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, float] = field(default_factory=dict)
    chi2: float = float("nan")
    dof: int = 0
    success: bool = False
    message: str = ""
    n_points: int = 0
    residual: Optional[np.ndarray] = None

    @property
    def red_chi2(self) -> float:
        """Reduced chi-square."""
        return float(self.chi2 / self.dof) if self.dof > 0 else float("nan")

    @property
    def snr(self) -> float:
        """Depth signal-to-noise from the fitted uncertainty."""
        depth = self.params.get("depth", 0.0)
        err = self.errors.get("depth", float("nan"))
        if not np.isfinite(err) or err <= 0:
            return float("nan")
        return float(depth / err)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary."""
        return {
            "params": {k: float(v) for k, v in self.params.items()},
            "errors": {k: float(v) for k, v in self.errors.items()},
            "chi2": float(self.chi2),
            "dof": int(self.dof),
            "red_chi2": float(self.red_chi2),
            "success": bool(self.success),
            "message": self.message,
            "n_points": int(self.n_points),
        }


def estimate_scatter(flux: np.ndarray) -> float:
    """Robust per-point scatter of a light curve."""
    return max(mad_std(flux), 1e-8)


def _errors_from_jacobian(jac: np.ndarray, chi2: float, dof: int) -> np.ndarray:
    """Parameter uncertainties from the least-squares Jacobian."""
    try:
        cov = np.linalg.pinv(jac.T @ jac)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return np.full(jac.shape[1], np.nan)
    scale = chi2 / dof if dof > 0 else 1.0
    return np.sqrt(np.clip(np.diag(cov) * scale, 0.0, None))


def fit_trapezoid(
    lightcurve: LightCurve | Tuple[np.ndarray, np.ndarray],
    period: float,
    t0: float,
    duration: float,
    depth: float,
    ingress_fraction: float = 0.2,
    fit_period: bool = True,
    flux_err: Optional[np.ndarray] = None,
    max_nfev: int = 400,
) -> FitResult:
    """Fit a periodic trapezoid transit model to a light curve.

    Free parameters are ``period`` (optional), ``t0``, ``depth``, ``duration``
    and ``ingress_fraction``.
    """
    if isinstance(lightcurve, LightCurve):
        time = lightcurve.time
        flux = lightcurve.flux
        err = lightcurve.flux_err if flux_err is None else flux_err
    else:
        time, flux = lightcurve
        err = flux_err
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if err is None:
        err = np.full_like(flux, estimate_scatter(flux))
    err = np.clip(np.asarray(err, dtype=float), 1e-12, None)

    p0 = [float(period), float(t0), max(float(depth), 1e-6), max(float(duration), 1e-4), float(np.clip(ingress_fraction, 0.0, 1.0))]
    period_step = max(0.02 * float(period), 4.0 * float(np.median(np.diff(np.sort(time)))) / max(len(time), 1))
    lower = [float(period) * (1.0 - 0.05), float(t0) - 0.6 * float(duration), 0.0, 0.3 * float(duration), 0.0]
    upper = [float(period) * (1.0 + 0.05), float(t0) + 0.6 * float(duration), 5.0 * float(depth), 3.0 * float(duration), 1.0]
    lower[0] = max(lower[0], 1e-4)
    del period_step

    def model(theta: np.ndarray) -> np.ndarray:
        per, t_zero, dep, dur, ing = theta
        return trapezoid_flux(time, per, t_zero, dep, dur, ing)

    free = np.arange(5) if fit_period else np.arange(1, 5)

    def expand(theta: np.ndarray) -> np.ndarray:
        full = np.asarray(p0, dtype=float).copy()
        full[free] = theta
        return full

    def residuals(theta: np.ndarray) -> np.ndarray:
        return (flux - model(expand(theta))) / err

    try:
        result = least_squares(
            residuals,
            x0=np.clip(np.asarray(p0)[free], np.array(lower)[free] + 1e-12, np.array(upper)[free] - 1e-12),
            bounds=(np.array(lower)[free], np.array(upper)[free]),
            x_scale=np.array([period * 1e-4, duration * 0.05, depth, duration * 0.05, 0.05])[free],
            max_nfev=int(max_nfev),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:  # pragma: no cover - defensive
        return FitResult(params={}, success=False, message=str(exc), n_points=time.size)

    names = ["period", "t0", "depth", "duration", "ingress_fraction"]
    chi2 = float(np.sum(result.fun**2))
    dof = int(time.size - len(free))
    errs = _errors_from_jacobian(result.jac, chi2, dof)
    params = {name: float(val) for name, val in zip(names, expand(result.x))}
    errors = {name: 0.0 for name in names}
    errors.update({names[i]: float(err) for i, err in zip(free, errs)})
    return FitResult(
        params=params,
        errors=errors,
        chi2=chi2,
        dof=dof,
        success=bool(result.success),
        message=str(result.message),
        n_points=int(time.size),
        residual=result.fun * err,
    )


def refine_period(
    lightcurve: LightCurve | Tuple[np.ndarray, np.ndarray],
    period: float,
    t0: float,
    duration: float,
    depth: float,
    span: float = 0.02,
    n_scan: int = 41,
) -> Tuple[float, float]:
    """Scan and then fit the period locally around a BLS peak.

    Returns ``(period, t0)`` where ``t0`` is the mid-transit time nearest the
    original ephemeris.
    """
    if isinstance(lightcurve, LightCurve):
        time, flux, err = lightcurve.time, lightcurve.flux, lightcurve.error()
    else:
        time, flux = lightcurve
        err = np.full_like(np.asarray(flux, dtype=float), estimate_scatter(np.asarray(flux, dtype=float)))
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.clip(np.asarray(err, dtype=float), 1e-12, None)
    best = (float("inf"), period, t0)
    for trial in period * np.linspace(1.0 - span, 1.0 + span, int(max(n_scan, 3))):
        # Re-anchor t0 so the trial period keeps the same transit epochs.
        n_epoch = np.round((t0 - time.min()) / trial)
        t_trial = t0 - (trial - period) * n_epoch
        model = trapezoid_flux(time, trial, t_trial, depth, duration, 0.2)
        chi2 = float(np.sum(((flux - model) / err) ** 2))
        if chi2 < best[0]:
            best = (chi2, float(trial), float(t_trial))
    return best[1], best[2]
