"""Baseline Box Least Squares period search (astropy backend).

The BLS box model is deliberately *not* shape-aware: it is the standard
workhorse that tells us where to look. Shape information enters afterwards
through the template bank and BIC model selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from astropy.timeseries import BoxLeastSquares

from shapehunter.core import LightCurve
from shapehunter.fitting.least_squares import fit_trapezoid, refine_period
from shapehunter.utils import phase_fold

__all__ = ["BLSResult", "run_bls", "default_durations", "refine_result", "period_grid"]


@dataclass
class BLSResult:
    """Best box found by the periodogram, optionally refined."""

    period: float
    t0: float
    duration: float
    depth: float
    power: float
    depth_err: float = float("nan")
    n_transits: float = 0.0
    objective: str = "snr"
    periods: np.ndarray = field(default_factory=lambda: np.array([]))
    power_spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    refined: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def snr(self) -> float:
        """Depth significance (refined value when available)."""
        if "snr" in self.extra:
            return float(self.extra["snr"])
        if not np.isfinite(self.depth_err) or self.depth_err <= 0:
            return float(self.power)
        return float(self.depth / self.depth_err)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary (periodogram arrays omitted)."""
        return {
            "period": float(self.period),
            "t0": float(self.t0),
            "duration": float(self.duration),
            "depth": float(self.depth),
            "power": float(self.power),
            "depth_err": float(self.depth_err),
            "snr": float(self.snr),
            "n_transits": float(self.n_transits),
            "objective": self.objective,
            "refined": bool(self.refined),
            "extra": {k: v for k, v in self.extra.items() if np.isscalar(v)},
        }


def default_durations(baseline: float, n_durations: int = 8) -> np.ndarray:
    """Logarithmically spaced trial box durations (days)."""
    span = max(float(baseline), 1.0)
    lo = max(0.015, span / 2000.0)
    hi = min(0.60, span / 12.0)
    if hi <= lo * 1.5:
        hi = lo * 1.5
    return np.geomspace(lo, hi, int(max(n_durations, 3)))


def period_grid(
    period_min: float,
    period_max: float,
    baseline: float,
    oversample: float = 5.0,
    max_periods: int = 30000,
    duration: float = 0.05,
) -> np.ndarray:
    """Frequency grid with standard BLS oversampling, returned as periods.

    Periods are ordered from long to short. The grid is capped at
    ``max_periods`` samples, which can reduce sensitivity to narrow transits.
    """
    if period_min <= 0 or period_max <= period_min or baseline <= 0 or duration <= 0 or oversample <= 0 or max_periods < 2:
        raise ValueError("invalid period grid parameters")
    f_min = 1.0 / period_max
    f_max = 1.0 / period_min
    n_freq = int(min(max_periods, max(2, np.ceil(oversample * baseline**2 * (f_max - f_min) / duration))))
    freqs = np.linspace(f_min, f_max, n_freq)
    return 1.0 / freqs


def _unpack(
    lightcurve: LightCurve | Tuple[np.ndarray, np.ndarray],
    flux_err: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(lightcurve, LightCurve):
        err = lightcurve.error() if flux_err is None else np.asarray(flux_err, dtype=float)
        return lightcurve.time, lightcurve.flux, err
    time, flux = lightcurve
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.full_like(flux, 1e-4) if flux_err is None else np.asarray(flux_err, dtype=float)
    return time, flux, np.clip(err, 1e-12, None)


def run_bls(
    lightcurve: LightCurve | Tuple[np.ndarray, np.ndarray],
    flux_err: Optional[np.ndarray] = None,
    period_min: float = 0.5,
    period_max: Optional[float] = None,
    durations: Optional[Sequence[float]] = None,
    minimal_n_transits: int = 2,
    oversample: float = 5.0,
    objective: str = "snr",
    refine: bool = True,
) -> BLSResult:
    """Run a BLS periodogram and (optionally) refine the best box.

    Parameters
    ----------
    lightcurve:
        A :class:`~shapehunter.core.LightCurve` or ``(time, flux)`` tuple.
    period_min, period_max:
        Search range in days. ``period_max`` defaults to half the baseline,
        which guarantees at least two transits.
    durations:
        Trial box durations in days.
    refine:
        When true, the peak is re-centred with :func:`refine_period` and a
        trapezoid least-squares fit, which yields the reported depth and SNR.
    """
    time, flux, err = _unpack(lightcurve, flux_err)
    baseline = float(time.max() - time.min())
    if period_max is None:
        period_max = max(baseline / 2.0, 2.0 * period_min)
    if minimal_n_transits < 2:
        raise ValueError("minimal_n_transits must be at least 2")
    period_max = float(min(period_max, baseline / (minimal_n_transits - 1)))
    if period_max <= period_min:
        raise ValueError("period_max must exceed period_min")
    durations_arr = np.asarray(
        durations if durations is not None else default_durations(baseline),
        dtype=float,
    )
    max_duration = min(baseline / 3.0, 0.9 * float(period_min))
    durations_arr = durations_arr[(durations_arr > 0.0) & (durations_arr < max_duration)]
    if durations_arr.size == 0:
        raise ValueError("no trial duration is positive and shorter than the minimum period")

    periods = period_grid(period_min, period_max, baseline, oversample=oversample, duration=float(np.min(durations_arr)))
    model = BoxLeastSquares(time, flux, err)
    try:
        results = model.power(
            periods, durations_arr, objective=objective, method="fast", oversample=1
        )
    except TypeError:  # pragma: no cover - older astropy
        results = model.power(periods, durations_arr, objective=objective)
    power = np.asarray(results.power, dtype=float)
    best = int(np.argmax(power))
    period = float(periods[best])
    duration = float(np.atleast_1d(results.duration)[best])
    depth = float(np.atleast_1d(results.depth)[best])
    depth_err = float(np.atleast_1d(results.depth_err)[best])
    t0 = float(np.atleast_1d(results.transit_time)[best])
    stats = model.compute_stats(period, duration, t0)
    n_transits = int(np.count_nonzero(np.asarray(stats["per_transit_count"]) > 0))

    result = BLSResult(
        period=period,
        t0=t0,
        duration=duration,
        depth=depth,
        power=float(power[best]),
        depth_err=depth_err,
        n_transits=n_transits,
        objective=objective,
        periods=periods,
        power_spectrum=power,
        extra={"duration_grid": durations_arr, "minimal_n_transits": minimal_n_transits},
    )
    if refine and depth > 0:
        return refine_result(lightcurve, result, flux_err=err)
    return result


def refine_result(
    lightcurve: LightCurve | Tuple[np.ndarray, np.ndarray],
    result: BLSResult,
    flux_err: Optional[np.ndarray] = None,
) -> BLSResult:
    """Re-centre a BLS peak with a local period scan and a trapezoid fit."""
    time, flux, err = _unpack(lightcurve, flux_err)
    try:
        period, t0 = refine_period((time, flux), result.period, result.t0, result.duration, max(result.depth, 1e-6), span=0.02)
        fit = fit_trapezoid(
            LightCurve(time=time, flux=flux, flux_err=err),
            period=period,
            t0=t0,
            duration=result.duration,
            depth=max(result.depth, 1e-6),
            flux_err=err,
        )
    except (ValueError, RuntimeError):  # pragma: no cover - defensive
        return result
    if not fit.success or not fit.params:
        return result
    out = BLSResult(
        period=float(fit.params["period"]),
        t0=float(fit.params["t0"]),
        duration=float(fit.params["duration"]),
        depth=float(fit.params["depth"]),
        power=float(result.power),
        depth_err=float(fit.errors.get("depth", float("nan"))),
        n_transits=float(phase_coverage(time, fit.params["period"], fit.params["t0"], fit.params["duration"])),
        objective=result.objective,
        periods=result.periods,
        power_spectrum=result.power_spectrum,
        refined=True,
        extra={
            "snr": float(fit.snr),
            "ingress_fraction": float(fit.params["ingress_fraction"]),
            "red_chi2": float(fit.red_chi2),
            "t0_error": float(fit.errors.get("t0", float("nan"))),
            "duration_error": float(fit.errors.get("duration", float("nan"))),
            "period_error": float(fit.errors.get("period", float("nan"))),
        },
    )
    return out


def phase_coverage(time: np.ndarray, period: float, t0: float, duration: float) -> int:
    """Number of distinct transits covered by the data."""
    phase = phase_fold(time, period, t0)
    in_transit = np.abs(phase) < 0.5 * duration / period
    if not np.any(in_transit):
        return 0
    epochs = np.round((time[in_transit] - t0) / period)
    return int(np.unique(epochs).size)
