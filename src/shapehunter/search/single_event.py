"""Detection of individual transit-like events without requiring periodicity.

A single-event search is useful for long-period planets that only transit once
in the baseline, and for vetting: every individual event of a periodic
candidate should be detected independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from shapehunter.core import LightCurve

__all__ = ["SingleEvent", "find_single_events", "event_mask"]


@dataclass
class SingleEvent:
    """One transit-like dip in a light curve."""

    t_mid: float
    duration: float
    depth: float
    depth_err: float
    snr: float
    n_points: int

    def to_dict(self) -> Dict[str, float]:
        """Serialisable summary."""
        return {
            "t_mid": float(self.t_mid),
            "duration": float(self.duration),
            "depth": float(self.depth),
            "depth_err": float(self.depth_err),
            "snr": float(self.snr),
            "n_points": int(self.n_points),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SingleEvent(t_mid={self.t_mid:.4f}, duration={self.duration:.4f}, "
            f"depth={self.depth:.5f}, snr={self.snr:.1f})"
        )


def _window_limits(time: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.searchsorted(time, time - half, side="left")
    hi = np.searchsorted(time, time + half, side="right")
    return lo, hi


def find_single_events(
    lightcurve: LightCurve | tuple,
    flux_err: Optional[np.ndarray] = None,
    durations: Sequence[float] | None = None,
    min_snr: float = 5.0,
    min_points: int = 3,
    max_events: int = 40,
) -> List[SingleEvent]:
    """Scan for isolated box-shaped dips of several trial durations.

    The statistic is the inverse-variance weighted mean depth in a sliding
    window; events are accepted greedily from the highest SNR down, excluding
    windows that overlap an accepted event.
    """
    if isinstance(lightcurve, LightCurve):
        time, flux = lightcurve.time, lightcurve.flux
        err = lightcurve.error() if flux_err is None else np.asarray(flux_err, dtype=float)
    else:
        time, flux = lightcurve
        flux = np.asarray(flux, dtype=float)
        err = np.full_like(flux, 1e-4) if flux_err is None else np.asarray(flux_err, dtype=float)
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.clip(np.asarray(err, dtype=float), 1e-12, None)
    if time.size < min_points:
        return []
    order = np.argsort(time)
    time, flux, err = time[order], flux[order], err[order]
    baseline = float(time.max() - time.min())
    if durations is None:
        durations = np.geomspace(max(baseline / 500.0, 0.01), max(baseline / 20.0, 0.05), 6)
    durations = np.asarray(list(durations), dtype=float)
    w = 1.0 / err**2
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cwf = np.concatenate([[0.0], np.cumsum(w * flux)])

    best_snr = np.full(time.size, -np.inf)
    best_dur = np.full(time.size, np.nan)
    for duration in durations:
        lo, hi = _window_limits(time, 0.5 * float(duration))
        counts = hi - lo
        sum_w = cw[hi] - cw[lo]
        sum_wf = cwf[hi] - cwf[lo]
        valid = (counts >= min_points) & (sum_w > 0)
        mean = np.where(valid, sum_wf / np.where(sum_w > 0, sum_w, 1.0), 1.0)
        sigma = np.where(valid, 1.0 / np.sqrt(np.where(sum_w > 0, sum_w, 1.0)), np.inf)
        snr = np.where(valid, (1.0 - mean) / sigma, -np.inf)
        better = snr > best_snr
        best_snr = np.where(better, snr, best_snr)
        best_dur = np.where(better, float(duration), best_dur)

    events: List[SingleEvent] = []
    used = np.zeros(time.size, dtype=bool)
    candidates = np.argsort(best_snr)[::-1]
    for idx in candidates:
        snr = float(best_snr[idx])
        if snr < min_snr:
            break
        duration = float(best_dur[idx])
        t_mid = float(time[idx])
        if used[idx] or any(abs(t_mid - event.t_mid) < 0.5 * (duration + event.duration) for event in events):
            continue
        lo, hi = _window_limits(time, 0.5 * duration)
        sel = slice(int(lo[idx]), int(hi[idx]))
        if time[sel].size < min_points:
            continue
        sum_w = float(np.sum(w[sel]))
        depth = float(1.0 - np.sum(w[sel] * flux[sel]) / sum_w)
        depth_err = float(1.0 / np.sqrt(sum_w))
        events.append(
            SingleEvent(
                t_mid=t_mid,
                duration=duration,
                depth=depth,
                depth_err=depth_err,
                snr=snr,
                n_points=int(time[sel].size),
            )
        )
        near = np.abs(time - t_mid) <= 0.5 * duration
        used |= near
        if len(events) >= int(max_events):
            break
    events.sort(key=lambda e: e.t_mid)
    return events


def event_mask(time: np.ndarray, period: float, t0: float, duration: float) -> np.ndarray:
    """Boolean mask of samples inside any transit of the ephemeris."""
    t = np.asarray(time, dtype=float)
    dt = np.mod(t - t0 + 0.5 * period, period) - 0.5 * period
    return np.abs(dt) <= 0.5 * float(duration)
