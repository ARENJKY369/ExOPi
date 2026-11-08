"""Core data containers shared by every layer of shapehunter."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

__all__ = ["LightCurve", "Candidate"]


def _as_float_array(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


@dataclass
class LightCurve:
    """A single-star light curve with optional centroids and metadata.

    ``flux`` is relative (out-of-transit ~ 1). ``flux_err`` may be ``None``, in
    which case a robust scatter estimate is used on demand.
    """

    time: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray] = None
    centroid_x: Optional[np.ndarray] = None
    centroid_y: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = _as_float_array(self.time, "time")
        self.flux = _as_float_array(self.flux, "flux")
        if self.time.size != self.flux.size:
            raise ValueError("time and flux must have the same length")
        if self.flux_err is not None:
            self.flux_err = _as_float_array(self.flux_err, "flux_err")
            if self.flux_err.size != self.time.size:
                raise ValueError("flux_err must match the light curve length")
        for name in ("centroid_x", "centroid_y"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _as_float_array(value, name))
                if getattr(self, name).size != self.time.size:
                    raise ValueError(f"{name} must match the light curve length")
        if not np.all(np.isfinite(self.time)):
            raise ValueError("time contains non-finite values")

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self.time.size)

    @property
    def baseline(self) -> float:
        """Time span of the observations."""
        return float(self.time.max() - self.time.min()) if self.time.size else 0.0

    @property
    def cadence(self) -> float:
        """Median sampling interval."""
        return float(np.median(np.diff(np.sort(self.time)))) if self.time.size > 1 else 0.0

    def copy(self) -> "LightCurve":
        """Deep copy of the light curve."""
        return LightCurve(
            time=self.time.copy(),
            flux=self.flux.copy(),
            flux_err=None if self.flux_err is None else self.flux_err.copy(),
            centroid_x=None if self.centroid_x is None else self.centroid_x.copy(),
            centroid_y=None if self.centroid_y is None else self.centroid_y.copy(),
            meta=dict(self.meta),
        )

    def with_flux(self, flux: np.ndarray, **meta: Any) -> "LightCurve":
        """Return a copy carrying new flux values (errors and centroids kept)."""
        out = self.copy()
        values = _as_float_array(flux, "flux")
        if values.size != len(out):
            raise ValueError("flux must match the light curve length")
        out.flux = values
        out.meta.update(meta)
        return out

    def mask(self, keep: np.ndarray) -> "LightCurve":
        """Return the subset of points where ``keep`` is true."""
        keep = np.asarray(keep, dtype=bool)
        return LightCurve(
            time=self.time[keep],
            flux=self.flux[keep],
            flux_err=None if self.flux_err is None else self.flux_err[keep],
            centroid_x=None if self.centroid_x is None else self.centroid_x[keep],
            centroid_y=None if self.centroid_y is None else self.centroid_y[keep],
            meta=dict(self.meta),
        )

    def sort(self) -> "LightCurve":
        """Return a time-sorted copy."""
        order = np.argsort(self.time)
        return self.mask(np.ones(len(self), dtype=bool)).__class__(
            time=self.time[order],
            flux=self.flux[order],
            flux_err=None if self.flux_err is None else self.flux_err[order],
            centroid_x=None if self.centroid_x is None else self.centroid_x[order],
            centroid_y=None if self.centroid_y is None else self.centroid_y[order],
            meta=dict(self.meta),
        )

    def error(self) -> np.ndarray:
        """Flux uncertainties, falling back to a robust scatter estimate."""
        if self.flux_err is not None:
            return self.flux_err
        if self.time.size < 3:
            return np.ones_like(self.flux) * 1e-4
        mad = np.median(np.abs(self.flux - np.median(self.flux)))
        sigma = 1.4826 * mad
        return np.full_like(self.flux, max(float(sigma), 1e-8))

    def relative_flux(self) -> "LightCurve":
        """Return a copy with the flux divided by its median."""
        med = float(np.median(self.flux)) if self.flux.size else 1.0
        if med == 0.0:
            med = 1.0
        err = None if self.flux_err is None else self.flux_err / abs(med)
        return replace(self, flux=self.flux / med, flux_err=err)


@dataclass
class Candidate:
    """A transit candidate produced by the search stage."""

    period: float
    t0: float
    duration: float
    depth: float
    snr: float
    family: str = "sphere"
    params: Dict[str, float] = field(default_factory=dict)
    vetting: Dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"
    star_id: str = ""
    delta_bic: float = 0.0
    n_transits: float = 0.0
    a_over_rstar: float = 0.0
    impact_parameter: float = 0.0
    epoch: float = 0.0
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation (plain Python types only)."""
        def convert(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            if isinstance(value, np.ndarray):
                return convert(value.tolist())
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, (np.bool_,)):
                return bool(value)
            return value

        return convert(
            {
                "star_id": self.star_id,
                "period": self.period,
                "t0": self.t0,
                "duration": self.duration,
                "depth": self.depth,
                "snr": self.snr,
                "family": self.family,
                "params": self.params,
                "vetting": self.vetting,
                "status": self.status,
                "delta_bic": self.delta_bic,
                "n_transits": self.n_transits,
                "a_over_rstar": self.a_over_rstar,
                "impact_parameter": self.impact_parameter,
                "epoch": self.epoch,
                "score": self.score,
                "meta": self.meta,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Candidate":
        """Rebuild a candidate from :meth:`to_dict` output."""
        known = {
            "star_id",
            "period",
            "t0",
            "duration",
            "depth",
            "snr",
            "family",
            "params",
            "vetting",
            "status",
            "delta_bic",
            "n_transits",
            "a_over_rstar",
            "impact_parameter",
            "epoch",
            "score",
            "meta",
        }
        payload = {k: v for k, v in data.items() if k in known}
        payload.setdefault("params", {})
        payload.setdefault("vetting", {})
        payload["params"] = {str(k): float(v) for k, v in dict(payload["params"]).items()}
        payload["meta"] = dict(payload.get("meta") or {})
        for key in ("period", "t0", "duration", "depth", "snr"):
            payload[key] = float(payload[key])
        return cls(**payload)

    def summary(self) -> str:
        """One-line human readable summary."""
        return (
            f"{self.star_id or 'star'}: P={self.period:.5f} d, depth={self.depth:.5f}, "
            f"T={self.duration:.4f} d, SNR={self.snr:.1f}, family={self.family}, "
            f"status={self.status}"
        )
