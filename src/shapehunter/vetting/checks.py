"""Offline, diagnostic transit-vetting checks with explicit unknown outcomes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from shapehunter.core import Candidate, LightCurve
from shapehunter.utils import mad_std


def _result(status: str, value: float | None, detail: str) -> dict[str, Any]:
    return {"status": status, "value": value, "detail": detail}


def _significance(values: np.ndarray, mask: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Estimate mean in-event depth and significance relative to reference."""
    if mask.sum() < 3 or reference.sum() < 3:
        return float("nan"), float("nan")
    a, b = values[mask], values[reference]
    sigma = max(float(mad_std(b)), 1e-10)
    depth = float(np.mean(b) - np.mean(a))
    error = sigma * np.sqrt(1 / a.size + 1 / b.size)
    return depth, float(depth / error)


def vet_candidate(
    curve: LightCurve,
    candidate: Candidate,
    known_objects: Sequence[Mapping[str, float]] | None = None,
    systematics_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run offline vetting; missing ancillary information is marked unknown.

    A FAIL requires significant evidence of a false positive or low SNR.
    Asymmetry and stellar variability are informational, not rejection criteria.
    Known-object matching requires a supplied catalogue of period/t0 mappings.
    """
    p, t0, duration = candidate.period, candidate.t0, candidate.duration
    if p <= 0 or duration <= 0 or duration >= p:
        raise ValueError("period and duration must satisfy 0 < duration < period")
    if len(curve) < 3:
        raise ValueError("at least three observations are required")
    dt = (curve.time - t0 + p / 2) % p - p / 2
    epoch = np.rint((curve.time - t0) / p).astype(int)
    primary = np.abs(dt) < duration / 2
    reference = (np.abs(dt) > duration) & (np.abs(dt) < p / 2 - duration)
    secondary = np.abs(np.abs(dt) - p / 2) < duration / 2
    checks: dict[str, Any] = {}
    depth, snr = _significance(curve.flux, primary, reference)
    checks["snr"] = _result("unknown" if not np.isfinite(snr) else ("pass" if snr >= 7 else "fail"),
                             float(snr) if np.isfinite(snr) else None, "Primary transit significance")
    odd = primary & (epoch % 2 != 0)
    even = primary & (epoch % 2 == 0)
    if odd.sum() >= 3 and even.sum() >= 3 and reference.sum() >= 3:
        sigma = max(float(mad_std(curve.flux[reference])), 1e-10)
        # Include within-event flux variation, not just out-of-transit noise:
        # uneven ingress/egress sampling otherwise mimics depth differences.
        odd_var = max(float(np.var(curve.flux[odd], ddof=1)), sigma**2)
        even_var = max(float(np.var(curve.flux[even], ddof=1)), sigma**2)
        odd_even_z = abs(float(np.mean(curve.flux[odd]) - np.mean(curve.flux[even]))) / np.sqrt(odd_var / odd.sum() + even_var / even.sum())
        checks["odd_even"] = _result("fail" if odd_even_z > 5 else "pass", odd_even_z, "Odd/even depth difference in sigma")
    else:
        checks["odd_even"] = _result("unknown", None, "Insufficient odd/even samples")
    sec_depth, sec_snr = _significance(curve.flux, secondary, reference)
    checks["secondary_eclipse"] = _result("unknown" if not np.isfinite(sec_snr) else ("fail" if sec_snr > 5 else "pass"),
                                          float(sec_snr) if np.isfinite(sec_snr) else None, "Secondary eclipse significance")
    left, right = primary & (dt < 0), primary & (dt >= 0)
    if left.sum() >= 3 and right.sum() >= 3:
        difference = float(np.mean(curve.flux[left]) - np.mean(curve.flux[right]))
        checks["asymmetry"] = _result("info", difference, "Ingress minus egress mean flux")
    else:
        checks["asymmetry"] = _result("unknown", None, "Insufficient ingress/egress samples")
    if curve.centroid_x is not None and curve.centroid_y is not None and primary.sum() >= 3 and reference.sum() >= 3:
        shifts = []
        for axis in (curve.centroid_x, curve.centroid_y):
            sigma = max(float(mad_std(axis[reference])), 1e-10)
            shifts.append(float((np.mean(axis[primary]) - np.mean(axis[reference])) / (sigma * np.sqrt(1 / primary.sum() + 1 / reference.sum()))))
        shift = float(np.hypot(*shifts))
        checks["centroid"] = _result("fail" if shift > 5 else "pass", shift, "Centroid shift in combined sigma")
    else:
        checks["centroid"] = _result("unknown", None, "Centroids or samples unavailable")
    if systematics_mask is None:
        checks["systematics"] = _result("unknown", None, "No systematics mask supplied")
    else:
        flags = np.asarray(systematics_mask, dtype=bool)
        if flags.shape != curve.time.shape:
            raise ValueError("systematics mask must match the light curve")
        fraction = float(np.mean(flags[primary])) if primary.any() else None
        checks["systematics"] = _result("unknown" if fraction is None else ("fail" if fraction > 0.5 else "pass"), fraction, "Fraction of transit samples flagged")
    scatter = float(mad_std(curve.flux[reference])) if reference.sum() >= 3 else None
    checks["starspot"] = _result("info" if scatter is not None else "unknown", scatter, "Out-of-transit robust scatter; not a starspot identification")
    if known_objects is None:
        checks["known_objects"] = _result("unknown", None, "No known-object catalogue supplied")
    else:
        matches = 0
        for obj in known_objects:
            op, ot = float(obj["period"]), float(obj["t0"])
            if op > 0 and abs(op - p) < 0.001 * p and abs((ot - t0 + p / 2) % p - p / 2) < duration:
                matches += 1
        checks["known_objects"] = _result("info", float(matches), "Matching supplied ephemerides")
    failed = [name for name, result in checks.items() if result["status"] == "fail"]
    unknown = [name for name, result in checks.items() if result["status"] == "unknown"]
    status = "FAIL" if failed else ("UNKNOWN" if checks["snr"]["status"] == "unknown" else "PASS")
    return {"status": status, "completeness": "incomplete" if unknown else "complete", "checks": checks, "failed": failed, "unknown": unknown}
