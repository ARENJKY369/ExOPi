"""Evaluation on independently labelled, target-level light-curve samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.pipeline.run_star import search_star
from shapehunter.validation.calibration import binomial_interval


@dataclass(frozen=True)
class ReferenceTarget:
    """One independently labelled target, with its source and optional period.

    A negative label means the source explicitly labels the target as a
    negative control; it must not be inferred merely from a missing catalog row.
    """

    target_id: str
    curve: LightCurve
    label: str
    reference_source: str
    expected_period: float | None = None


def evaluate_population(targets: Sequence[ReferenceTarget], min_power: float = 8.0,
                        period_tolerance: float = 0.01) -> dict:
    """Evaluate a fixed threshold once per labelled target and report counts.

    Positive recovery requires a matching period; vetted recovery also requires
    PASS. Any candidate in a negative is a false alarm, even if vetting fails.
    Only completed targets enter rate denominators; excluded targets are listed
    with a reason. Intervals assume independent targets and do not correct for
    selection bias, label error, or correlation across stars.
    """
    if not targets or not np.isfinite(min_power) or min_power <= 0 or not 0 < period_tolerance < 1:
        raise ValueError("nonempty targets and valid threshold/tolerance required")
    ids = [t.target_id.strip().casefold() for t in targets]
    if any(not name for name in ids) or len(ids) != len(set(ids)):
        raise ValueError("target IDs must be nonempty and unique")
    for target in targets:
        if not target.reference_source.strip() or target.label not in {"positive", "negative"}:
            raise ValueError("each target needs a source and a positive/negative label")
        if target.label == "positive" and (target.expected_period is None or not np.isfinite(target.expected_period) or target.expected_period <= 0):
            raise ValueError("positive targets require a finite reference period")
        if target.label == "negative" and target.expected_period is not None:
            raise ValueError("negative targets must not specify an expected period")
    outcomes = []
    for target in targets:
        row = {"target_id": target.target_id, "label": target.label,
               "reference_source": target.reference_source, "expected_period": target.expected_period,
               "baseline": target.curve.baseline}
        if target.label == "positive" and target.curve.baseline < 2 * target.expected_period:
            row.update({"status": "excluded", "reason": "baseline shorter than two reference periods"})
            outcomes.append(row)
            continue
        try:
            candidates = search_star(target.curve, min_power=min_power)
        except (ValueError, RuntimeError) as exc:
            row.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            outcomes.append(row)
            continue
        row.update({"status": "evaluated", "periods": [float(c.period) for c in candidates],
                    "candidate_statuses": [c.status for c in candidates]})
        if target.label == "positive":
            matched = [c for c in candidates if abs(c.period - target.expected_period) <= period_tolerance * target.expected_period]
            row.update({"recovered": bool(matched), "vetted_recovery": any(c.status == "PASS" for c in matched)})
        else:
            row["false_alarm"] = bool(candidates)
        outcomes.append(row)
    positive = [r for r in outcomes if r["status"] == "evaluated" and r["label"] == "positive"]
    negative = [r for r in outcomes if r["status"] == "evaluated" and r["label"] == "negative"]
    def summarize(rows: list[dict], key: str) -> dict:
        count = sum(bool(row[key]) for row in rows)
        return {"successes": count, "trials": len(rows),
                "fraction": count / len(rows) if rows else None,
                "interval_95": binomial_interval(count, len(rows)) if rows else None}
    return {"min_power": min_power, "period_tolerance": period_tolerance, "targets": outcomes,
            "recovery": summarize(positive, "recovered"), "vetted_recovery": summarize(positive, "vetted_recovery"),
            "false_alarm": summarize(negative, "false_alarm"),
            "excluded_or_error": sum(r["status"] != "evaluated" for r in outcomes),
            "note": "These sample statistics are not survey-wide guarantees; selection and labels matter."}
