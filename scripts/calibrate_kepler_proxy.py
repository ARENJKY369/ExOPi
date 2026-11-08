"""Train and evaluate a threshold on disjoint labelled-flux proxy samples."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.preprocess import prepare_lightcurve
from shapehunter.search import run_bls
from shapehunter.utils import mad_std
from shapehunter.validation.calibration import binomial_interval


def score_row(values: np.ndarray, cadence: float) -> float:
    """Compute the pipeline screening score for an assumed uniform cadence."""
    values = np.asarray(values, dtype=float)
    if values.size < 10 or not np.all(np.isfinite(values)):
        raise ValueError("nonfinite or insufficient flux samples")
    centred = values - np.median(values)
    scale = float(np.max(np.abs(centred)))
    sigma = float(mad_std(np.diff(centred)) / np.sqrt(2.0))
    if not np.isfinite(sigma) or sigma <= 0 or scale <= 0:
        raise ValueError("cannot estimate noise from flux differences")
    factor = 0.001 / scale
    curve = LightCurve(np.arange(values.size) * cadence, 1.0 + centred * factor,
                       np.full(values.size, sigma * factor))
    result = run_bls(prepare_lightcurve(curve))
    # Match search_star's screening and pathological-duration rejection.
    if (not np.isfinite(result.period) or result.period <= 0 or
            not np.isfinite(result.duration) or not 0 < result.duration < result.period / 2 or
            not np.isfinite(result.power) or not np.isfinite(result.snr)):
        return 0.0
    return float(max(0.0, min(result.power, result.snr)))


def calibrate(path: Path, excluded_negative_rows: list[int], training_size: int = 40,
              holdout_size: int = 30, target_rate: float = 0.1,
              seed: int = 912, cadence: float = 0.02043365) -> dict:
    """Fit a threshold on negative labels and measure it on disjoint rows.

    Excluded negative rows must be supplied explicitly so previously inspected
    controls can be kept out of the experiment. All positive labels are used
    for a separate exploratory check; no reference periods are available.
    This is not a real-survey calibration or blind validation of the positives.
    """
    if training_size < 1 or holdout_size < 1 or not 0 < target_rate < 1 or not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("invalid sample size, target rate, or assumed cadence")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0].strip('"').upper() != "LABEL":
            raise ValueError("expected LABEL followed by flux columns")
        rows = list(reader)
    labels = np.asarray([row[0].strip() for row in rows])
    if not np.all(np.isin(labels, ["1", "2"])) or not np.any(labels == "2"):
        raise ValueError("expected labels 1 and 2 with at least one positive")
    excluded = {int(i) for i in excluded_negative_rows}
    if any(i < 0 or i >= len(rows) or labels[i] != "1" for i in excluded):
        raise ValueError("excluded indices must be label-1 rows")
    available = np.asarray([i for i in np.flatnonzero(labels == "1") if int(i) not in excluded], dtype=int)
    if available.size < training_size + holdout_size:
        raise ValueError("not enough independent label-1 rows")
    rank = math.ceil((training_size + 1) * (1.0 - target_rate))
    if rank > training_size:
        raise ValueError("not enough training rows to resolve the requested target")
    chosen = np.random.default_rng(seed).choice(available, training_size + holdout_size, replace=False)
    training = sorted(map(int, chosen[:training_size]))
    holdout = sorted(map(int, chosen[training_size:]))
    positives = list(map(int, np.flatnonzero(labels == "2")))

    def score(index: int) -> float:
        return score_row(np.asarray(rows[index][1:], dtype=float), cadence)

    # A training error invalidates the threshold; never silently drop one.
    training_scores = [{"row_index": i, "score": score(i)} for i in training]
    threshold = float(np.nextafter(sorted(row["score"] for row in training_scores)[rank - 1], np.inf))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("calibration produced a nonpositive or nonfinite threshold")
    def evaluate(indices: list[int]) -> tuple[list[dict], dict]:
        trials = []
        for i in indices:
            try:
                value = score(i)
                trials.append({"row_index": i, "status": "evaluated", "score": value, "detected": bool(value >= threshold)})
            except (ValueError, RuntimeError) as exc:
                trials.append({"row_index": i, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        completed = [row for row in trials if row["status"] == "evaluated"]
        detections = sum(row["detected"] for row in completed)
        return trials, {"detected": detections, "evaluated": len(completed), "errors": len(trials) - len(completed),
                        "fraction": detections / len(completed) if completed else None,
                        "interval_95": binomial_interval(detections, len(completed)) if completed else None}
    noise_trials, noise_summary = evaluate(holdout)
    planet_trials, planet_summary = evaluate(positives)
    return {"dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": seed,
            "excluded_negative_rows": sorted(excluded), "target_rate": target_rate,
            "training_rank": rank, "threshold": threshold, "assumed_cadence_days": cadence,
            "training_scores": training_scores, "label_1_holdout": noise_trials,
            "label_1_summary": noise_summary, "label_2_exploratory": planet_trials,
            "label_2_summary": planet_summary,
            "limitations": "Curated proxy labels; unverified cadence, flux scale, target identity and reference periods. Positives were previously inspected. No claim of representative real-survey calibration."}


def main(argv: list[str] | None = None) -> int:
    """Write a reproducible disjoint proxy calibration report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--exclude-report", type=Path, required=True,
                        help="prior proxy JSON; its label-1 row indices are excluded")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    previous = json.loads(args.exclude_report.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if previous.get("dataset_sha256") != digest:
        raise ValueError("prior report does not match this dataset")
    excluded = [row["row_index"] for row in previous["trials"] if row["label"] == 1]
    result = calibrate(args.dataset, excluded)
    result["excluded_report_sha256"] = hashlib.sha256(args.exclude_report.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"threshold": result["threshold"], "label_1": result["label_1_summary"], "label_2": result["label_2_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
