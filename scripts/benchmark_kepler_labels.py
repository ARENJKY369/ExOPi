"""Exploratory detection-rate benchmark for a labelled Kepler-flux CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from shapehunter.core import LightCurve
from shapehunter.pipeline.run_star import search_star
from shapehunter.utils import mad_std
from shapehunter.validation.calibration import binomial_interval


def evaluate(path: Path, negatives: int = 30, seed: int = 422, cadence: float = 0.02043365) -> dict:
    """Measure detections by label on all positives and sampled negative labels.

    This curated table has no target IDs, original time array, error bars, or
    reference periods. Cadence and relative-flux scale are assumptions, not
    measured values. Detection of any period is not confirmed-planet recovery.
    The label-1 fraction is not a calibrated survey false-alarm rate.
    """
    if negatives < 1 or not np.isfinite(cadence) or cadence <= 0:
        raise ValueError("negative count and assumed cadence must be positive")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0].strip('"').upper() != "LABEL":
            raise ValueError("expected label followed by flux columns")
        rows = list(reader)
    labels = np.asarray([row[0].strip() for row in rows])
    if not np.all(np.isin(labels, ["1", "2"])) or not np.any(labels == "2"):
        raise ValueError("expected labels 1 and 2 with at least one positive")
    available = np.flatnonzero(labels == "1")
    if negatives > len(available):
        raise ValueError("not enough label-1 rows")
    rng = np.random.default_rng(seed)
    selected = np.sort(np.concatenate((np.flatnonzero(labels == "2"), rng.choice(available, negatives, replace=False))))
    trials = []
    for index in selected:
        label = labels[index]
        record = {"row_index": int(index), "label": int(label)}
        try:
            values = np.asarray(rows[index][1:], dtype=float)
            if values.size < 10 or not np.all(np.isfinite(values)):
                raise ValueError("nonfinite or insufficient flux samples")
            centred = values - np.median(values)
            scale = float(np.max(np.abs(centred)))
            sigma = float(mad_std(np.diff(centred)) / np.sqrt(2.0))
            if not np.isfinite(sigma) or sigma <= 0 or scale <= 0:
                raise ValueError("cannot estimate noise from flux differences")
            # An arbitrary affine scaling places zero-centred table values near
            # unity. It cannot reconstruct the original relative transit depth.
            factor = 0.001 / scale
            lc = LightCurve(np.arange(values.size) * cadence, 1.0 + centred * factor,
                            np.full(values.size, sigma * factor), meta={"target": f"row-{index}"})
            candidates = search_star(lc)
            record.update({"status": "evaluated", "detected": bool(candidates),
                           "periods_assuming_cadence": [float(c.period) for c in candidates],
                           "vetting": [c.status for c in candidates]})
        except (ValueError, RuntimeError) as exc:
            record.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        trials.append(record)
    summary = {}
    for label in (1, 2):
        group = [r for r in trials if r["label"] == label and r["status"] == "evaluated"]
        count = sum(r["detected"] for r in group)
        summary[str(label)] = {"detected": count, "evaluated": len(group),
                               "fraction": count / len(group) if group else None,
                               "interval_95": binomial_interval(count, len(group)) if group else None}
    return {"dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sample_seed": seed,
            "assumed_cadence_days": cadence, "selection": "all label-2 rows plus random label-1 rows",
            "trials": trials, "summary": summary,
            "limitations": "Curated labels, unknown identities, unknown times and uncertainties, assumed cadence and arbitrary flux scale; no true-period or survey-representativeness validation."}


def main(argv: list[str] | None = None) -> int:
    """Run the proxy benchmark on a local CSV or gzipped CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--negatives", type=int, default=30)
    parser.add_argument("--seed", type=int, default=422)
    parser.add_argument("--cadence", type=float, default=0.02043365)
    args = parser.parse_args(argv)
    result = evaluate(args.dataset, args.negatives, args.seed, args.cadence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
