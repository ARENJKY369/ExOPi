"""Evaluate locally supplied labelled light curves from a JSON manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shapehunter.validation import ReferenceTarget, evaluate_population, load_flux_csv, load_kepler_fits, load_tess_fits


def evaluate_manifest(path: Path, min_power: float = 8.0, period_tolerance: float = 0.01) -> dict:
    """Evaluate one row per target; relative data paths resolve beside manifest."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a nonempty list")
    targets = []
    files = []
    for entry in entries:
        data_path = (path.parent / entry["path"]).resolve()
        kind = entry["format"]
        if kind == "kepler_fits":
            curve = load_kepler_fits(data_path)
        elif kind == "tess_fits":
            curve = load_tess_fits(data_path)
        elif kind == "flux_csv":
            curve = load_flux_csv(data_path, entry["target_id"])
        else:
            raise ValueError(f"unknown file format {kind!r}")
        if kind != "flux_csv" and entry.get("expected_header_target"):
            if curve.meta["target"].strip().casefold() != entry["expected_header_target"].strip().casefold():
                raise ValueError(f"target header mismatch for {entry['target_id']}")
        targets.append(ReferenceTarget(entry["target_id"], curve, entry["label"], entry["reference_source"], entry.get("expected_period")))
        files.append({"target_id": entry["target_id"], "path": entry["path"],
                      "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                      "data_source": entry["data_source"], "format": kind})
    result = evaluate_population(targets, min_power=min_power, period_tolerance=period_tolerance)
    result["files"] = files
    result["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main(argv: list[str] | None = None) -> int:
    """Run a manifest benchmark without downloading data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-power", type=float, default=8.0)
    parser.add_argument("--period-tolerance", type=float, default=0.01)
    args = parser.parse_args(argv)
    result = evaluate_manifest(args.manifest, args.min_power, args.period_tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluated {len(result['targets'])} targets; {result['excluded_or_error']} excluded or failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
