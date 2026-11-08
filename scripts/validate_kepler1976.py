"""Validate locally supplied Kepler-1976 b quarters against a catalog CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from shapehunter.validation import load_kepler_fits, validate_reference


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Read the catalog and independently evaluate each supplied quarter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--reference-source", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    with args.catalog.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("pl_name") == "Kepler-1976 b"]
    if len(rows) != 1:
        raise ValueError("catalog must contain exactly one Kepler-1976 b row")
    period = float(rows[0]["pl_orbper"])
    results = []
    for path in args.files:
        curve = load_kepler_fits(path)
        if curve.meta["target"].strip().upper() != "KIC 11138155":
            raise ValueError(f"unexpected target in {path}: {curve.meta['target']}")
        result = validate_reference(curve, period, args.reference_source)
        result.update({"file": path.name, "sha256": digest(path), "samples": len(curve)})
        results.append(result)
    payload = {"catalog_sha256": digest(args.catalog), "reference_source": args.reference_source,
               "planet": "Kepler-1976 b", "results": results,
               "note": "One known host across three quarters; this is not an independent-population benchmark."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
