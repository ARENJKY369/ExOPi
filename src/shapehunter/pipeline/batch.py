"""Run the search sequentially on saved light curves."""

from __future__ import annotations

import argparse
from pathlib import Path

from shapehunter.data import load_lightcurve
from shapehunter.pipeline.results_db import store_candidates
from shapehunter.pipeline.run_star import run_star


def run_batch(paths: list[Path], out: Path) -> None:
    """Write one indexed output directory and database run per input."""
    out.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        run_id = f"{index:04d}-{path.stem}"
        candidates = run_star(load_lightcurve(path), out / run_id)
        store_candidates(out / "results.sqlite", run_id, candidates)


def main(argv: list[str] | None = None) -> int:
    """Parse batch input paths and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    run_batch(args.inputs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
