"""Evaluate sphere-versus-oblate ambiguity on a small seeded synthetic grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapehunter.data import generate_lightcurve, simulate_planet_transit
from shapehunter.pipeline.run_star import search_star


def benchmark(seeds: list[int], flattenings: list[float], impacts: list[float]) -> dict:
    """Record candidate labels and ambiguity for spheres and oblate injections."""
    if not seeds or len(set(seeds)) != len(seeds) or not flattenings or not impacts:
        raise ValueError("seeds, flattenings and impacts must be nonempty; seeds must be unique")
    if any(not 0 <= f < 1 for f in flattenings) or any(not 0 <= b < 1 for b in impacts):
        raise ValueError("flattenings and impacts must lie in [0, 1)")
    rows = []
    scenarios = [("sphere", None)] + [("oblate", f) for f in flattenings]
    for family, flattening in scenarios:
        for impact in impacts:
            for seed in seeds:
                params = {} if flattening is None else {"flattening": flattening}
                curve = simulate_planet_transit(generate_lightcurve(seed=seed), family=family,
                                                params=params, impact_parameter=impact)
                candidates = search_star(curve)
                match = next((c for c in candidates if abs(c.period - 3.5) < 0.035), None)
                rows.append({"injected": family, "flattening": flattening, "impact": impact,
                             "seed": seed, "predicted": match.family if match else "missed",
                             "assessment": match.meta.get("shape_classification") if match else None,
                             "score_gap": match.meta.get("shape_score_gap") if match else None})
    return {"trials": rows, "note": "Finite synthetic stress test; shape assessments are heuristic, not calibrated probabilities."}


def main(argv: list[str] | None = None) -> int:
    """Write the predefined reproducible stress test to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = benchmark([501, 502], [0.0, 0.25, 0.5], [0.0, 0.4])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
