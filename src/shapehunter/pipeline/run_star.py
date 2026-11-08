"""Search a synthetic star or an offline saved light curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapehunter.core import Candidate, LightCurve
from shapehunter.data import generate_lightcurve, load_lightcurve, simulate_planet_transit
from shapehunter.preprocess import prepare_lightcurve
from shapehunter.search import TemplateBank, compare_families, run_bls
from shapehunter.vetting import vet_candidate


def search_star(
    curve: LightCurve,
    min_power: float = 8.0,
    max_candidates: int = 1,
    max_periods: int | None = 30000,
) -> list[Candidate]:
    """Search iteratively, masking detected events between BLS passes.

    The empirical screening threshold is not a calibrated false-alarm
    probability. Masking can miss overlapping events or introduce aliases.
    Vetting is run on the data available in each iteration.
    """
    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if not 0 < min_power < float("inf"):
        raise ValueError("min_power must be finite and positive")
    import numpy as np

    remaining = prepare_lightcurve(curve)
    candidates: list[Candidate] = []
    bank = None
    for _ in range(max_candidates):
        if len(remaining) < 30 or remaining.baseline <= 1.0:
            break
        if max_periods == 30000:
            result = run_bls(remaining)
        else:
            result = run_bls(remaining, max_periods=max_periods)
        if not np.isfinite(result.power) or not np.isfinite(result.snr) or result.power < min_power or result.snr < min_power:
            break
        if not np.isfinite(result.duration) or not np.isfinite(result.period) or result.period <= 0 or result.duration <= 0 or result.duration >= result.period / 2:
            break
        if bank is None:
            bank = TemplateBank.build(radii=(0.07, 0.10, 0.12), grid=101, n_phase=81)
        comparison = compare_families(bank, remaining, result.period, result.t0, result.duration, optimize_ephemeris=True)
        template = next((item for item in bank if item.label == comparison.results[comparison.family].label), None)
        ranked = sorted(comparison.chi2.items(), key=lambda item: item[1])
        score_gap = float(ranked[1][1] - ranked[0][1]) if len(ranked) > 1 else None
        plausible = [name for name, chi2 in ranked if chi2 - ranked[0][1] <= comparison.threshold]
        classification = "ambiguous" if len(plausible) > 1 or score_gap is None else "provisional"
        candidate = Candidate(
            period=result.period, t0=result.t0, duration=result.duration,
            depth=result.depth, snr=result.snr, family=comparison.family,
            params=dict(template.params) if template is not None else {},
            n_transits=result.n_transits, delta_bic=comparison.delta_bic,
            star_id=str(remaining.meta.get("target", "")),
            meta={"screening_calibration": "uncalibrated", "family_chi2": dict(comparison.chi2), "sphere_bic": comparison.bic_sphere,
                  "shape_threshold": comparison.threshold, "shape_classification": classification,
                  "shape_score_gap": score_gap, "plausible_shapes": plausible,
                  "oblate_sphere_decision": ("inconclusive" if {"sphere", "oblate"} <= set(plausible)
                                             else "not_tested_or_provisional")},
        )
        candidate.vetting = vet_candidate(remaining, candidate)
        candidate.status = candidate.vetting["status"]
        candidates.append(candidate)
        if len(candidates) == max_candidates:
            break
        phase = (remaining.time - result.t0 + result.period / 2) % result.period - result.period / 2
        keep = np.abs(phase) > 0.75 * result.duration
        if np.all(keep):
            break
        remaining = remaining.mask(keep)
    return candidates


def run_star(
    curve: LightCurve,
    out: str | Path,
    min_power: float = 8.0,
    max_candidates: int = 1,
    max_periods: int | None = 30000,
) -> list[Candidate]:
    """Run a search and write a JSON candidate list and text report."""
    from shapehunter.report.summary import write_report

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = search_star(curve, min_power=min_power, max_candidates=max_candidates, max_periods=max_periods)
    (directory / "candidates.json").write_text(
        json.dumps([candidate.to_dict() for candidate in candidates], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(candidates, directory / "report.txt")
    return candidates


def main(argv: list[str] | None = None) -> int:
    """Parse CLI options and execute the search."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="generate synthetic data")
    source.add_argument("--input", type=Path, help="saved NPZ light curve")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-injection", action="store_true", help="do not inject the synthetic planet")
    parser.add_argument("--max-candidates", type=int, default=1, help="maximum iterative detections (default: 1)")
    parser.add_argument("--min-power", type=float, default=8.0, help="screening threshold for BLS power and fitted SNR")
    parser.add_argument("--max-periods", type=int, default=30000, help="maximum period-grid samples; use 0 for an uncapped precision grid")
    args = parser.parse_args(argv)
    if args.no_injection and not args.synthetic:
        parser.error("--no-injection requires --synthetic")
    if args.synthetic:
        curve = generate_lightcurve(seed=args.seed)
        if not args.no_injection:
            curve = simulate_planet_transit(curve, period=3.5, t0=2.0)
    else:
        curve = load_lightcurve(args.input)
    if args.max_candidates < 1 or not 0 < args.min_power < float("inf"):
        parser.error("--max-candidates and --min-power must be positive")
    max_periods = None if args.max_periods == 0 else args.max_periods
    if max_periods is not None and max_periods < 2:
        parser.error("--max-periods must be 0 or at least 2")
    candidates = run_star(curve, args.out, min_power=args.min_power, max_candidates=args.max_candidates, max_periods=max_periods)
    print(f"Wrote {len(candidates)} candidate(s) to {args.out / 'candidates.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
