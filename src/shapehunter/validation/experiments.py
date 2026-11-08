"""Reproducible synthetic injection-recovery and false-alarm experiments."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from shapehunter.data import generate_lightcurve, simulate_planet_transit
from shapehunter.pipeline.run_star import search_star


def injection_recovery(seeds: Sequence[int], period: float = 3.5, t0: float = 2.0, rp: float = 0.1, period_tolerance: float = 0.01) -> dict:
    """Inject one planet per seed and record exact-period recovery.

    Recovery requires a PASS candidate within the given relative period
    tolerance; harmonic matches are deliberately not counted as recoveries.
    """
    if len(seeds) == 0 or period <= 0 or rp <= 0 or period_tolerance <= 0:
        raise ValueError("supply seeds and positive period, radius, and tolerance")
    trials = []
    for seed in seeds:
        curve = simulate_planet_transit(generate_lightcurve(seed=int(seed)), period=period, t0=t0, rp=rp)
        candidates = search_star(curve)
        recovered = any(c.status == "PASS" and abs(c.period - period) <= period_tolerance * period for c in candidates)
        trials.append({"seed": int(seed), "recovered": recovered, "periods": [float(c.period) for c in candidates]})
    return {"trials": trials, "n_trials": len(trials), "n_recovered": sum(t["recovered"] for t in trials), "recovery_fraction": float(np.mean([t["recovered"] for t in trials]))}


def false_alarm_rate(seeds: Sequence[int]) -> dict:
    """Measure fraction of noise-only synthetic trials with any candidate.

    This is an empirical finite-sample rate for this generator and threshold,
    not a universal or statistically calibrated false-alarm probability.
    """
    if len(seeds) == 0:
        raise ValueError("supply at least one seed")
    trials = []
    for seed in seeds:
        candidates = search_star(generate_lightcurve(seed=int(seed)))
        trials.append({"seed": int(seed), "false_alarm": bool(candidates), "n_candidates": len(candidates)})
    return {"trials": trials, "n_trials": len(trials), "n_false_alarms": sum(t["false_alarm"] for t in trials), "false_alarm_fraction": float(np.mean([t["false_alarm"] for t in trials]))}
