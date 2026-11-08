"""Empirical threshold calibration with independent synthetic holdout trials."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import beta

from shapehunter.data import generate_lightcurve, simulate_planet_transit
from shapehunter.pipeline.run_star import search_star
from shapehunter.preprocess import prepare_lightcurve
from shapehunter.search import run_bls


def binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the two-sided exact Clopper-Pearson binomial interval."""
    if trials < 1 or not 0 <= successes <= trials or not 0 < confidence < 1:
        raise ValueError("invalid binomial interval parameters")
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def screening_score(seed: int) -> float:
    """Calculate the pipeline's minimum BLS power/refined SNR on one noise curve."""
    result = run_bls(prepare_lightcurve(generate_lightcurve(seed=int(seed))))
    score = min(result.power, result.snr)
    if not np.isfinite(score):
        raise ValueError(f"non-finite screening score for seed {seed}")
    return float(score)


def calibrate_threshold(training_seeds: Sequence[int], target_rate: float = 0.05) -> dict:
    """Set a threshold using a conservative order statistic of training scores.

    Requires enough independent trials to resolve the requested rate. This is
    a synthetic-data ranking rule, not a guarantee about real light curves.
    """
    seeds = [int(seed) for seed in training_seeds]
    if not 0 < target_rate < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique and 0 < target_rate < 1")
    rank = math.ceil((len(seeds) + 1) * (1 - target_rate))
    if rank > len(seeds) or not seeds:
        raise ValueError("not enough training seeds to resolve target_rate")
    scores = [screening_score(seed) for seed in seeds]
    threshold = float(np.nextafter(sorted(scores)[rank - 1], np.inf))
    return {"threshold": threshold, "target_rate": float(target_rate), "training_seeds": seeds,
            "training_scores": scores, "rank": rank,
            "note": "Empirical synthetic threshold; not a real-data false-alarm probability."}


def evaluate_holdout(calibration: dict, noise_seeds: Sequence[int], injection_seeds: Sequence[int],
                     period: float = 3.5, t0: float = 2.0, rp: float = 0.1,
                     period_tolerance: float = 0.01) -> dict:
    """Evaluate frozen threshold on disjoint noise and injection seeds.

    A recovery requires a PASS candidate within relative period tolerance.
    False alarms include candidates that fail vetting, matching pipeline output.
    Intervals reflect trial counts, not systematic differences from real data.
    """
    train = [int(s) for s in calibration["training_seeds"]]
    noise = [int(s) for s in noise_seeds]
    injected = [int(s) for s in injection_seeds]
    groups = [set(train), set(noise), set(injected)]
    if not noise or not injected or any(len(group) != len(seq) for group, seq in zip(groups, (train, noise, injected))) or groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("training, noise, and injection seeds must be nonempty, unique, and disjoint")
    threshold = float(calibration["threshold"])
    if not np.isfinite(threshold) or threshold <= 0 or period <= 0 or rp <= 0 or period_tolerance <= 0:
        raise ValueError("threshold and injection parameters must be positive and finite")
    noise_trials = []
    for seed in noise:
        candidates = search_star(generate_lightcurve(seed=seed), min_power=threshold)
        noise_trials.append({"seed": seed, "detected": bool(candidates)})
    injection_trials = []
    for seed in injected:
        curve = simulate_planet_transit(generate_lightcurve(seed=seed), period=period, t0=t0, rp=rp)
        candidates = search_star(curve, min_power=threshold)
        recovered = any(c.status == "PASS" and abs(c.period - period) <= period_tolerance * period for c in candidates)
        injection_trials.append({"seed": seed, "recovered": recovered, "periods": [float(c.period) for c in candidates]})
    false_alarms = sum(t["detected"] for t in noise_trials)
    recovered = sum(t["recovered"] for t in injection_trials)
    return {"threshold": threshold, "noise_trials": noise_trials, "injection_trials": injection_trials,
            "false_alarm_fraction": false_alarms / len(noise), "false_alarm_interval_95": binomial_interval(false_alarms, len(noise)),
            "recovery_fraction": recovered / len(injected), "recovery_interval_95": binomial_interval(recovered, len(injected)),
            "injection": {"period": period, "t0": t0, "rp": rp, "period_tolerance": period_tolerance},
            "note": "Holdout results describe this synthetic generator, not real-survey performance."}
