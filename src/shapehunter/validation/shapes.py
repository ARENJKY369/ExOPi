"""Synthetic injection experiments for shape-family confusion."""

from __future__ import annotations

from typing import Sequence

from shapehunter.data import generate_lightcurve, simulate_planet_transit
from shapehunter.pipeline.run_star import search_star
from shapehunter.shapes import FAMILIES


def shape_confusion(seeds: Sequence[int], families: Sequence[str] | None = None, period: float = 3.5,
                    t0: float = 2.0, rp: float = 0.1, min_power: float = 8.0,
                    period_tolerance: float = 0.01, impact_parameter: float = 0.0) -> dict:
    """Inject each family per seed and count classifications and missed events.

    Only period-matched candidates are classified; other detections count as
    missed. These experiments measure the current finite template bank, not
    physical classification reliability for real stars.
    """
    seeds = [int(s) for s in seeds]
    names = list(FAMILIES) if families is None else list(families)
    if not seeds or len(seeds) != len(set(seeds)) or not names or len(names) != len(set(names)):
        raise ValueError("provide unique, nonempty seeds and families")
    if any(name not in FAMILIES for name in names) or period <= 0 or rp <= 0 or min_power <= 0 or period_tolerance <= 0 or not 0 <= impact_parameter < 1:
        raise ValueError("unknown family or invalid injection parameters")
    matrix = {name: {**{other: 0 for other in FAMILIES}, "missed": 0} for name in names}
    trials = []
    for name in names:
        for seed in seeds:
            curve = simulate_planet_transit(generate_lightcurve(seed=seed), period=period, t0=t0, rp=rp, family=name, impact_parameter=impact_parameter)
            candidates = search_star(curve, min_power=min_power)
            matched = next((c for c in candidates if abs(c.period - period) <= period_tolerance * period), None)
            predicted = matched.family if matched is not None else "missed"
            matrix[name][predicted] += 1
            trials.append({"seed": seed, "injected": name, "predicted": predicted,
                           "status": matched.status if matched is not None else None})
    return {"matrix": matrix, "trials": trials, "period": period, "rp": rp, "impact_parameter": impact_parameter, "min_power": min_power,
            "note": "Classification at one set of injected parameters; expand the grid before drawing conclusions."}
