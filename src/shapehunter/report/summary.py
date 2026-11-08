"""Plain-text candidate reports."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from shapehunter.core import Candidate


def write_report(candidates: Sequence[Candidate], path: str | Path) -> Path:
    """Write a human-readable summary without claiming planet confirmation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Shapehunter candidate report", f"Candidates: {len(candidates)}"]
    for number, c in enumerate(candidates, 1):
        lines.append(f"{number}. period={c.period:.6f} d, t0={c.t0:.6f}, duration={c.duration:.6f} d, depth={c.depth:.6g}, SNR={c.snr:.2f}, family={c.family}, vetting={c.status}")
        if c.meta.get("screening_calibration") == "uncalibrated":
            lines.append("   Screening threshold is not calibrated for real observations.")
        if c.meta.get("shape_classification"):
            lines.append("   Shape assessment: " + c.meta["shape_classification"] + " (not a confirmed classification)")
            lines.append("   Plausible shapes (heuristic): " + ", ".join(c.meta.get("plausible_shapes", [])))
            if c.meta.get("oblate_sphere_decision") == "inconclusive":
                lines.append("   Sphere versus oblate: inconclusive; neither can be ruled out by this template comparison.")
        if c.meta.get("family_chi2"):
            lines.append("   Family chi-square (lower is better at fixed ephemeris): " + ", ".join(f"{name}={value:.2f}" for name, value in c.meta["family_chi2"].items()))
        lines.append("   Vetting completeness: " + c.vetting.get("completeness", "unknown"))
        lines.append("   Unknown checks: " + (", ".join(c.vetting.get("unknown", [])) or "none"))
    lines.append("Candidates require independent follow-up; they are not confirmed planets.")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
