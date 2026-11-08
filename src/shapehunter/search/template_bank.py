"""A bank of raycast transit templates, one per shape family.

Every template is expressed on a *normalised phase* axis: ``phi = -0.5`` is
first contact, ``phi = 0`` mid-transit and ``phi = +0.5`` last contact. The
stored ``shape`` is the normalised occultation profile ``(1 - flux) / peak`` so
that a single amplitude (the transit depth) can be fitted linearly for every
family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from shapehunter.shapes.families import FAMILIES, default_params, get_family
from shapehunter.shapes.raycast import simulate_transit

__all__ = ["Template", "TemplateBank", "template_profile"]


def template_profile(
    family: str,
    params: Mapping[str, float],
    n_phase: int = 121,
    grid: int = 201,
    u: Sequence[float] = (0.4, 0.25),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Raycast a normalised profile for one family.

    Returns ``(phi, shape, peak_depth)`` where ``phi`` runs from -0.5 to 0.5 and
    ``shape = (1 - flux) / peak_depth``.
    """
    fam = get_family(family)
    extent = float(fam.extent(params))
    half = 1.0 + extent
    x = np.linspace(-half, half, int(n_phase))
    flux = simulate_transit(np.zeros_like(x), x, np.zeros_like(x), family=family, params=params, u=tuple(u), grid=int(grid))
    profile = 1.0 - np.asarray(flux, dtype=float)
    peak = float(profile.max()) if profile.size else 0.0
    shape = profile / peak if peak > 0 else np.zeros_like(profile)
    phi = x / (2.0 * half)
    return phi, shape, peak


@dataclass(frozen=True)
class Template:
    """A single normalised transit shape."""

    family: str
    params: Dict[str, float]
    phi: np.ndarray
    shape: np.ndarray
    peak_depth: float
    label: str = ""

    def evaluate(self, phi: np.ndarray) -> np.ndarray:
        """Interpolate the profile at ``phi`` (zero outside the transit)."""
        values = np.interp(phi, self.phi, self.shape, left=0.0, right=0.0)
        return np.where(np.abs(phi) <= 0.5, values, 0.0)


@dataclass
class TemplateBank:
    """An ordered collection of templates plus lookup helpers."""

    templates: List[Template] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.templates)

    def __iter__(self):
        return iter(self.templates)

    def families(self) -> List[str]:
        """Unique family names in bank order."""
        seen: List[str] = []
        for template in self.templates:
            if template.family not in seen:
                seen.append(template.family)
        return seen

    def by_family(self) -> Dict[str, List[Template]]:
        """Templates grouped by family."""
        groups: Dict[str, List[Template]] = {}
        for template in self.templates:
            groups.setdefault(template.family, []).append(template)
        return groups

    @classmethod
    def build(
        cls,
        families: Optional[Iterable[str]] = None,
        radii: Sequence[float] = (0.07, 0.12),
        n_phase: int = 121,
        grid: int = 201,
        u: Sequence[float] = (0.4, 0.25),
        overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> "TemplateBank":
        """Raycast the default template for every family and radius."""
        names = list(families) if families is not None else list(FAMILIES)
        overrides = dict(overrides or {})
        templates: List[Template] = []
        for name in names:
            base = dict(overrides.get(name, {}))
            for rp in radii:
                params = default_params(name, float(rp))
                params.update({k: float(v) for k, v in base.items()})
                params["rp"] = float(rp)
                phi, shape, peak = template_profile(name, params, n_phase=n_phase, grid=grid, u=u)
                templates.append(
                    Template(
                        family=name,
                        params=dict(params),
                        phi=phi,
                        shape=shape,
                        peak_depth=peak,
                        label=f"{name}:rp={rp:.3f}",
                    )
                )
        return cls(templates=templates)
