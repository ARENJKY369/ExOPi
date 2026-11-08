"""Shape families and the pixel-grid transit raycaster."""

from __future__ import annotations

from shapehunter.shapes.limb_darkening import (
    analytic_center_depth,
    intensity_at_radius,
    integrated_flux,
)
from shapehunter.shapes.raycast import simulate_transit, transit_flux_curve
from shapehunter.shapes.families import FAMILIES, build_template, get_family, list_families

__all__ = [
    "simulate_transit",
    "transit_flux_curve",
    "FAMILIES",
    "get_family",
    "list_families",
    "build_template",
    "intensity_at_radius",
    "integrated_flux",
    "analytic_center_depth",
]
