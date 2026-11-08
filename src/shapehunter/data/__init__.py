"""Light-curve data access: synthetic generation, injection and IO."""

from __future__ import annotations

from shapehunter.data.synthetic import (
    LightCurve,
    generate_lightcurve,
    inject_transit,
    simulate_eclipsing_binary,
    simulate_planet_transit,
)
from shapehunter.data.io import (
    download_lightkurve,
    load_lightcurve,
    save_lightcurve,
)

__all__ = [
    "LightCurve",
    "generate_lightcurve",
    "inject_transit",
    "simulate_eclipsing_binary",
    "simulate_planet_transit",
    "save_lightcurve",
    "load_lightcurve",
    "download_lightkurve",
]
