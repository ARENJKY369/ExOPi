"""shapehunter: a shape-aware exoplanet transit search pipeline.

The package is organised in eight layers:

``data``        synthetic light curves, injection, cached lightkurve access
``preprocess``  detrending, normalisation, flare clipping
``shapes``      pixel-grid raycasting of non-spherical occulters
``search``      template bank, BLS, matched filtering, BIC model selection
``vetting``     false-positive rejection diagnostics
``physics``     N-body integration, transit timing variations
``fitting``     transit-shape models and least-squares fits
``pipeline``    CLI, batch runner, scheduler, results database
``report``      candidate report writers
``validation``  injection-recovery and false-alarm-rate studies

Everything runs on CPU with numpy/scipy/astropy only; optional heavy
dependencies (lightkurve, streamlit, emcee, h5py) are imported lazily.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
