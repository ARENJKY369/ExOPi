"""Offline synthetic pipeline validation."""

from shapehunter.validation.calibration import binomial_interval, calibrate_threshold, evaluate_holdout
from shapehunter.validation.experiments import false_alarm_rate, injection_recovery
from shapehunter.validation.shapes import shape_confusion
from shapehunter.validation.population import ReferenceTarget, evaluate_population
from shapehunter.validation.observed_calibration import calibrate_observed_population, observed_screening_score
from shapehunter.validation.real_data import load_flux_csv, load_kepler_fits, load_tess_fits, validate_reference

__all__ = ["false_alarm_rate", "injection_recovery", "ReferenceTarget", "evaluate_population", "calibrate_observed_population", "observed_screening_score", "binomial_interval", "calibrate_threshold", "evaluate_holdout", "shape_confusion", "load_tess_fits", "load_kepler_fits", "load_flux_csv", "validate_reference"]
