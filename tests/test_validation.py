"""Offline validation and optional dashboard import checks."""

import importlib.util
from pathlib import Path

import pytest

from shapehunter.validation import false_alarm_rate, injection_recovery


def test_validation_runs_real_pipeline() -> None:
    """A seeded injection is recovered and seeded noise yields no candidate."""
    recovered = injection_recovery([7])
    assert recovered["n_trials"] == 1
    assert recovered["n_recovered"] == 1
    assert recovered["recovery_fraction"] == 1
    noise = false_alarm_rate([7])
    assert noise["n_trials"] == 1
    assert noise["n_false_alarms"] == 0
    assert noise["false_alarm_fraction"] == 0


def test_validation_rejects_empty_trials() -> None:
    """An empty experiment cannot define a recovery or false-alarm rate."""
    with pytest.raises(ValueError):
        injection_recovery([])
    with pytest.raises(ValueError):
        false_alarm_rate([])


def test_dashboard_import_does_not_require_streamlit() -> None:
    """Importing the dashboard module does not launch or import Streamlit."""
    path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    spec = importlib.util.spec_from_file_location("dashboard_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
