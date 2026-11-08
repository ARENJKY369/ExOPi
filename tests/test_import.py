"""Smoke-test the package scaffold without optional network dependencies."""

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "shapehunter", "shapehunter.data", "shapehunter.preprocess",
        "shapehunter.shapes", "shapehunter.shapes.families",
        "shapehunter.search", "shapehunter.vetting", "shapehunter.validation",
        "shapehunter.physics", "shapehunter.fitting", "shapehunter.pipeline",
        "shapehunter.report",
    ],
)
def test_package_imports(name: str) -> None:
    """All documented package namespaces must be importable."""
    assert importlib.import_module(name).__name__ == name


def test_optional_integrations_are_lazy() -> None:
    """Importing the scaffold must not load either optional integration."""
    assert "lightkurve" not in sys.modules
    assert "streamlit" not in sys.modules
