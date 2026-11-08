"""Test-suite fixtures shared by every stage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture()
def rng() -> np.random.Generator:
    """A reproducible generator."""
    return np.random.default_rng(1234)
