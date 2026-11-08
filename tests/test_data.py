"""Deterministic synthetic data, injection, and offline cache tests."""

from pathlib import Path

import numpy as np

from shapehunter.data import (
    download_lightkurve, generate_lightcurve, load_lightcurve, save_lightcurve,
    simulate_planet_transit,
)
from shapehunter.data.io import cache_path


def test_seeded_generation_and_injection() -> None:
    """A seed reproduces data, and injection reduces flux without mutation."""
    original = generate_lightcurve(seed=7)
    repeat = generate_lightcurve(seed=7)
    np.testing.assert_array_equal(original.time, repeat.time)
    np.testing.assert_array_equal(original.flux, repeat.flux)
    injected = simulate_planet_transit(original, period=3.5, t0=2.0)
    assert np.any(injected.flux < original.flux * 0.995)
    assert np.all(injected.flux <= original.flux)
    assert "injected" not in original.meta
    assert injected.meta["injected"]["period"] == 3.5


def test_roundtrip_and_cached_download(tmp_path: Path) -> None:
    """A cached light curve loads offline without the optional dependency."""
    original = generate_lightcurve(seed=12, n_points=100)
    path = cache_path(tmp_path, "synthetic target")
    save_lightcurve(original, path)
    for loaded in (load_lightcurve(path), download_lightkurve("synthetic target", cache_dir=tmp_path)):
        np.testing.assert_array_equal(loaded.time, original.time)
        np.testing.assert_array_equal(loaded.flux, original.flux)
        np.testing.assert_array_equal(loaded.centroid_x, original.centroid_x)
        assert loaded.meta == original.meta


def test_download_cache_is_keyed_by_full_request(tmp_path: Path, monkeypatch) -> None:
    """Different cadence or author requests never reuse another cache entry."""
    import sys
    from types import SimpleNamespace

    calls = []
    class FakeCurve:
        time = SimpleNamespace(value=np.array([1., 2., 3.]))
        flux = SimpleNamespace(value=np.array([1., 0.99, 1.]))
        flux_err = SimpleNamespace(value=np.array([0.001, 0.001, 0.001]))
        def remove_nans(self):
            return self
        def normalize(self):
            return self
    class Search:
        def __len__(self):
            return 1
        def download_all(self):
            return SimpleNamespace(stitch=lambda: FakeCurve())
    def lookup(target, mission, author, cadence):
        calls.append((target, mission, author, cadence))
        return Search()
    monkeypatch.setitem(sys.modules, "lightkurve", SimpleNamespace(search_lightcurve=lookup))
    for cadence, author in [("fast", None), ("fast", None), ("long", None), ("fast", "SPOC")]:
        result = download_lightkurve("TIC 1", cache_dir=tmp_path, cadence=cadence, author=author)
        assert result.meta["cadence"] == cadence
        assert result.meta["author"] == author
    assert len(calls) == 3
    assert cache_path(tmp_path, "TIC 1", cadence="fast") != cache_path(tmp_path, "TIC 1", cadence="long")
    assert cache_path(tmp_path, "A/B") != cache_path(tmp_path, "A?B")


def test_save_returns_the_file_that_was_written(tmp_path: Path) -> None:
    """A suffixless NPZ path is not silently rewritten by NumPy."""
    curve = generate_lightcurve(seed=9, n_points=100)
    path = tmp_path / "curve"
    saved = save_lightcurve(curve, path)
    assert saved == path
    assert path.is_file()
    np.testing.assert_array_equal(load_lightcurve(saved).flux, curve.flux)


def test_with_flux_rejects_wrong_length() -> None:
    """Flux replacement cannot silently corrupt the light-curve shape."""
    import pytest
    curve = generate_lightcurve(seed=9, n_points=100)
    with pytest.raises(ValueError, match="length"):
        curve.with_flux(np.ones(99))
