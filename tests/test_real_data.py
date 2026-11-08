"""Test the local reference-data adapter using synthetic FITS files only."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from shapehunter.core import Candidate
from shapehunter.validation import load_tess_fits, validate_reference
from shapehunter.validation import real_data


def test_tess_loader_filters_normalizes_and_converts_time(tmp_path: Path) -> None:
    """Synthetic FITS quality and nonfinite rows are discarded consistently."""
    time = np.arange(15, dtype=float)
    flux = np.full(15, 1000.0)
    flux[3] = np.nan
    quality = np.zeros(15, dtype=np.int32)
    quality[5] = 1
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="PDCSAP_FLUX", format="D", array=flux),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.ones(15)),
        fits.Column(name="QUALITY", format="J", array=quality),
    ])
    table.header["BJDREFI"] = 2457000
    primary = fits.PrimaryHDU()
    primary.header["OBJECT"] = "synthetic"
    path = tmp_path / "synthetic.fits"
    fits.HDUList([primary, table]).writeto(path)
    curve = load_tess_fits(path)
    assert len(curve) == 13
    assert curve.meta["discarded_samples"] == 2
    assert curve.time[0] == 2457000
    np.testing.assert_allclose(curve.flux, 1)
    np.testing.assert_allclose(curve.flux_err, 0.001)


def test_reference_comparison_requires_baseline_and_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference matching is explicit and short baselines cannot claim recovery."""
    from shapehunter.data import generate_lightcurve

    curve = generate_lightcurve(seed=42)
    monkeypatch.setattr(real_data, "search_star", lambda curve, min_power: [Candidate(period=3.5, t0=2, duration=0.1, depth=0.01, snr=20)])
    assert validate_reference(curve, 3.5, "synthetic test label")["recovered"]
    with pytest.raises(ValueError, match="baseline"):
        validate_reference(curve, 100, "synthetic test label")
    with pytest.raises(ValueError, match="reference source"):
        validate_reference(curve, 3.5, "")


def test_kepler_loader_requires_and_filters_sap_quality(tmp_path: Path) -> None:
    """Synthetic Kepler FITS uses SAP_QUALITY, not the TESS quality column."""
    from shapehunter.validation import load_kepler_fits

    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.arange(12, dtype=float)),
        fits.Column(name="PDCSAP_FLUX", format="D", array=np.ones(12) * 100),
        fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=np.ones(12)),
        fits.Column(name="SAP_QUALITY", format="J", array=np.array([1] + [0] * 11)),
    ])
    table.header["BJDREFI"] = 2454833
    path = tmp_path / "kepler.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    curve = load_kepler_fits(path)
    assert len(curve) == 11
    assert curve.time[0] == 2454834
    assert curve.meta["source"] == "Kepler FITS"
    with pytest.raises(ValueError, match="quality"):
        load_tess_fits(path)


def test_two_column_csv_has_explicit_unknown_quality(tmp_path: Path) -> None:
    """Synthetic CSV normalizes data without inventing errors or quality flags."""
    from shapehunter.validation import load_flux_csv

    path = tmp_path / "flux.csv"
    path.write_text("time,flux\n" + "\n".join(f"{i},{100 if i != 3 else 'nan'}" for i in range(12)))
    curve = load_flux_csv(path, "synthetic")
    assert len(curve) == 11
    assert curve.flux_err is None
    assert curve.meta["quality_flags"] == "unavailable"
    assert curve.meta["uncertainties"] == "estimated"
    np.testing.assert_allclose(curve.flux, 1)
