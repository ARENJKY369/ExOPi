"""Light-curve persistence and a cached, lazily imported lightkurve loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from shapehunter.core import LightCurve

__all__ = ["save_lightcurve", "load_lightcurve", "download_lightkurve", "cache_path"]


def _arrays(lc: LightCurve) -> Dict[str, np.ndarray]:
    data = {"time": lc.time, "flux": lc.flux}
    if lc.flux_err is not None:
        data["flux_err"] = lc.flux_err
    if lc.centroid_x is not None:
        data["centroid_x"] = lc.centroid_x
    if lc.centroid_y is not None:
        data["centroid_y"] = lc.centroid_y
    return data


def save_lightcurve(lc: LightCurve, path: str | Path) -> Path:
    """Write a light curve to ``.npz`` (or ``.h5``/``.hdf5`` via h5py)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(lc)
    if target.suffix.lower() in {".h5", ".hdf5"}:
        import h5py  # lazy optional dependency

        with h5py.File(target, "w") as handle:
            for name, values in arrays.items():
                handle.create_dataset(name, data=np.asarray(values))
            handle.attrs["meta"] = json.dumps(lc.meta, default=str)
    else:
        payload = {f"arr_{k}": v for k, v in arrays.items()}
        payload["meta_json"] = np.array(json.dumps(lc.meta, default=str))
        # Passing an open file avoids NumPy silently appending .npz.
        with target.open("wb") as handle:
            np.savez_compressed(handle, **payload)
    return target


def load_lightcurve(path: str | Path) -> LightCurve:
    """Read a light curve written by :func:`save_lightcurve`."""
    source = Path(path)
    if source.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(source, "r") as handle:
            arrays = {name: np.asarray(handle[name]) for name in handle.keys()}
            meta = json.loads(handle.attrs.get("meta", "{}"))
    else:
        with np.load(source, allow_pickle=False) as data:
            arrays = {
                key[4:]: np.asarray(data[key])
                for key in data.files
                if key.startswith("arr_")
            }
            meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
    return LightCurve(
        time=arrays["time"],
        flux=arrays["flux"],
        flux_err=arrays.get("flux_err"),
        centroid_x=arrays.get("centroid_x"),
        centroid_y=arrays.get("centroid_y"),
        meta=meta,
    )


def cache_path(cache_dir: str | Path, target: str, mission: str = "TESS", author: Optional[str] = None, cadence: str = "fast") -> Path:
    """Cache key includes the full request to prevent accidental collisions."""
    request = json.dumps([str(target), str(mission), author, str(cadence)], ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(request.encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir) / f"lightcurve_{digest}.npz"


def download_lightkurve(
    target: str,
    mission: str = "TESS",
    cache_dir: str | Path = "data_cache",
    author: Optional[str] = None,
    cadence: str = "fast",
) -> LightCurve:
    """Download and normalise a light curve through lightkurve (lazy import).

    Results are cached on disk as ``.npz``; a cached file is returned without
    touching the network. ``lightkurve`` is imported only inside this function,
    so importing :mod:`shapehunter` never requires it.
    """
    cached = cache_path(cache_dir, target, mission, author, cadence)
    if cached.exists():
        return load_lightcurve(cached)

    try:  # pragma: no cover - requires the optional dependency + network
        import lightkurve as lk
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "download_lightkurve needs the optional dependency lightkurve. "
            "Install it with `pip install lightkurve`, or use the synthetic "
            "generators in shapehunter.data.synthetic."
        ) from exc

    search = lk.search_lightcurve(target, mission=mission, author=author, cadence=cadence)
    if len(search) == 0:
        raise ValueError(f"lightkurve found no light curves for {target!r} ({mission})")
    collection = search.download_all()
    if collection is None:  # pragma: no cover - network dependent
        raise RuntimeError(f"download failed for {target!r}")
    lc = collection.stitch()
    lc = lc.remove_nans().normalize()
    fluxes = np.asarray(lc.flux.value, dtype=float)
    errors = np.asarray(lc.flux_err.value, dtype=float)
    times = np.asarray(lc.time.value, dtype=float)
    out = LightCurve(
        time=times,
        flux=fluxes,
        flux_err=errors,
        meta={"target": str(target), "mission": mission, "author": author, "cadence": cadence, "source": "lightkurve"},
    )
    save_lightcurve(out, cached)
    return out


def lightcurve_to_dict(lc: LightCurve) -> Dict[str, Any]:
    """Small summary dict (used by reports and the dashboard)."""
    return {
        "n_points": len(lc),
        "baseline": lc.baseline,
        "cadence": lc.cadence,
        "median_flux": float(np.median(lc.flux)) if len(lc) else 0.0,
        "meta": dict(lc.meta),
    }
