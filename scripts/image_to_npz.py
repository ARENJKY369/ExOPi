"""Convert an image into a synthetic brightness-array NPZ for UI testing.

This utility does not recover scientific light-curve measurements. It samples
mean RGB brightness across image columns and creates synthetic time/flux arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_to_npz(image_path: Path, output_path: Path, samples: int = 2400) -> Path:
    if samples < 30:
        raise ValueError("samples must be at least 30")
    image = Image.open(image_path).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    # Sample every image column from the whole image, then normalize brightness.
    columns = np.linspace(0, pixels.shape[1] - 1, samples).round().astype(int)
    brightness = pixels[:, columns].mean(axis=(0, 2))
    flux = brightness / max(float(np.median(brightness)), 1e-12)

    time = np.linspace(0.0, 50.0, samples, dtype=np.float64)
    noise = np.median(np.abs(flux - np.median(flux)))
    flux_err = np.full(samples, max(float(noise), 1e-8), dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, time=time, flux=flux, flux_err=flux_err)

    metadata = {
        "target": image_path.stem,
        "source": "image_brightness_test",
        "source_type": "not_astronomical_data",
        "image_width": int(image.width),
        "image_height": int(image.height),
        "warning": "Brightness sampling is a UI test, not a measured light curve",
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG, JPEG, or WebP image to sample")
    parser.add_argument("--out", type=Path, default=Path("image-test.npz"))
    parser.add_argument("--samples", type=int, default=2400)
    args = parser.parse_args()
    try:
        output = image_to_npz(args.image, args.out, args.samples)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Created {output} (UI test data only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
