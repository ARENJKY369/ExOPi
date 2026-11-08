# shapehunter

A CPU-first, experimental transit search that combines a quadratic limb-darkened
pixel-grid raycaster, six occulting shape families, BLS, matched templates, and
basic vetting. It is a research prototype, **not** a planet-confirmation tool.

## Install and run

Python 3.10+ is recommended. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
python -m shapehunter.pipeline.run_star --synthetic --seed 7 --out results
python -m shapehunter.pipeline.run_star --synthetic --seed 7 --out noise --no-injection
python -m pytest -q
```

The search writes `candidates.json` and `report.txt`. Search a saved `.npz`
light curve with `--input FILE` instead of `--synthetic`. Batch processing is
available through `python -m shapehunter.pipeline.batch --out DIR FILE...`.
Optional `lightkurve` is imported only when downloading an uncached target;
optional `streamlit` is imported only when launching `streamlit run dashboard/app.py`.
Tests use synthetic data and cached files only, with no network access.

## Interactive website

Launch the public-ready dashboard locally:

```sh
python -m pip install streamlit
streamlit run dashboard/app.py
```

For a non-scientific dashboard image upload test, place a downloaded PNG/JPEG in
this repository and run:

```sh
python scripts/image_to_npz.py your-image.jpg --out image-test.npz
```

The generated NPZ samples image brightness across columns; it is UI test data,
not an astronomical measurement. Delete or clearly label it before any scientific use.

The dashboard supports synthetic searches, saved `.npz` light curves, saved
`candidates.json` findings, and downloadable findings. Uploads are limited to
20 MB, stay in server memory for the current session, and are not written by
the app. Set `SHAPEHUNTER_PASSWORD` to require a private-access password; for
user accounts, role controls, audit logs, or persistent findings, deploy with a
proper identity provider and backend database instead of using a shared password.
Use HTTPS, a host-managed secret store, and dependency security scanning in
production.

## Scientific boundaries

- BLS uses a capped frequency grid and an empirical threshold; the threshold is
  not a calibrated false-alarm probability. Iterative multi-candidate search is
  opt-in via `--max-candidates`; masking may hide overlapping transits or create aliases.
- BIC compares a finite template bank at a fixed period with bounded timing and duration fits;
  the preferred family can differ from the injected shape, especially for near-degenerate
  shapes. It is not a reliable
  physical classification or evidence of a non-spherical planet on its own.
  Close competitors are marked ambiguous; all other labels are provisional.
- Vetting is approximate. Missing catalogues, systematics masks, and centroids
  are marked unknown; a PASS can coexist with unknown checks. Starspot and
  asymmetry diagnostics are informational. Real-world follow-up is essential.
- Validation rates apply only to the specific seeded synthetic generator and
  sampled trials; small experiments do not establish survey performance. Three
  real Kepler quarters of one known target recover its period, but do not
  establish false-alarm rates or reliable real shape classification.
- Dynamics use fixed-step Newtonian point masses, with no collision physics or
  adaptive timesteps. Timing extraction uses a fixed trapezoid template.
- Batch scheduling is caller-driven and synchronous, not a persistent service.
- Light-curve cache keys include target, mission, author, and cadence. Older
  cache filenames are not automatically reused.

See [docs/methods.md](docs/methods.md) for the workflow and assumptions.
