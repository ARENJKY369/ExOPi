# Real-data ingestion check (not transit-recovery validation)

Source: Lightkurve's 100-cadence TESS fixture, pinned to commit
`f762d52b09f0e928b691e4a5700dcb849fcbcb27`:
https://github.com/lightkurve/lightkurve/blob/f762d52b09f0e928b691e4a5700dcb849fcbcb27/tests/data/test-lc-tess-pimen-100-cadences.fits

SHA-256: `7577c14d6eef06d5c09c0f3d9521e8e756eea584bcc91ac407fd3d090e5979a1`.
The loader retained 99 of 100 samples, with a 0.1361-day baseline. The FITS
header identifies TIC 261136679. This is a truncated fixture rather than a
full target light curve. Its raw 0.1375-day baseline is insufficient to validate
recovery of an independently sourced planetary period; no real-transit
recovery or real-data false-alarm claim is made. The file is not bundled.

```sh
gh api 'repos/lightkurve/lightkurve/contents/tests/data/test-lc-tess-pimen-100-cadences.fits?ref=f762d52b09f0e928b691e4a5700dcb849fcbcb27' --jq .content | base64 -d > pimen.fits
sha256sum pimen.fits
python -c 'from shapehunter.validation import load_tess_fits; c=load_tess_fits("pimen.fits"); print(c.meta, len(c), c.baseline)'
```

For a full locally supplied TESS PDCSAP FITS light curve and independently
sourced reference period, use `load_tess_fits(path)` followed by
`validate_reference(curve, expected_period, reference_source)`. The latter
rejects baselines shorter than two reference periods. Obtain the reference
period and file independently and evaluate a representative collection of
planet and false-positive targets before drawing conclusions. Tests remain
fully offline and use synthetic FITS files.

## Kepler-1976 b: real period recovery

Three separate public Kepler PDCSAP quarters for KIC 11138155 are mirrored in
[NarrSf/KeplerTransitAnalysis at commit fea4040](https://github.com/NarrSf/KeplerTransitAnalysis/tree/fea40407691a78c88c593e97a4f17091671ff401/kepler1976b_fits).
The independently retrieved reference value, **4.959319244 days**, is read
from [this catalog CSV at commit 442f73f](https://github.com/berkeley-stat238/spring-2026/blob/442f73f93c5d200edd3746d2b27a6fb723725042/exoplanet_data.csv).
The CSV contains the period but does not itself document its original upstream
catalog provenance. This comparison is independent of the light-curve source,
but the star was selected because it was known to host this planet.

Using quality-zero samples and the pipeline defaults, the recovered periods
were 4.95992504, 4.95929699, and 4.95934222 days, respectively; all passed the
1% period criterion. File hashes, source, sample counts and full output are in
`real-kepler1976-results.json`. The observations are three quarters of **one
star**, not three independent targets, and the evaluation did not validate
false positives or shape classification. The source FITS files are not bundled.
The target-specific `scripts/validate_kepler1976.py` reproduces the analysis
when supplied the three files and the catalog CSV; the script checks the target
header and hashes all inputs.

## Kepler-10 b: second source-labelled target

A time/flux CSV from [MohammedHamadi/Kepler-10-flux-modeling at commit e3cb83b](https://github.com/MohammedHamadi/Kepler-10-flux-modeling/blob/e3cb83bf31a783bd459522eb13f9dcc4e5030e44/kepler-10_flux.csv)
is labelled Kepler-10 by its source. Against the independent reference CSV
listed above, the pipeline returned 0.8377992 days versus the catalog value of
0.8374907 days (PASS, within 1%). The CSV does not include mission quality
flags, flux errors, or a target identifier, so identity and data provenance
cannot be independently established from the file itself; uncertainties were
estimated from the flux. `real-kepler10-results.json` contains hashes and the
full result. Both targets were selected because they are known planets; these
examples do not measure false-positive performance or population completeness.

## Exploratory Kepler labelled-flux proxy

We also evaluated all five label-2 rows and 30 randomly sampled label-1 rows
(seed 422) from a mirror of the curated Kepler labelled time-series **test**
file. The gzip file was obtained from [rvarun95/Clusters at commit b920a5e](https://github.com/rvarun95/Clusters/blob/b920a5eaee74cc16634aa8a19c8fcd401610ca39/predict_exoplanet/webapp/web/exoTest.csv.gz),
with compressed SHA-256 `f56e24ec1d05c896cdef308a7b927f2269426c2855a583b2273208676b9b9ecf`
and decompressed SHA-256 `0b9f9aabcaf4058fd38d99431e61e29eeb031d5073c3b74f6139a917356bd163`.
The source describes the dataset as cleaned Kepler observations with labels 2
and 1 corresponding to planet-host and non-planet-host examples respectively;
see [the dataset description](https://github.com/anabr4/Exoplanet-Star-Classification#the-data-star-light-intensities-vs-time).

**Critical limitations:** the table does not provide target identities, real
sample timestamps, quality flags, physical flux normalization, uncertainties,
or ephemerides. We assumed uniform 0.02043365-day sampling, rescaled centred
flux arbitrarily, and estimated uncertainties from consecutive differences.
Any-period detections cannot be verified against reference periods, the label-1
sample is not a vetted set of guaranteed negatives, and rows may not be
independent stars. The curated sample is not proven representative of a survey.
This is a stress test only—not a population-level calibration or an absolute
false-alarm rate.

The search emitted candidates for 5/5 label-2 and 18/30 sampled label-1 rows
at its default threshold, irrespective of vetting status. These results warn
that the default threshold is unsuitable to assume reliable on this proxy.
The sample is tiny and its confidence intervals are wide. Full per-row output,
including failures and assumed parameters, is in `kepler-labelled-proxy.json`;
`scripts/benchmark_kepler_labels.py` reproduces the selection and calculation
from a locally supplied copy. This experiment also found BLS fits with transit
duration at least half a period; the pipeline now rejects these pathological
fits instead of crashing the template comparison.

## Disjoint threshold calibration on the curated proxy

A second experiment (`kepler-proxy-calibration.json`) used 40 label-1 rows to
choose an empirical threshold and 30 *different* label-1 rows for evaluation.
All 30 negative rows used in the earlier proxy experiment were excluded from
both sets. The five label-2 rows were evaluated separately but had already
been inspected in the earlier experiment, so they are **not blind holdouts**.
The fixed target was 10% of label-1 rows, with seed 912 and the same assumed
cadence and arbitrary flux scaling described above. The selected threshold was
18.11229; 5/30 held-out label-1 rows and 3/5 previously inspected label-2 rows
passed the BLS screening rule. The 95% binomial interval for the label-1 rate
is approximately 5.6%–34.7%; the five-row label-2 interval is approximately
14.7%–94.7%. There were no processing errors in this run.

These are *screening detections at any period*, not vetted planet recoveries.
No labels of true period are available. The assumed sampling, unknown physical
flux normalization, catalogue labels, and sample selection prevent this from
being a representative real-survey calibration; the threshold has **not**
been installed as the pipeline default. The report records every row ID,
training score, evaluation score, and dataset hash. Reproduce it with
`scripts/calibrate_kepler_proxy.py` and the original exclusion report.

The real-star results above and the earlier shape-grid benchmark were generated
before the uncertainty-propagation fix in preprocessing. They remain recorded
as historical results and should be rerun before using their exact numbers
with the corrected pipeline.
