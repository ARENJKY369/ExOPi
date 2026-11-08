# Methods and assumptions

1. Seeded synthetic data or a cached/saved light curve is cleaned with
   upward-only flare clipping, isolated outlier handling, and a local trend.
2. Astropy BLS supplies a periodic ephemeris; a trapezoid fit refines it. The
   capped grid and noise assumptions limit sensitivity and significance.
3. Raycast family templates are compared at the BLS period while fitting each
   template’s depth, timing offset, and duration within fixed bounds. A BIC
   difference compares these fits; discrete template search and possible
   ephemeris errors complicate its interpretation.
4. Vetting reports checks separately. Missing inputs are unknown, not passes;
   overall PASS means no implemented rejection criterion was triggered.
5. Synthetic validation reports each seed and the empirical recovery or
   false-alarm fraction. It does not provide uncertainty intervals or establish
   performance on real stellar noise.

The velocity-Verlet integrator assumes Newtonian point masses and a fixed
step. TTV extraction requires sufficient observations around each predicted
transit and fits a fixed-shape template. Choose units consistently for the
integrator and verify numerical convergence for each application.

## Calibration and classification experiments

`shapehunter.validation.calibrate_threshold` uses a noise-only training set to
select an empirical BLS screening threshold. `evaluate_holdout` applies that
frozen threshold to disjoint noise and planet-injection seeds, reporting trial
outcomes and exact binomial intervals. At least 19 independent training seeds
are needed to resolve a 5% target; considerably more are needed for useful
precision. These rates describe the synthetic generator, not real data.

`shapehunter.validation.shape_confusion` records a per-family confusion matrix
and missed detections for specified seeds and injection parameters. Repeat over
period, size, and noise grids before interpreting it. A tiny reproducible smoke
run is saved in `calibration-pilot.json`; its intentionally high target and two
holdout trials are not a production calibration. The normal CLI continues to
use its fixed default threshold; these experiments do not silently change it.

A larger (still small) example is saved as `calibration-benchmark.json`: 19
training seeds, 10 noise holdouts, 10 injection holdouts, and two seeds per
shape. Its 10% target is intentionally illustrative, not a recommended survey
threshold. Confidence intervals are wide, and the shape classifications show
substantial confusion. The output records every seed for reproducibility.

Use `--min-power VALUE` to apply a chosen threshold and `--max-candidates N` to
opt into iterative search. Each detection is masked before the next search;
results are conditional on earlier masks, which can hide overlapping events or
introduce aliases. The default remains one candidate. Candidate JSON includes
`meta.family_chi2` for the competing fixed-ephemeris template fits.


The updated shape-specific timing fit was evaluated on the same two seeds per
family as the original shape benchmark; results are in
`shape-benchmark-optimized.json`. Ten of twelve injected examples matched their
family; both oblate examples were reported as spheres. This narrow experiment
does not demonstrate reliable classification over different parameters,
impact parameters, or real observations.

For local real-data evaluation, `load_tess_fits` reads quality-zero TESS
PDCSAP data and `validate_reference` compares the search with a user-supplied,
independently sourced period. See `real-data-check.md` for the checked fixture
and why it cannot validate transit recovery.

A wider shape grid (`shape-benchmark-grid.json`) evaluates 48 synthetic trials:
6 default families, two radii, two impact parameters and two noise seeds.
Thirty-eight of 48 predictions matched their injected family. All eight oblate
injections were classified as spheres; six of eight ringed injections matched.
This illustrates substantial degeneracy, not reliable general classification.
The candidate metadata flags close chi-square competitors as `ambiguous`;
other labels remain `provisional`, not confirmed. The gap threshold is a
heuristic and has not been calibrated on real observations.

`oblate-stress-test.json` records 16 additional sphere/oblate synthetic cases
with two seeds, two impact parameters and flattenings 0, 0.25 and 0.5. All
16 were flagged ambiguous in this experiment. An oblate body with zero
flattening is mathematically identical to a sphere, which an automated
classifier cannot distinguish from photometry alone. The reported
`plausible_shapes` are families within a heuristic chi-square gap of 10 of the
best fit, not confidence sets or calibrated probabilities. Higher-quality data,
broader physical models, and independent calibration are required before
claiming oblateness; a single selected family should not be treated as proof.

## Target-level population evaluation

`scripts/validate_population.py` evaluates a local JSON manifest containing one
row per target. It records input hashes, sources, exclusions, period recovery,
PASS-vetted recovery, and false alarms on explicitly labelled negative controls,
with exact binomial intervals. Duplicate target IDs are rejected; multiple
quarters of one star must not be counted as independent targets. Targets with
insufficient baseline or processing errors are listed separately rather than
silently counted as successes or failures. Negative labels must come from an
independent control set, not merely from an absent catalog entry.

A manifest is a JSON list with `target_id`, `path`, `format` (`kepler_fits`,
`tess_fits` or `flux_csv`), `data_source`, `reference_source`, `label`
(`positive` or `negative`), and for positives `expected_period` in days.
Relative paths are resolved next to the manifest. FITS rows may additionally
supply `expected_header_target` to check the target header. Run:

```sh
python -m pip install -e .
python scripts/validate_population.py manifest.json --out population-results.json
```

`real-population-pilot.json` demonstrates the tool using the two previously
identified positive stars. There are no negative controls: its false-alarm rate
is therefore null, not zero. Two hand-selected hosts cannot establish
population-level performance. The Kepler-10 CSV lacks verifiable target metadata,
quality flags and per-point uncertainties; see `real-data-check.md`.

The output field `oblate_sphere_decision` reports `inconclusive` when both
families fall within the heuristic score gap. This is not a calibrated
confidence interval or a definitive physical diagnosis. A zero-flattening
ellipse exactly equals a sphere, so a definitive discrimination is impossible
in that case; all other physical claims need independently calibrated models,
selection-aware real observations, and follow-up.

The curated Kepler labelled-flux stress test in `kepler-labelled-proxy.json`
found many detections in its label-1 sample under an assumed cadence and
arbitrary flux scaling. It lacks source timestamps and reference periods and
is not a representative survey validation; see `real-data-check.md`. Do not
use this result or the synthetic threshold as a calibrated real-data false
alarm rate.

A disjoint threshold-calibration experiment on the curated Kepler proxy is
recorded in `kepler-proxy-calibration.json`. Its previous negative rows were
excluded and training/holdout negatives are disjoint; its five positives were
previously inspected and are not blind. See `real-data-check.md` before using
its scores: the dataset does not establish a real-survey false-alarm rate.

### Calibrating on independently labelled observations

`calibrate_observed_population(training_controls, holdout_targets)` accepts
`ReferenceTarget` objects loaded from local real observations. The training
split must contain independently labelled negative controls; the holdout split
must contain both positive and negative labels. The function rejects repeated
IDs and identical time/flux arrays across splits, freezes a threshold from the
training controls, and reports holdout recovery, false alarms, exclusions and
exact binomial intervals. It does not change the pipeline default threshold.

The result's `numeric_false_alarm_goal_met` is true only when every holdout
completed and the upper 95% interval on the negative-control detection rate is
at most the requested rate. Even then it is only a numerical check for the
supplied sample: the code cannot establish representative selection, verify
labels, or detect different quarters from the same star with unrelated IDs.
A representative labelled real-data sample is still required; the two-target
pilot and the cadence-assumed proxy dataset do not meet that requirement.
