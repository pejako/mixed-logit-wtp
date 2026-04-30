# mixed-logit-choice-lab

A reproducible Mixed Logit (random-coefficient) discrete choice modeling toolkit
in **Python and R**, built around synthetic data with known ground truth.

The goal is not to ship yet another logit library — it is to demonstrate, end to
end, how a Mixed Logit (MXL) model is specified, estimated, and interpreted,
and to *prove* the estimator is correct by recovering the parameters that
generated the data.

## Why this exists

Plain multinomial logit (MNL) models are everywhere, but they impose
Independence of Irrelevant Alternatives (IIA) and assume every decision-maker
shares the same preferences. Both assumptions break in real choice data.
Mixed Logit relaxes them by letting coefficients vary across individuals,
which produces realistic substitution patterns and lets you estimate
**willingness to pay (WTP)** as a distribution rather than a single number.

This repo walks through that machinery on synthetic data so the math is
unambiguous: you set the true parameters, generate choices, estimate, and
check that you got the truth back.

## What's inside

- **Synthetic DGP** — a configurable data generator for choice experiments
  with random coefficients (normal, log-normal, fixed) on price and attributes.
- **MXL estimator** — Simulated Maximum Likelihood with Halton draws,
  implemented from scratch in NumPy and benchmarked against `xlogit`.
- **Elasticity module** — own- and cross-price elasticities at the individual
  and aggregate level, with side-by-side comparison to MNL to show how IIA
  fails.
- **WTP module** — marginal WTP distributions for each non-price attribute,
  with both preference-space and WTP-space parameterizations.
- **Parameter-recovery tests** — every push to CI re-simulates data and
  verifies estimates land within 2 SE of the true betas.
- **R parity layer** — the same model, same data, same outputs via `mlogit`,
  to demonstrate cross-language reproducibility.

## Quickstart (Python)

```bash
cd python
pip install -e ".[dev]"
python -c "from mixedlogit.dgp import simulate_choices; print(simulate_choices(seed=42).head())"
```

Run the full test suite:

```bash
python -m pytest tests/ -m "not slow"     # quick (~30s)
python -m pytest tests/                    # includes parameter-recovery tests (~2m)
```

See `python/notebooks/` for the walkthrough:

1. `01_dgp_and_mnl_baseline.ipynb` — generate data, fit plain MNL, see the attenuation bias
2. `02_mixed_logit_estimation.ipynb` — fit MXL, recover the truth (means *and* SDs)
3. `03_elasticities_and_substitution.ipynb` — MNL vs MXL substitution patterns and the IIA failure
4. `04_wtp_and_feature_preference.ipynb` — WTP distributions and feature ranking

Each notebook is built from a `_build_*.py` script in the same directory, so
edits go to the script and the `.ipynb` is regenerated and re-executed in CI.

## Quickstart (R)

```r
devtools::load_all("r/")
vignette("mxl-walkthrough")
```

## Quickstart (R parity layer)

```bash
# Generate the shared dataset and Python results (one-time)
cd python
python -m mixedlogit.export_csv
python scripts/export_results.py
```

```r
# In R, from r/
install.packages(c("mlogit", "dfidx", "jsonlite", "testthat"))
testthat::test_dir("tests/testthat")
```

The R parity tests assert that R's `mlogit` recovers the same MNL and
MXL estimates the Python side does, on the same canonical dataset, to
within simulation tolerance. See [`r/README.md`](r/README.md).

## Methodology

A short, math-first writeup of the model, the SML estimator, the
elasticity derivation, and the WTP-space discussion lives in
[`docs/methodology.md`](docs/methodology.md). About 15 minutes to read
end to end; the appendix at the bottom defends a few non-textbook design
choices.

## Status

**v0.1 complete.** Python side: estimator, elasticity, WTP, four executable
notebooks, methodology writeup, CI workflow with parameter-recovery tests
on a weekly schedule, 79/79 tests passing. R side: parity layer using
`mlogit`, with cross-language coefficient comparison tests in CI.

**v0.2 planned:** analytical gradient in the SML estimator, WTP-space
parameterization (Train & Weeks 2005), and a Shiny/Streamlit dashboard
for interactive what-if pricing scenarios.

## License

MIT.
