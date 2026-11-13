# Auditing a public benchmark for WWTP energy prediction (Melbourne Eastern Treatment Plant)

Code, data and results for the manuscript

> **Auditing a public benchmark for wastewater treatment plant energy prediction: misdated flows, an incomplete target and validation leakage in the Melbourne Eastern Treatment Plant dataset**
> Viet Nam Tran, Faculty of Environment, Van Lang University, Ho Chi Minh City, Vietnam.
> Draft v3, 13 November 2026. Not peer reviewed.

The study compares the public benchmark file `Data-Melbourne_F.csv` (Mendeley Data, doi:10.17632/pprkvz3vbd.1) column by column with the primary Melbourne Water open-data tables, rebuilds a date-consistent daily table, and repeats the energy-prediction task with naive baselines, time-ordered validation and conformal prediction intervals.

Vietnamese summary: [`README_VI.md`](README_VI.md).

## Reproduce everything

```bash
pip install -r code/requirements.txt     # numpy, pandas, scipy, scikit-learn, matplotlib
cd code
python run_all.py                        # about 7 minutes on 2 CPU cores
```

`run_all.py` runs steps 01–11 in order. Step 01 first checks the SHA-256 of every raw file and stops if one differs. All tables in `results/tables/` and all figures in `results/figures/` are regenerated. Random seeds are fixed (`SEED` in `code/common.py`); results were produced with Python 3.13.18 and the versions pinned in `code/requirements.txt` (see `results/environment_versions.txt`). Small numerical differences are possible with other library versions.

No network access is needed: the raw data are in `data/raw/`. `code/00_download_data.py` is an optional script that re-downloads the raw files from their sources and compares checksums; it was written for reviewers and **has not been executed in the authoring environment**, which had no direct network access (see `data/README.md` for how the files were obtained).

## Where each result in the manuscript comes from

| Manuscript item | Script | Output |
|---|---|---|
| Table 1, Figs 1–3, Fig. S1 (provenance audit, exact flow reconstruction) | `01_build_dataset.py`, `02_audit.py` | `T0`–`T6`, `T2b`, `Fig1`–`Fig3`, `FigS1` |
| Table 2, Fig. 4a,d (chronological hold-out, five model families, baselines, DM tests, bootstrap intervals, Holm) | `03_models.py`, `08_summary_figures.py` | `T7`, `T8`, `T8b`, `T15`, `Fig4` |
| Table 3 (same-day explanation vs day-ahead forecast, complete calendar) | `09_explain_vs_forecast.py` | `T16` |
| Table 4, Fig. 4b (validation protocols) | `03_models.py` | `T9` |
| Table 5, Fig. 4c (augmentation before/after splitting) | `06_augmentation_leakage.py` | `T13` |
| Table 6, Fig. 5 (prediction intervals) | `04_uncertainty.py` | `T10`, `Fig5` |
| Section 3.7, Fig. 6 (drivers, flow baseline, specific electricity) | `05_interpret.py` | `T11`, `T11b`, `T12`, `Fig6` |
| Section 3.7 (is grid + generation the plant's demand?) | `10_generation_check.py` | `T17` |
| Section 3.8 (out-of-time check, 2019–2021) | `07_out_of_time.py` | `T14`, `T14b` |
| Supplementary Material | `11_make_supplement.py` | `manuscript/Supplementary_Material.md` |

Model families (`code/features.py`): ridge, random forest, histogram gradient boosting, classical gradient boosting (GBM) and a deep feed-forward network (MLP, 3 hidden layers). Feature sets: `BM` (benchmark file as distributed), `BM_flowfix`, `RC` (reconstructed), `RC_noflow`, `LAG_only`, `RC_dyn`, `BM_rowlag`, and the sensitivity sets `RC_unscreened`, `RC_dyn_unscreened`, `RC_qalign`, `RC_dyn_qalign`.

## Repository layout

```
code/               analysis scripts (01-08), shared modules, run_all.py, requirements.txt
data/raw/           six raw files + SHA256SUMS.txt        (see data/README.md for sources and licences)
data/processed/     reconstructed daily table, model table, hold-out predictions
results/tables/     result tables T0-T15 (CSV)
results/figures/    figures (PNG 300 dpi and PDF)
manuscript/         manuscript draft (Word and Markdown)
```

## Main findings

1. The benchmark target `total_grid` is grid import only; on-site generation (29.9 % of electricity supply) is not in the file.
2. The two flow columns are daily means from 31 Dec 2008 – Oct 2012 attached by row position to dates in 2014–2019. One rule (mean of the hourly records of the i-th UTC day, omitting the 00:00 UTC record) reproduces 1,382 of 1,382 inflow values and 1,381 of 1,382 outflow values.
3. 90 % of total-nitrogen and 30 % of BOD values are fills (all filled BOD = COD/2).
4. With the file as distributed, none of five model families beats a previous-day persistence forecast on the chronological hold-out (RMSE 43.5–44.6 vs 40.7 MWh/d).
5. Re-dating the two flow columns alone gives 27.8 MWh/d. With same-day flows and lags (explanatory task): 17.8–19.0 MWh/d for total supply. With information available the day before only (day-ahead forecast): 31–32 MWh/d, 15–17 % better than persistence.
6. A two-parameter line in effluent flow matches gradient boosting on 16 inputs.
7. Ten-fold spline augmentation before splitting gives a test MAPE of 1.35 %; copying the previous row gives 1.29 %; with the correct order, 11.6 %.

## Status and caveats

- Draft, not peer reviewed. The authors of the benchmark and of the studies discussed have not yet been consulted.
- Single plant; weather fields were screened but not compared with a primary meteorological record.
- The deep model is a feed-forward network (scikit-learn MLP); recurrent networks, XGBoost/LightGBM and SHAP were not available in the authoring environment.
- Forward-citation screening used OpenAlex (titles only), not Scopus or Web of Science.
- Generative AI (Claude, Anthropic) was used for literature search, coding and drafting; see the declaration in the manuscript.

Repository: https://github.com/phamnam123lite-svg/etp-energy-benchmark-audit

## Licence and citation

Code: MIT (`LICENSE`). Data: see `data/README.md` (CC BY 4.0 sources, redistributed with attribution). Please cite the manuscript and the original data sources; see `CITATION.cff`.
