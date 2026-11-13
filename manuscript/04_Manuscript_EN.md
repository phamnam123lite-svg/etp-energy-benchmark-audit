---
title: "Auditing a public benchmark for wastewater treatment plant energy prediction: misdated flows, an incomplete target and validation leakage in the Melbourne Eastern Treatment Plant dataset"
author: "Viet Nam Tran — Faculty of Environment, Van Lang University, Ho Chi Minh City, Vietnam. Corresponding author: nampham123ntt@gmail.com"
date: "Draft v3, 13 November 2026 — not peer reviewed"
---

# Highlights

- A public WWTP energy benchmark was checked against its primary source tables.
- Its flow columns hold 2008–2012 values on 2014–2019 dates; all rows reproduced.
- The target is grid import only; 30 % of electricity supply is generated on site.
- With the file as distributed, five model families do not beat persistence.
- Same-day effluent pumping explains demand; day-ahead skill is only 15–17 %.

# Abstract

Open datasets are scarce in wastewater engineering, so the few that exist become benchmarks. One, a daily table for Melbourne's Eastern Treatment Plant (1,382 rows, 2014–2019), has been used in at least six machine-learning energy-prediction studies. We compared this file column by column with its primary open-data tables, then repeated the prediction task on a reconstructed table. Four defects were found. (1) The target is grid import only; on-site generation, 30 % of electricity supply, is absent. (2) Both flow columns are daily means for 2008–2012 attached by row position to dates in 2014–2019; a simple rule reproduces all 1,382 inflow values exactly. (3) 90 % of total-nitrogen and 30 % of BOD values are fills, not measurements. (4) Each date label is one day earlier than the day its energy value describes. With the file as distributed, none of five model families beat a previous-day persistence forecast on a chronological hold-out (RMSE 43.5–44.6 against 40.7 MWh d⁻¹), yet shuffled cross-validation returned R² = 0.31. Re-dating the two flow columns alone lowered RMSE to 27.8 MWh d⁻¹. Same-day effluent flow, pumped through a 56-km outfall, explained most of the variation (RMSE 17.8–19.0 MWh d⁻¹ for total supply), but it is not known in advance: day-ahead forecasts reached only 31–32 MWh d⁻¹, 15–17 % better than persistence. Spline augmentation before splitting reproduced a published 1.4 % MAPE that copying the previous row also attains. A Gaussian residual band covered 78–86 % of hold-out days at a nominal 90 %.

**Keywords:** wastewater treatment plant; energy consumption; benchmark dataset; data provenance; data leakage; conformal prediction; effluent pumping

# 1. Introduction

Electricity is one of the largest operating costs of a wastewater treatment plant (WWTP), and its monitoring, benchmarking and reduction have been studied for decades (Longo et al., 2016; Gu et al., 2017; Cardoso et al., 2021). Data-driven models are now routinely proposed for predicting plant electricity use from flow, load and weather (Torregrossa et al., 2018; Harrou et al., 2021; Zhang et al., 2021; Oliveira et al., 2021), within a wider movement toward machine learning (ML) in wastewater operation (Newhart et al., 2019; Bahramian et al., 2023).

Full-scale operating data are rarely published, so the few open tables attract repeated use. A prominent example is the daily dataset for Melbourne Water's Eastern Treatment Plant (ETP) deposited on Mendeley Data (Bagherzadeh, 2021) and introduced by Bagherzadeh et al. (2021). It joins grid electricity, influent and effluent flow, influent quality and airport weather for 1,382 days between 2014 and 2019. The same file has since served as the test bed for comparisons of 15 to 23 ML models (Alali et al., 2022, 2023), deep learning with data augmentation (Harrou et al., 2023), a knowledge-enhanced graph network (Allen and Cordiner, 2025) and an interval type-2 neuro-fuzzy system (Khaled et al., 2026). Each study proposed a new model and reported lower errors than its predecessors.

Three issues have not been examined in this line of work. First, none of the studies compared the file with the primary Melbourne Water tables from which it was assembled, so the meaning and integrity of its columns rest on a single join performed in 2021. Second, none reported a naive reference such as the previous day's value, although such references are the standard yardstick for forecast skill (Hyndman and Koehler, 2006). Third, reported errors span almost one order of magnitude, from a mean absolute percentage error (MAPE) of 10–15 % (Alali et al., 2023; Khaled et al., 2026) to 1.4 % (Harrou et al., 2023), without an explanation of the difference. General reviews have warned that data leakage and inadequate splitting are widespread in environmental ML (Zhu et al., 2023) and across ML-based science (Kapoor and Narayanan, 2023), and that data handling is the weak link of the wastewater "data pipeline" (Therrien et al., 2020). Temporal data in particular require validation schemes that respect time order (Bergmeir and Benítez, 2012; Roberts et al., 2017). Recent studies have begun to enforce temporally admissible inputs and leakage-aware pipelines for wastewater forecasting at other plants and for other targets (Mateos Bravo et al., 2026; Paepae, 2026). A screening of the 232 works that cite the benchmark or its three most-cited users found none that compares the file with its sources (Supplementary Note S1); to our knowledge, no study has audited a specific wastewater benchmark in this way and quantified what the defects do to model performance.

This paper does so for the ETP file. The objectives were (i) to verify every column of the benchmark against the primary open-data tables; (ii) to rebuild a date-consistent table from those sources, including on-site generation; (iii) to quantify how the benchmark's defects and the validation protocol change apparent model skill relative to naive baselines; and (iv) to give calibrated prediction intervals and an engineering interpretation of what drives daily electricity use at this plant. Four hypotheses, formulated after the audit and before the prediction experiments, were tested. H1: with the file as distributed, ML models do not outperform a previous-day persistence forecast under chronological validation. H2: correcting the data, without changing the algorithm, yields a larger gain than any algorithm change. H3: total electricity supply (grid plus on-site generation) is more predictable from process variables than grid import. H4: shuffled validation and augmentation before splitting inflate apparent accuracy.

# 2. Materials and methods

## 2.1 Plant

ETP, at Bangholme in south-east Melbourne, treats about 330 ML d⁻¹ (Bagherzadeh et al., 2021). Treated effluent leaves the site through an outfall pump station and a 56-km pipeline to Bass Strait (Melbourne Water, 2023). Biogas from anaerobic digestion fuels an on-site power station that supplies about 30 % of the plant's electricity (Melbourne Water, n.d.).

## 2.2 Data sources

Six files were used (Supplementary Table S0 lists sizes and SHA-256 checksums).

*Benchmark.* `Data-Melbourne_F.csv` (Bagherzadeh, 2021; CC BY 4.0), 1,382 rows and 20 columns: outflow, inflow, grid electricity (`total_grid`), ammonia, BOD, COD, total nitrogen (TN), ten weather fields and the date. The downloaded file matches the checksum published by the repository.

*Primary tables.* Five feature services of the Melbourne Water Open Data Hub were retrieved in full on 12 November 2026 through their public REST interface: daily grid electricity by feeder with on-site generation (2,588 days, 1 January 2014 to 31 January 2021); daily influent quality (1,382 samples); daily effluent quality (1,716 records); and hourly influent and effluent flow from 2009 (91,993 and 93,529 records). The Victorian government data catalogue lists the electricity, flow and influent-quality tables under a CC BY 4.0 licence (DataVic, 2024a–d); the licence of the effluent-quality table, which is used here only for one descriptive indicator, was not confirmed. According to the catalogue, the two feeders plus generation give the plant's total electricity consumption, the effluent series is the hourly flow rate at the outfall pump station, and influent samples are flow-weighted 24-h composites collected at 07:00 that relate to the preceding 24 h.

## 2.3 Provenance audit

Each benchmark column was compared with its source. For electricity, the benchmark value was matched to the sum of the two grid feeders. For flows, daily aggregates of the hourly source series were compared with the benchmark values under several alignments: the same calendar date, the following date, and row position (the *i*-th benchmark row against the *i*-th day of the source series). Candidate aggregation rules (day boundary, statistic, treatment of duplicated and missing records) were then searched until the benchmark values were reproduced to the three decimals given in the file. For quality variables, rows were classed as measured (a value exists in the source and equals the benchmark) or filled (no source value). Weather fields have no primary table in this study; they were screened for impossible values and for seasonal and event plausibility.

The electricity records are stamped at local midnight. To establish which 24 h each record covers, we correlated electricity with the mean of 24-h hourly-flow windows starting between −48 h and +36 h from the stamp.

## 2.4 Reconstructed table

All records were keyed to the local calendar day (Australia/Melbourne) that they describe: electricity to the day starting at its stamp (Section 3.1), hourly flows aggregated to local days with at least 20 valid hours (mean, maximum, standard deviation), and quality samples to the local date of their 07:00 stamp, which is the pairing used in the benchmark file. Because a composite collected at 07:00 covers mainly the previous day, the alternative pairing of each day with the sample collected the next morning was run as a sensitivity analysis. Grid import, on-site generation and their sum (hereafter total supply) were kept as separate series. Quality values were used only where measured; nothing was imputed in the table. The benchmark's weather fields were retained except sea-level pressure, visibility and gust, which failed screening (Section 3.1); because of the one-day label offset they are the previous day's weather. The effluent meter reported whole-day means below 10 % of the same-day influent mean on 21 days, 18 of them exactly zero, including 17 consecutive days in March 2019 during which inflow and electricity were normal. Such values are incompatible with a plant mass balance and were set to missing; all results are also reported without this screen.

## 2.5 Prediction experiments

*Targets.* Daily grid import and daily total supply (MWh d⁻¹).

*Explanation and forecasting.* Two tasks are kept apart. In the *explanatory* task, as in the earlier studies, the inputs include measurements of the same day (flows, quality); such a model describes what drives electricity use but cannot be run before the day begins. In the *day-ahead forecasting* task, every input is known at 00:00 of the day to be predicted.

*Feature sets on the benchmark rows (explanatory task).* **BM**: the 16 predictors of the benchmark file plus an annual harmonic, as distributed. **BM-flowfix**: BM with the two flow columns replaced by correctly dated daily means, everything else unchanged. **RC**: reconstructed inputs (influent mean, maximum and standard deviation; effluent mean; measured NH₃-N, COD and BOD; seven weather fields; annual harmonic). **RC-noflow**: RC without the four flow variables. **LAG**: target on the previous day, seven days earlier and the mean of the previous seven days, plus the harmonic. **RC-dyn**: RC plus the LAG variables. **BM-rowlag**: BM plus the target of the previous *row*, which is how lagged energy can be built from the file alone; 23 % of rows are not preceded by the previous calendar day (gaps of 2–20 days).

*Feature sets on the complete daily calendar.* Electricity and hourly flows exist for every day, so a second table with all days from 8 January 2014 to 28 June 2019 was used to compare, on a regularly spaced series, **LAGS** (target lags and harmonic), **FORECAST** (LAGS plus the flows of the previous days) and **EXPLAIN** (LAGS plus the flows of the same day). Weather and influent quality exist only on sampling days and were not used in this table.

*Models.* Five families were run with scikit-learn 1.9.1 (Pedregosa et al., 2011): ridge regression; random forest (Breiman, 2001; 300 trees); histogram gradient boosting (HGB), tuned over 12 configurations by three-fold expanding-window validation inside the training data; the classical gradient boosting machine (GBM; Friedman, 2001), the algorithm that performed best in Bagherzadeh et al. (2021); and a deep feed-forward network (MLP; three hidden layers of 64, 64 and 32 ReLU units, Adam, early stopping, mean of five initialisations). Recurrent networks and XGBoost/LightGBM were not available in the computing environment. Missing predictors were handled natively by HGB and by training-set median imputation with missingness indicators for the others.

*Naive references.* Previous-day value (persistence), value seven days earlier, mean of the previous seven days, training mean and training monthly mean.

*Validation.* The primary protocol is the chronological hold-out used by the earlier studies: benchmark labels up to 28 January 2018 for training (1,032 rows) and later rows for testing (346 rows); on the complete calendar the same date separates 1,483 training days from 515 test days. Selected models were also evaluated by shuffled five-fold cross-validation and by five-fold rolling-origin validation, with tuning nested inside each fold.

*Metrics and tests.* RMSE, MAE, MAPE, R², skill relative to persistence (1 − RMSE/RMSE of persistence) and MASE (Hyndman and Koehler, 2006). Forecasts were compared with the Diebold–Mariano test (Diebold and Mariano, 1995) with the small-sample correction of Harvey et al. (1997). Because the benchmark rows are irregularly spaced, the long-run variance was estimated with a Bartlett kernel on the calendar distance between observations (bandwidth 14 days), and confidence intervals were obtained by resampling non-overlapping calendar blocks of 21 days (2,000 resamples). Four comparisons per target were designated as primary and their p-values were adjusted by the Holm procedure (Supplementary Table S6); all other p-values are descriptive.

## 2.6 Augmentation experiment

Harrou et al. (2023) enlarged the 1,382 rows ten-fold by cubic-spline interpolation, then split the augmented table chronologically and used lagged energy as an input. We emulated this order of operations with the BM columns and HGB, and compared it with no augmentation and with augmentation of the training period only. The emulation follows the description in that paper; their code was not available to us.

## 2.7 Prediction intervals

Nominal 90 % intervals were built on the hold-out for HGB with RC and RC-dyn inputs by four methods: a Gaussian band of ±1.645 in-sample residual standard deviations; split conformal prediction (Angelopoulos and Bates, 2023), calibrated on the most recent 20 % of the training period; conformalized quantile regression (Romano et al., 2019); and adaptive conformal inference (Gibbs and Candès, 2021; step 0.02, trailing pool of 250 residuals). Coverage, mean width and the interval (Winkler) score are reported. Adaptive conformal inference adjusts its level after every observed miss, so its long-run coverage is close to nominal by construction; its coverage is reported for completeness and its width is the informative quantity.

## 2.8 Interpretation, generation check and out-of-time check

Permutation importance was computed on the hold-out for groups of related inputs, which avoids splitting importance among correlated columns. A two-parameter engineering baseline, E = a + b·Q, was fitted by least squares on the training period, with block-bootstrap intervals. Specific electricity use was computed per cubic metre of influent and per kilogram of COD removed.

Treating grid import plus generation as the plant's demand assumes that no generated electricity is exported. Three implications were examined on the daily record: generation should never approach total supply; a change in generation should be offset by an opposite change in grid import (slope −1 in first differences, with Newey–West standard errors); and total supply should not depend on generation once effluent flow is accounted for.

Finally, because the primary electricity table extends 19 months beyond the benchmark (to 31 January 2021), lag-only models trained on days up to 29 January 2018 were evaluated separately on that later window.

All steps are scripted in Python 3.13; the full pipeline runs in about seven minutes on two CPU cores.

# 3. Results

## 3.1 What the benchmark file contains

Table 1 summarises the audit.

**Table 1.** Comparison of the benchmark file with the primary Melbourne Water tables.

| Column(s) | Finding | Evidence |
|---|---|---|
| `total_grid` (target) | Equals feeder 1 + feeder 2, i.e. grid import. On-site generation is not included. | Exact match in 1,380 of 1,382 rows. Generation averages 115.8 MWh d⁻¹ against 275.1 MWh d⁻¹ of grid import (29.9 % of supply); r(grid, generation) = −0.33. |
| `avg_inflow`, `avg_outflow` | Daily means of flows from 31 Dec 2008 to Oct 2012, attached by row position to dates in 2014–2019. | Rule: mean of the hourly records of the *i*-th UTC day from 31 Dec 2008, omitting the record stamped 00:00 UTC. Inflow: all 1,382 of 1,382 values reproduced to three decimals. Outflow: 1,381 of 1,382 reproduced once two source days (1–2 May 2011) are skipped. Correlation with the flow of the day described: r = 0.12 (inflow), 0.14 (outflow). |
| `TN` | 90.0 % of values are fills. | TN was measured on 138 of the 1,382 sample days. |
| `BOD` | 30.4 % of values are fills. | All 420 filled values equal COD/2 exactly. |
| `COD`, `Am` | 3.0 % and 0.1 % fills. | 42 and 1 rows without a source value. |
| `SLP`, `VV`, `VG` | Not usable. | Pressure is 0 in 1,377 rows; visibility reaches 512 km (27 rows above 50 km); gust is 0 in 51.8 % of rows. |
| date | Label is one day earlier than the local day described by the energy value. | Correlation of day-to-day changes between total supply and a 24-h effluent-flow window peaks at 0.88 when the window starts at the record stamp, and is −0.33 for the preceding 24 h (Supplementary Table S3, Fig. S1). |

The target of every earlier study is therefore net grid import. Total supply rose from 375 MWh d⁻¹ in 2014 to 430 MWh d⁻¹ in 2020, while the generation share stayed near 30 %, with a dip to 27 % in 2017 (Fig. 1). Because generation varies independently of treatment load, it adds variance to grid import that process variables cannot explain; the variance of generation is 20 % of the variance of grid import.

![Daily grid import, on-site generation and total electricity supply at ETP, 2014–2021 (thin lines: daily values; thick lines: 30-day means). The benchmark file contains only the grid series up to mid-2019.](../results/figures/Fig1_grid_vs_total_supply.png){width=85%}

The flow finding is the most consequential. The benchmark values are reproduced exactly by one rule: take the hourly source records of consecutive UTC days starting on 31 December 2008, omit the record stamped 00:00 UTC, average the rest, and write the result to consecutive rows (Supplementary Table S1). The first row, labelled 1 January 2014, is the mean of the eleven records of 31 December 2008; the last row, labelled 27 June 2019, is the inflow of 12 October 2012 (Fig. 2). For the outflow the same rule reproduces 1,381 of 1,382 values, the sequence omitting 1–2 May 2011. The omission of midnight records is the pattern left when timestamps without a time component are dropped or mis-parsed, and a positional join is what results when two tables are concatenated without matching on date. The benchmark's inflow and outflow thus describe a period five years before the dates to which they are attached, with no physical relation to the electricity values beside them.

![Benchmark inflow against the source series: (a) on the day described by the row; (b) reconstructed by the rule in the text; (c) the first 120 rows.](../results/figures/Fig2_flow_misdating.png){width=95%}

![Measured and filled influent-quality values in the benchmark file.](../results/figures/Fig3_filled_quality_values.png){width=50%}

## 3.2 Skill with the file as distributed

On the chronological hold-out, persistence gave an RMSE of 40.7 MWh d⁻¹ (MAPE 10.5 %) for grid import (Table 2). No model trained on the BM inputs did better: the five families gave 43.5–44.6 MWh d⁻¹ (R² between −0.19 and −0.13), none distinguishable from persistence (Diebold–Mariano p = 0.21–0.42; Holm-adjusted p = 0.84 for the primary test). For total supply the same inputs gave 39.1–41.1 MWh d⁻¹ against 40.3 for persistence (p = 0.69–0.93). Adding the previous row's energy brought HGB to 37.0 MWh d⁻¹, 9 % better than persistence and no better than three lag variables alone (35.5 MWh d⁻¹). H1 is supported.

These figures are close to those published for the same split: RMSE 33.9–39.8 MWh d⁻¹ with R² of 0.00–0.18 (Bagherzadeh et al., 2021), 37.1–41.4 MWh d⁻¹ (Alali et al., 2023), 37.8 MWh d⁻¹ without augmentation (Harrou et al., 2023) and 35.3–51.5 MWh d⁻¹ (Khaled et al., 2026). The earlier studies removed about 5 % of rows as outliers, so the comparison is approximate, but the published errors sit in the range that a persistence forecast attains.

**Table 2.** Chronological hold-out on the benchmark rows (346 rows; explanatory task). HGB unless stated; 95 % calendar-block bootstrap interval for RMSE in brackets; skill is relative to persistence. All model families are in Supplementary Table S5.

| Inputs | Grid import: RMSE (MWh d⁻¹) | MAPE (%) | R² | Skill (%) | Total supply: RMSE (MWh d⁻¹) | MAPE (%) | R² | Skill (%) |
|---|---|---|---|---|---|---|---|---|
| Persistence | 40.7 [35.8, 45.1] | 10.5 | 0.01 | 0 | 40.3 [36.1, 44.2] | 7.3 | −0.16 | 0 |
| BM (file as distributed) | 43.5 [37.0, 50.4] | 12.7 | −0.13 | −7 | 39.1 [34.2, 44.3] | 7.8 | −0.10 | 3 |
| BM, GBM | 43.8 | 12.8 | −0.15 | −8 | 39.8 | 8.0 | −0.13 | 1 |
| BM, deep MLP | 43.9 | 12.8 | −0.15 | −8 | 40.0 | 7.9 | −0.15 | 1 |
| BM-rowlag | 37.0 | 10.5 | 0.18 | 9 | 35.3 | 7.0 | 0.11 | 12 |
| BM-flowfix | 27.8 [22.5, 33.0] | 7.7 | 0.54 | 32 | 23.3 [17.9, 29.1] | 4.4 | 0.61 | 42 |
| RC-noflow | 45.2 | 13.3 | −0.22 | −11 | 40.8 | 8.2 | −0.19 | −1 |
| RC | 28.0 [22.5, 33.3] | 7.7 | 0.53 | 31 | 22.7 [17.1, 28.7] | 4.2 | 0.63 | 44 |
| LAG | 35.5 | 10.0 | 0.25 | 13 | 33.1 | 6.5 | 0.21 | 18 |
| RC-dyn | 24.0 [20.3, 28.1] | 6.6 | 0.66 | 41 | 19.0 [16.1, 22.5] | 3.7 | 0.74 | 53 |
| RC-dyn, ridge | 24.2 | 6.6 | 0.65 | 41 | 18.6 | 3.6 | 0.75 | 54 |
| RC-dyn, random forest | 24.1 | 6.9 | 0.65 | 41 | 18.7 | 3.7 | 0.75 | 54 |
| RC-dyn, GBM | 23.4 | 6.6 | 0.67 | 43 | 17.8 | 3.5 | 0.77 | 56 |
| RC-dyn, deep MLP | 26.0 | 7.1 | 0.59 | 36 | 22.5 | 4.0 | 0.64 | 44 |

## 3.3 Effect of correcting the data

Replacing only the two misdated flow columns (BM-flowfix) lowered the HGB hold-out RMSE for grid import from 43.5 to 27.8 MWh d⁻¹, a change of −15.7 MWh d⁻¹ (95 % interval −19.4 to −11.1; Holm-adjusted p < 0.001), and raised R² from −0.13 to 0.54 (Fig. 4a). For total supply the change was −15.9 MWh d⁻¹ (−20.2 to −10.0). The fully reconstructed inputs gave the same result (28.0 MWh d⁻¹), and removing the flows from them returned the error to the level of the benchmark file (45.2 MWh d⁻¹). The other corrections — dropping the filled TN, the defective weather fields and the COD/2 fills — had no measurable effect on accuracy; their importance lies in interpretation (Section 4).

With calendar-true lags added (RC-dyn), RMSE fell to 23.4–24.2 MWh d⁻¹ for grid import and 17.8–19.0 MWh d⁻¹ for total supply across ridge, random forest, HGB and GBM, 41–43 % and 53–56 % better than persistence. The deep network was the weakest family on the RC and RC-dyn inputs (26.0 and 22.5 MWh d⁻¹ for RC-dyn), as is common for small tabular samples. The spread among the four other families (0.8 and 1.2 MWh d⁻¹) is small beside the 13–16 MWh d⁻¹ gained from the data correction. H2 is supported.

Total supply was fitted better than grid import in every configuration that contained real signal (R² 0.63 against 0.53 for RC; 0.74 against 0.66 for RC-dyn; MAPE 3.7 % against 6.6 %), which supports H3. Without the mass-balance screen on the effluent meter, RC-dyn gave 25.3 (grid) and 22.5 MWh d⁻¹ (total); the differences from the screened results were not significant (p = 0.25 and 0.23), and none of the conclusions changes. Pairing each day with the composite sample collected the next morning (available for 1,036 of the 1,378 rows) changed the HGB hold-out RMSE by +0.1 to +0.6 MWh d⁻¹ (p = 0.10–0.55).

![Re-evaluation summary. (a) Hold-out RMSE by input set, with 95 % intervals; dashed lines mark persistence. (b) R² of the same models under three validation protocols. (c) Test MAPE with and without ten-fold spline augmentation. (d) Observed and fitted total supply over the last 150 hold-out rows.](../results/figures/Fig4_reevaluation_summary.png){width=95%}

## 3.4 Same-day explanation versus day-ahead forecasting

The models in Table 2 use the flow of the day they describe. On the complete daily calendar, where the two tasks can be separated on a regular series (515 test days), the difference is large (Table 3). With same-day flows, the five families reached 17.9–20.7 MWh d⁻¹ for total supply (MAPE 3.5–3.8 %, R² 0.70–0.77). With only information available before the day begins, they reached 31.1–31.8 MWh d⁻¹ (MAPE 6.1–6.2 %, R² 0.28–0.32), which is 15–17 % better than persistence and about 1 MWh d⁻¹ better than target lags alone. The gap between the two tasks was 10–14 MWh d⁻¹ for total supply and 8–11 MWh d⁻¹ for grid import (p < 0.001 for every family). Split-conformal 90 % intervals for the day-ahead forecasts covered 87–90 % of test days with widths of 94–98 MWh d⁻¹ for total supply, about ±12 % of mean demand, and 107–111 MWh d⁻¹ for grid import.

**Table 3.** Chronological hold-out on the complete daily calendar (515 test days). Range over the five model families.

| Task and inputs | Grid import: RMSE (MWh d⁻¹) | MAPE (%) | R² | Skill (%) | Total supply: RMSE (MWh d⁻¹) | MAPE (%) | R² | Skill (%) |
|---|---|---|---|---|---|---|---|---|
| Persistence | 38.2 | 10.2 | 0.11 | 0 | 37.6 | 6.9 | 0.00 | 0 |
| Target lags only | 33.2–33.8 | 9.4–9.6 | 0.30–0.33 | 11–13 | 31.9–32.5 | 6.3–6.4 | 0.25–0.28 | 13–15 |
| Day-ahead forecast: lags + flows of previous days | 32.0–33.1 | 9.2–9.4 | 0.33–0.37 | 13–16 | 31.1–31.8 | 6.1–6.2 | 0.28–0.32 | 15–17 |
| Explanation: lags + flows of the same day | 22.2–24.1 | 6.2–6.6 | 0.64–0.70 | 37–42 | 17.9–20.7 | 3.5–3.8 | 0.70–0.77 | 45–52 |

## 3.5 Effect of the validation protocol

With the BM inputs, shuffled five-fold cross-validation gave R² = 0.31 for grid import, whereas rolling-origin validation gave −0.04 and the chronological hold-out −0.13 (Table 4, Fig. 4b). The shuffled estimate credits the models with interpolating slow level changes between neighbouring days, which is unavailable in operation. For inputs that contain real signal the gap narrows but remains (RC: 0.63 shuffled against 0.48–0.53 in time order).

**Table 4.** R² of HGB for grid import under three validation protocols (explanatory task; all metrics in Supplementary Table S8).

| Inputs | Shuffled 5-fold | Rolling origin | Chronological hold-out |
|---|---|---|---|
| BM | 0.31 | −0.04 | −0.13 |
| BM-rowlag | 0.40 | 0.27 | 0.18 |
| RC | 0.63 | 0.48 | 0.53 |
| RC-dyn | 0.73 | 0.68 | 0.66 |

The augmentation experiment gave a sharper result (Table 5, Fig. 4c). Interpolating the whole table ten-fold before the chronological split and feeding the model the previous (interpolated) row produced a test MAPE of 1.35 %, RMSE of 4.9 MWh d⁻¹ and R² of 0.985. The values reported for this procedure by Harrou et al. (2023) are 1.36 %, 5.8 MWh d⁻¹ and 0.978. Copying the previous interpolated row, with no model, gave 1.29 %. When the split preceded augmentation and the untouched test rows were scored, MAPE was 11.6 %, the same as copying the previous row of the file. The low error is therefore a property of the smooth interpolant between adjacent real observations and carries no information on predictive skill. H4 is supported.

**Table 5.** Ten-fold cubic-spline augmentation, grid import, BM inputs with lagged energy (HGB).

| Order of operations | Test rows | MAPE (%) | RMSE (MWh d⁻¹) | R² | MAPE of copying the previous row (%) |
|---|---|---|---|---|---|
| No augmentation | 346 | 11.1 | 38.0 | 0.14 | 11.6 |
| Augment all rows, then split | 3,460 | 1.35 | 4.9 | 0.985 | 1.29 |
| Split, then augment training rows only | 346 | 11.6 | 43.1 | −0.11 | 11.6 |

## 3.6 Prediction intervals

The Gaussian band covered 78 % (total supply) and 86 % (grid import) of hold-out days at a nominal 90 % (Table 6, Fig. 5), because in-sample residuals of a flexible model understate out-of-sample error. Split conformal intervals, calibrated on a held-out block, erred on the wide side (93.4–95.4 % coverage), and conformalized quantile regression lay between 89.0 and 95.1 %. Adaptive conformal inference returned 89.9–91.0 %, as expected from its design; its value here is that the intervals were narrower than the static conformal ones at the same level (57 against 68 MWh d⁻¹ for total supply with RC-dyn). These intervals belong to the explanatory task; the day-ahead intervals in Section 3.4 are 40–70 % wider.

**Table 6.** Nominal 90 % intervals on the hold-out, HGB with RC-dyn inputs (explanatory task).

| Method | Grid: coverage (%) | Mean width (MWh d⁻¹) | Total: coverage (%) | Mean width (MWh d⁻¹) |
|---|---|---|---|---|
| Gaussian band, in-sample residuals | 85.8 | 65.0 | 78.3 | 42.9 |
| Split conformal | 93.6 | 86.7 | 94.8 | 68.4 |
| Conformalized quantile regression | 89.0 | 82.3 | 91.6 | 69.5 |
| Adaptive conformal inference | 91.0 | 80.7 | 90.5 | 57.1 |

![Top: hold-out predictions of total supply with 90 % adaptive conformal intervals. Bottom: coverage over the trailing 60 rows for three interval methods.](../results/figures/Fig5_prediction_intervals.png){width=85%}

## 3.7 What drives daily electricity use

Permuting effluent flow raised the hold-out RMSE by 24.4 MWh d⁻¹ for total supply and 19.8 MWh d⁻¹ for grid import; no other input group contributed more than 2.1 MWh d⁻¹ (influent flow 2.1, influent quality 1.2, weather 0.3, season 0.5 for total supply; Fig. 6a). The response to effluent flow is close to linear over the observed range (Fig. 6b). The engineering baseline fitted on the training period was

E_total = 238 (228–250) + 0.43 (0.40–0.46) · Q_eff,

with E in MWh d⁻¹ and Q_eff in ML d⁻¹, so that the slope is in kWh m⁻³. On the hold-out this line gave an RMSE of 23.3 MWh d⁻¹ (R² = 0.61, MAPE 4.6 %), the same as HGB with all 16 reconstructed inputs (22.7 MWh d⁻¹). The same line in influent flow explained almost nothing (hold-out R² = 0.02). About 61 % of supply at mean flow is a flow-independent base load.

Median specific electricity use was 1.04 kWh m⁻³ of influent on the basis of total supply and 0.72 kWh m⁻³ on the basis of grid import (1.26 against 0.88 kWh per kg COD removed). An indicator computed from the benchmark target would understate the plant's electricity intensity by 31 % (Fig. 6c).

*Is total supply the plant's demand?* Over the 2,588 days of the record, grid import never fell below 117 MWh d⁻¹ and generation exceeded half of supply on only two days (maximum 53 %), so there was no day on which the plant could have been a net exporter. A change in generation was offset by an opposite change in grid import of 0.84 MWh per MWh (95 % interval 0.77–0.91) once the change in effluent flow was accounted for, and total supply rose by 0.11 MWh (0.03–0.19) per MWh of generation (Supplementary Table S12). The offset is therefore close to, but statistically below, one-for-one. Measurement error in the generation series, auxiliary consumption of the power station, or common drivers of generation and demand would each produce this pattern; small exports cannot be excluded from these data. Total supply is used here as the publisher defines it, as the best available measure of demand.

![(a) Grouped permutation importance on the hold-out. (b) Partial dependence on effluent and influent flow. (c) Median specific electricity use by year.](../results/figures/Fig6_drivers_and_specific_energy.png){width=95%}

## 3.8 Out-of-time check

For the 583 days after the benchmark ends (29 June 2019 to 31 January 2021), only electricity is published. A ridge model on lagged values, trained on data up to January 2018, kept its advantage over persistence in this later window (grid import: skill 11.8 % inside the benchmark period and 12.8 % after it; total supply: 13.9 % and 13.4 %; all p < 0.001), and its split-conformal 90 % intervals covered 89.7 % (grid) and 86.1 % (total) of days. The modest size of this skill is consistent with Section 3.4: without same-day flow, forecasts improve on persistence by 12–17 % (lag-1 autocorrelation is 0.68 for grid import).

# 4. Discussion

**The benchmark carried almost no usable signal.** In the file as distributed, the only inputs with a physical link to the dated electricity values were ammonia, measured COD, about two-thirds of the BOD values, a tenth of the TN values and the previous day's weather. These proved to be weak predictors: the reconstructed inputs without flow did no better than the training mean. This explains why six studies using increasingly elaborate models reported R² values near zero or errors near 10–15 % on the chronological split: they were, in effect, competing to model noise around a slowly drifting level. It also explains why lagged energy appeared so valuable in those studies (Alali et al., 2023; Harrou et al., 2023), since it was the one informative input.

**Earlier feature rankings need re-reading.** TN was reported as the most influential variable (Bagherzadeh et al., 2021), yet nine in ten TN values are fills whose rule is undocumented, and inflow was ranked among the top variables although the column belongs to another period. Importance scores computed from training-set impurity can assign weight to such columns; in our chronological evaluation neither improved predictions. We suggest that process conclusions drawn from the benchmark (for example, that nitrogen load governs energy use at ETP) should not be relied upon.

**Effluent pumping, not biological load, governs day-to-day variation.** Once correctly dated, effluent flow alone explained 61 % of hold-out variance in total supply, and influent flow almost none. This is consistent with the plant layout: the effluent series is metered at the outfall pump station (DataVic, 2024c), which lifts treated effluent into a 56-km pipeline, and storage on site decouples the hourly discharge from the hourly inflow. The slope of 0.43 kWh m⁻³ is of the order expected for a long transfer main; the base load of about 240 MWh d⁻¹ covers aeration, tertiary treatment and solids handling, whose day-to-day variation was not resolved by influent NH₃-N, COD or BOD. Variables that would resolve the remaining 25–35 % of variance (blower air flow, dissolved-oxygen set-points, ozone dose, sludge processing) are not in the open tables.

**Explanation is not forecasting.** Effluent flow is measured at the pumps that consume the electricity, during the same day. A model given that flow describes the load; it does not predict it. When only earlier days were available to the models, errors rose from 18–21 to 31–32 MWh d⁻¹ for total supply, and previous-day flows added about 1 MWh d⁻¹ to what the target's own history provides. The benchmark task as posed in the literature, with same-day inputs, is therefore an explanatory task, and accuracies reported for it should not be read as forecasting accuracies. A useful day-ahead forecast at this plant would need the planned discharge schedule, which the operator sets and which is not published; the size of the gap (10–14 MWh d⁻¹) indicates how much such information could be worth.

**The target matters for both prediction and benchmarking.** Grid import is what the utility pays for, and it is a legitimate target for procurement. It is, however, the difference between demand and on-site generation, and generation depends on digester gas production and engine availability. Models that aim to explain energy use from treatment variables should use total supply; models of grid import should include generation or its drivers. The offset between generation and grid import was 0.84 for 1, not exactly one-for-one, so total supply is itself an approximation of demand whose error we could not resolve with the open data. Intensity indicators built on the benchmark target are about 30 % too low and are not comparable with plants that have no generation (Longo et al., 2016; Gu et al., 2017).

**Simple references would have exposed the problem.** A previous-day forecast, a two-parameter line and a time-ordered split are each a few lines of code. Had they been reported alongside the first models, the absence of skill would have prompted a look at the inputs. The augmentation case is instructive for another reason: the 1.4 % MAPE is reproduced by copying the previous interpolated row, so any model given that row will appear excellent. Interpolated points are functions of their real neighbours on both sides, and a chronological split does not separate them. These observations agree with general accounts of leakage (Kapoor and Narayanan, 2023; Zhu et al., 2023) and extend them with a quantified wastewater example.

**Uncertainty should be validated, not asserted.** The band used with the original study, a normal-theory interval on residuals, was not checked for coverage. In our re-evaluation a comparable band under-covered by 4–12 points. Intervals calibrated on held-out data did not under-cover, at the price of greater width, and for day-ahead use they were ±12 % of mean demand. The practical point is modest: coverage should be measured on unseen days, and intervals should be reported for the task that will actually be performed.

**Recommendations.** For dataset publishers: deposit the join script with the table; keep measured and filled values distinguishable; state the time zone and whether a daily stamp marks the start or end of the day. For users: verify a sample of rows against the primary source, report persistence and a physically motivated baseline, split in time before any transformation that mixes rows, and report interval coverage. For this dataset: use the reconstructed table released with this paper, or rebuild it from the primary tables with the scripts provided.

# 5. Practical implications

For Melbourne Water and similar utilities, the open data support a day-ahead forecast of total supply with an RMSE of about 31 MWh d⁻¹ and a 90 % interval of about ±12 %, only modestly better than repeating yesterday's value; the planned effluent discharge is the missing input that could close much of the gap to the 18–19 MWh d⁻¹ reached with same-day flow. For researchers, the reconstructed ETP table offers a small but sound test bed with documented reference points for total supply: persistence 38–40 MWh d⁻¹; day-ahead forecast 31 MWh d⁻¹; two-parameter line in same-day effluent flow 23 MWh d⁻¹; best explanatory model 18 MWh d⁻¹. New methods should state which of the two tasks they address and be judged against the corresponding reference on the chronological split.

# 6. Limitations

The study concerns one plant and one benchmark; the prevalence of similar defects elsewhere is unknown. The reconstruction rule for the flow columns is inferred from the data: it reproduces 1,382 of 1,382 inflow values and 1,381 of 1,382 outflow values, but the original processing script was not available and the cause of the positional join is a conjecture. The authors of the benchmark and of the cited studies have not yet been consulted and should be given the opportunity to respond before publication. Weather fields could not be compared with a primary meteorological record and were only screened. Total supply approximates demand; the generation offset of 0.84 leaves a part that the open data cannot attribute. Day-ahead forecasts used no weather forecast and no operating plan, so they are a lower bound on what the operator could achieve. The mass-balance screen on the effluent meter was defined after the anomaly was noticed in hold-out predictions; results without it are reported. Influent quality contributed little under either pairing of composite samples with days, so its small importance is not an artefact of sample timing, but it may reflect the absence of load variables at sub-daily resolution. The deep model was a feed-forward network; recurrent networks, XGBoost, LightGBM and SHAP were not available in the computing environment, and the hyper-parameter searches were deliberately small. Since the conclusions concern data and validation, and five families agree, we do not expect them to depend on this, but it was not tested. Hypotheses were formulated after the audit and were not pre-registered; apart from the eight Holm-adjusted primary tests, p-values are descriptive. The emulation of the augmentation procedure follows a published description and may differ in detail from the original implementation. Forward-citation screening used OpenAlex and titles only.

# 7. Conclusions

1. The public ETP energy benchmark differs from its primary sources in four ways: an incomplete target, flow columns that belong to 2008–2012 (reproduced exactly by a simple rule), undisclosed fills and a one-day label offset.
2. With the file as distributed, none of five model families outperforms a previous-day persistence forecast under chronological validation; shuffled validation and augmentation before splitting conceal this.
3. Correcting the flow dates, without changing the algorithm, reduces the hold-out RMSE for grid import by 36 %; the choice among ridge, random forest and two boosting methods changes it by about 1 MWh d⁻¹.
4. Same-day effluent pumping explains most of the day-to-day variation (RMSE 18–19 MWh d⁻¹ for total supply), and a two-parameter line in effluent flow nearly matches the learned models; this is explanation, not forecasting.
5. True day-ahead forecasts from the open data reach 31–32 MWh d⁻¹, 15–17 % better than persistence, with 90 % intervals of about ±12 %.
6. A Gaussian residual band under-covers on unseen days; intervals calibrated on held-out data do not.

# Data availability statement

The benchmark file is available from Mendeley Data (https://doi.org/10.17632/pprkvz3vbd.1, CC BY 4.0). The primary tables are available from the Melbourne Water Open Data Hub (https://data-melbournewater.opendata.arcgis.com; services `MWC_ETP_Daily_ElectricityUse_From2014`, `MWC_ETP_Daily_InfluentQuality_From2014`, `MWC_ETP_Daily_EffluentQuality_From2014`, `MWC_ETP_Hourly_InfluentFlow_From2009`, `MWC_ETP_Hourly_EffluentFlow_From2009`). The copies used here, their SHA-256 checksums and the reconstructed daily table are provided with the code in a public repository (https://github.com/phamnam123lite-svg/etp-energy-benchmark-audit), and an archived release with a DOI will be created on acceptance. The data are redistributed with attribution to Melbourne Water and to Bagherzadeh (2021). Supplementary tables S0–S14, Fig. S1 and Note S1 are provided in the Supplementary Material.

# Code availability statement

All Python scripts (`01_build_dataset.py` to `11_make_supplement.py`, `run_all.py`), the list of library versions and all result tables are in the same repository under the MIT licence. `python run_all.py` regenerates every table and figure from the raw files in about seven minutes. No proprietary software was used.

# CRediT authorship contribution statement

**Viet Nam Tran:** Conceptualization, Methodology, Formal analysis, Data curation, Validation, Writing – review and editing.

# Declaration of generative AI in the research and writing process

During the preparation of this work the author used Claude (Anthropic) to search the literature, retrieve the open data, write and run the analysis code, and draft the text and figures. The author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

# Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

# Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# References

Alali, Y., Harrou, F., Sun, Y., 2022. Predicting energy consumption in wastewater treatment plants through light gradient boosting machine: a comparative study. 2022 10th International Conference on Systems and Control (ICSC), 137–142. https://doi.org/10.1109/ICSC57768.2022.9993872

Alali, Y., Harrou, F., Sun, Y., 2023. Unlocking the potential of wastewater treatment: machine learning based energy consumption prediction. Water 15 (13), 2349. https://doi.org/10.3390/w15132349

Allen, L., Cordiner, J., 2025. Knowledge-enhanced data-driven modeling of wastewater treatment processes for energy consumption prediction. Comput. Chem. Eng. 194, 108982. https://doi.org/10.1016/j.compchemeng.2024.108982

Angelopoulos, A.N., Bates, S., 2023. Conformal prediction: a gentle introduction. Found. Trends Mach. Learn. 16 (4), 494–591. https://doi.org/10.1561/2200000101

Bagherzadeh, F., 2021. Full scale wastewater treatment plant data. Mendeley Data, V1. https://doi.org/10.17632/pprkvz3vbd.1

Bagherzadeh, F., Nouri, A.S., Mehrani, M.-J., Thennadil, S., 2021. Prediction of energy consumption and evaluation of affecting factors in a full-scale WWTP using a machine learning approach. Process Saf. Environ. Prot. 154, 458–466. https://doi.org/10.1016/j.psep.2021.08.040

Bahramian, M., Dereli, R.K., Zhao, W., Giberti, M., Casey, E., 2023. Data to intelligence: the role of data-driven models in wastewater treatment. Expert Syst. Appl. 217, 119453. https://doi.org/10.1016/j.eswa.2022.119453

Bergmeir, C., Benítez, J.M., 2012. On the use of cross-validation for time series predictor evaluation. Inf. Sci. 191, 192–213. https://doi.org/10.1016/j.ins.2011.12.028

Breiman, L., 2001. Random forests. Mach. Learn. 45 (1), 5–32. https://doi.org/10.1023/A:1010933404324

Cardoso, B.J., Rodrigues, E., Gaspar, A.R., Gomes, Á., 2021. Energy performance factors in wastewater treatment plants: a review. J. Clean. Prod. 322, 129107. https://doi.org/10.1016/j.jclepro.2021.129107

DataVic, 2024a. Melbourne Wastewater Treatment (East) – Daily Grid Electricity Consumption. Melbourne Water Corporation. https://discover.data.vic.gov.au/dataset/melbourne-wastewater-treatment-east-daily-grid-electricity-consumption (accessed 12 November 2026)

DataVic, 2024b. Wastewater Inlet – Hourly Flow – Eastern Treatment Plant (ETP). Melbourne Water Corporation. https://discover.data.vic.gov.au/dataset/wastewater-inlet-hourly-flow-eastern-treatment-plant-etp (accessed 12 November 2026)

DataVic, 2024c. Wastewater Outlet – Hourly Flow – Eastern Treatment Plant (ETP). Melbourne Water Corporation. https://discover.data.vic.gov.au/dataset/wastewater-outlet-hourly-flow-eastern-treatment-plant-etp (accessed 12 November 2026)

DataVic, 2024d. Wastewater Inlet – Daily Raw Water Quality – Eastern Treatment Plant (ETP). Melbourne Water Corporation. https://discover.data.vic.gov.au/dataset/wastewater-inlet-daily-raw-water-quality-eastern-treatment-plant-etp (accessed 12 November 2026)

Diebold, F.X., Mariano, R.S., 1995. Comparing predictive accuracy. J. Bus. Econ. Stat. 13 (3), 253–263. https://doi.org/10.1080/07350015.1995.10524599

Friedman, J.H., 2001. Greedy function approximation: a gradient boosting machine. Ann. Stat. 29 (5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Gibbs, I., Candès, E., 2021. Adaptive conformal inference under distribution shift. Adv. Neural Inf. Process. Syst. 34.

Gu, Y., Li, Y., Li, X., Luo, P., Wang, H., Robinson, Z.P., Wang, X., Wu, J., Li, F., 2017. The feasibility and challenges of energy self-sufficient wastewater treatment plants. Appl. Energy 204, 1463–1475. https://doi.org/10.1016/j.apenergy.2017.02.069

Harrou, F., Cheng, T., Sun, Y., Leiknes, T., Ghaffour, N., 2021. A data-driven soft sensor to forecast energy consumption in wastewater treatment plants: a case study. IEEE Sens. J. 21 (4), 4908–4917. https://doi.org/10.1109/JSEN.2020.3030584

Harrou, F., Dairi, A., Dorbane, A., Sun, Y., 2023. Energy consumption prediction in water treatment plants using deep learning with data augmentation. Results Eng. 20, 101428. https://doi.org/10.1016/j.rineng.2023.101428

Harvey, D., Leybourne, S., Newbold, P., 1997. Testing the equality of prediction mean squared errors. Int. J. Forecast. 13 (2), 281–291. https://doi.org/10.1016/S0169-2070(96)00719-4

Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy. Int. J. Forecast. 22 (4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001

Kapoor, S., Narayanan, A., 2023. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 4 (9), 100804. https://doi.org/10.1016/j.patter.2023.100804

Khaled, Q., Mallak, B., Kaymak, U., Genga, L., 2026. Explainable uncertainty quantification for wastewater treatment energy prediction via interval type-2 neuro-fuzzy system. arXiv:2601.18897.

Longo, S., d'Antoni, B.M., Bongards, M., Chaparro, A., Cronrath, A., Fatone, F., Lema, J.M., Mauricio-Iglesias, M., Soares, A., Hospido, A., 2016. Monitoring and diagnosis of energy consumption in wastewater treatment plants. A state of the art and proposals for improvement. Appl. Energy 179, 1251–1268. https://doi.org/10.1016/j.apenergy.2016.07.043

Mateos Bravo, J., Pacheco Rodríguez, C.M., Berrocal, J., Herrera, J.L., Cortés Pérez, J.P., 2026. Data quality over model complexity: a leakage-aware, horizon-aware forecasting pipeline for degraded sensor streams in a small-population wastewater treatment plant. Comput.-Aided Civ. Infrastruct. Eng. 49, 100134. https://doi.org/10.1016/j.cacaie.2026.100134

Melbourne Water, 2023. Major Hazard Facility Safety Case Summary: Eastern Treatment Plant, March 2023. https://melbournewater.com.au/media/21291/download

Melbourne Water, n.d. Melbourne Water wins Earth Award for bioenergy project. https://www.melbournewater.com.au/node/8231 (accessed 12 November 2026)

Newhart, K.B., Holloway, R.W., Hering, A.S., Cath, T.Y., 2019. Data-driven performance analyses of wastewater treatment plants: a review. Water Res. 157, 498–513. https://doi.org/10.1016/j.watres.2019.03.030

Oliveira, P., Fernandes, B., Analide, C., Novais, P., 2021. Forecasting energy consumption of wastewater treatment plants with a transfer learning approach for sustainable cities. Electronics 10 (10), 1149. https://doi.org/10.3390/electronics10101149

Paepae, T., 2026. Temporally admissible and explainability-driven one-day-ahead soft-sensing of effluent total nitrogen in wastewater treatment: mechanistic feature engineering, statistical benchmarking, and process discovery. Results Eng. 32, 112202. https://doi.org/10.1016/j.rineng.2026.112202

Pedregosa, F., et al., 2011. Scikit-learn: machine learning in Python. J. Mach. Learn. Res. 12, 2825–2830.

Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J.J., Schröder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F., Dormann, C.F., 2017. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40 (8), 913–929. https://doi.org/10.1111/ecog.02881

Romano, Y., Patterson, E., Candès, E., 2019. Conformalized quantile regression. Adv. Neural Inf. Process. Syst. 32.

Therrien, J.-D., Nicolaï, N., Vanrolleghem, P.A., 2020. A critical review of the data pipeline: how wastewater system operation flows from data to intelligence. Water Sci. Technol. 82 (12), 2613–2634. https://doi.org/10.2166/wst.2020.393

Torregrossa, D., Leopold, U., Hernández-Sancho, F., Hansen, J., 2018. Machine learning for energy cost modelling in wastewater treatment plants. J. Environ. Manage. 223, 1061–1067. https://doi.org/10.1016/j.jenvman.2018.06.092

Zhang, S., Wang, H., Keller, A.A., 2021. Novel machine learning-based energy consumption model of wastewater treatment plants. ACS ES&T Water 1 (12), 2531–2540. https://doi.org/10.1021/acsestwater.1c00283

Zhu, J.-J., Yang, M., Ren, Z.J., 2023. Machine learning in environmental research: common pitfalls and best practices. Environ. Sci. Technol. 57 (46), 17671–17689. https://doi.org/10.1021/acs.est.3c00026
