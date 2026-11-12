# Data: sources, licences and how the files were obtained

All files in `raw/` are unmodified in content. Checksums are in `raw/SHA256SUMS.txt` and are verified by `code/01_build_dataset.py`.

| File | Source | Licence | Notes |
|---|---|---|---|
| `Data-Melbourne_F.csv` | Bagherzadeh, F. (2021). *Full scale wastewater treatment plant data*. Mendeley Data, V1. https://doi.org/10.17632/pprkvz3vbd.1 | CC BY 4.0 | The benchmark under audit. SHA-256 equals the hash published by the repository. |
| `MWC_ETP_Daily_ElectricityUse_From2014.csv` | Melbourne Water Corporation, *Melbourne Wastewater Treatment (East) – Daily Grid Electricity Consumption* | CC BY 4.0 (DataVic) | Two grid feeders and on-site generation, kWh per day. |
| `MWC_ETP_Hourly_InfluentFlow_From2009_compact.csv` | Melbourne Water Corporation, *Wastewater Inlet – Hourly Flow – ETP* | CC BY 4.0 (DataVic) | Compact form, see below. |
| `MWC_ETP_Hourly_EffluentFlow_From2009_compact.csv` | Melbourne Water Corporation, *Wastewater Outlet – Hourly Flow – ETP* | CC BY 4.0 (DataVic) | Flow rate at the outfall pump station. Compact form. |
| `MWC_ETP_Daily_InfluentQuality_From2014.csv` | Melbourne Water Corporation, *Wastewater Inlet – Daily Raw Water Quality – ETP* | CC BY 4.0 (Research Data Australia record of the DataVic entry) | Flow-weighted 24-h composites collected at 07:00, relating to the preceding 24 h. |
| `MWC_ETP_Daily_EffluentQuality_From2014.csv` | Melbourne Water Corporation, *Wastewater Outlet – Daily Treated Water Quality – ETP* | **not confirmed** | Grab samples. Used only for the kWh per kg COD removed indicator (table T12). Remove this file if the licence turns out to be restrictive; all other results are unaffected except that column. |

Melbourne Water tables: https://data-melbournewater.opendata.arcgis.com (feature services named as the files, on `services5.arcgis.com/ZSYwjtv8RKVhkXIL`). Catalogue entries with licence information: https://discover.data.vic.gov.au (search "Eastern Treatment Plant"). Melbourne Water makes no representations as to the accuracy or completeness of the data.

## Retrieval (12 November 2026)

The analysis machine had no direct internet access. Files were fetched in a web browser and transferred to the analysis machine as text.

- The Mendeley file was fetched from the repository's public download URL; its SHA-256 matches the hash listed by the Mendeley Data API.
- Each Melbourne Water table was read in full through the ArcGIS REST `query` endpoint (`where=1=1`, `outFields=*`, ordered by `ObjectId`, pages of 2,000 records) and written as CSV with the service's field order; `null` became an empty field. SHA-256 was computed in the browser on the assembled text and again after transfer; both agree.
- **Compact hourly files.** The two hourly services return `recordstart`, `recordend` (epoch milliseconds), the flow in kL/s and `ObjectId`. To keep the files small only two columns were kept: `start_epoch_hour` = `recordstart` / 3,600,000 (hours since 1970-01-01 00:00 UTC; every `recordstart` is an exact hour) and the flow. In each series 22 records have `recordend − recordstart ≠ 1 h` and 11 hour stamps are duplicated; duplicates are averaged in `01_build_dataset.py`.

`code/00_download_data.py` repeats this retrieval with the Python standard library. It has not been executed by the author. If Melbourne Water revises a table, its checksum will change; the script then reports the difference and leaves the stored file untouched.

## Processed files

| File | Content |
|---|---|
| `processed/etp_daily_reconstructed.csv` | One row per local calendar day, 1 Jan 2014 – 31 Jan 2021: grid import, on-site generation, total supply (MWh/d); influent flow statistics; effluent flow (screened and raw); measured influent and effluent quality. |
| `processed/etp_model_table.csv` | The 1,382 benchmark rows keyed by the local day their energy value describes, with benchmark columns (`bm_*`) next to the reconstructed variables. |
| `processed/holdout_predictions_{grid,total}.csv` | Observed values and all model and baseline predictions on the chronological hold-out. |
