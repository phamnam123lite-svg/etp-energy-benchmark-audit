"""Step 1 - verify raw files and rebuild a date-consistent daily table for the Eastern
Treatment Plant (ETP) from the primary Melbourne Water open-data tables.

Outputs (data/processed):
  etp_daily_reconstructed.csv : one row per local calendar day, 2014-01-01 .. 2021-01-31
  etp_model_table.csv         : the 1,382 benchmark dates with benchmark columns (bm_*)
                                side by side with the reconstructed variables
"""
import hashlib
import numpy as np
import pandas as pd
from common import RAW, PROC, TAB, TZ, RAW_SHA256

# ---------------------------------------------------------------- 0. integrity
rows = []
for fn, sha in RAW_SHA256.items():
    h = hashlib.sha256((RAW / fn).read_bytes()).hexdigest()
    rows.append({"file": fn, "bytes": (RAW / fn).stat().st_size, "sha256": h, "matches_reference": h == sha})
    assert h == sha, f"checksum mismatch for {fn}"
pd.DataFrame(rows).to_csv(TAB / "T0_raw_file_checksums.csv", index=False)

# ---------------------------------------------------------------- 1. benchmark
bm = pd.read_csv(RAW / "Data-Melbourne_F.csv")
bm["date"] = pd.to_datetime(dict(year=bm.year, month=bm.month, day=bm.day))
assert bm.date.is_unique and bm.date.is_monotonic_increasing

# ---------------------------------------------------------------- 2. electricity
# 'recorddate' is a UTC wall-clock string (13:00 or 14:00) = 00:00 local time. The record is
# taken as the total for the local day that STARTS at that stamp: the hourly-inflow window
# that correlates best with the electricity record starts at the stamp (02_audit.py, T5).
# The benchmark file instead labels each record with the UTC calendar date (one day earlier).
el = pd.read_csv(RAW / "MWC_ETP_Daily_ElectricityUse_From2014.csv")
ts = pd.to_datetime(el.recorddate, format="%d/%m/%Y %H:%M")
assert (ts.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.hour == 0).all()
el["date"] = ts.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
el["grid_MWh"] = (el.feeder1_kwh + el.feeder2_kwh) / 1000.0
el["gen_MWh"] = el.Generation_kWh / 1000.0
el["total_MWh"] = el.grid_MWh + el.gen_MWh
assert el.date.is_unique
el = el.set_index("date")[["grid_MWh", "gen_MWh", "total_MWh"]].sort_index()


# ---------------------------------------------------------------- 3. hourly flows -> local days
def daily_flow(fn, col, prefix):
    h = pd.read_csv(RAW / fn)
    h["ts"] = pd.to_datetime(h.start_epoch_hour * 3600, unit="s", utc=True)
    h = h.groupby("ts", as_index=False)[col].mean()          # 11 duplicated hour stamps
    h["date"] = h.ts.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    g = h.groupby("date")[col]
    d = pd.DataFrame({f"{prefix}_mean": g.mean(), f"{prefix}_max": g.max(), f"{prefix}_min": g.min(),
                      f"{prefix}_std": g.std(), f"{prefix}_nh": g.count()})
    d.loc[d[f"{prefix}_nh"] < 20, [f"{prefix}_mean", f"{prefix}_max", f"{prefix}_min", f"{prefix}_std"]] = np.nan
    return d


qin = daily_flow("MWC_ETP_Hourly_InfluentFlow_From2009_compact.csv", "ETP_Inflow_kL_s_1", "qin")
qout = daily_flow("MWC_ETP_Hourly_EffluentFlow_From2009_compact.csv", "ETP_Outflow_kL_s_1", "qout")
qin["qin_ML_d"] = qin.qin_mean * 86.4            # kL/s -> ML/d
qin["qin_peak"] = qin.qin_max / qin.qin_mean      # daily peaking factor

# ---------------------------------------------------------------- 4. influent quality (measured only)
iq = pd.read_csv(RAW / "MWC_ETP_Daily_InfluentQuality_From2014.csv")
its = pd.to_datetime(iq.recorddate, unit="ms", utc=True)
assert (its.dt.tz_convert(TZ).dt.hour == 7).all()            # stamped 07:00 local
# Publisher metadata: flow-weighted 24-h composites collected at 07:00 that "relate to the previous 24 hours".
# The table is keyed by the local date of the stamp (the pairing used by the benchmark file); the
# alternative pairing with the day that holds 17 of the 24 composite hours is a sensitivity run (RC_qalign).
iq["date"] = its.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()   # local date of the 07:00 stamp
iq = iq.rename(columns={"Ammonia_mg_L_1": "nh3_in", "BOD_mg_L_1": "bod_in", "COD_mg_L_1": "cod_in",
                        "Nitrogentotal_mg_L_1": "tn_in"}).set_index("date")[["nh3_in", "bod_in", "cod_in", "tn_in"]]
assert iq.index.is_unique

# ---------------------------------------------------------------- 5. effluent quality
eq = pd.read_csv(RAW / "MWC_ETP_Daily_EffluentQuality_From2014.csv", dtype=str)
ets = pd.to_datetime(eq.recorddate.astype("int64"), unit="ms", utc=True)
eq["date"] = ets.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
ren = {"COD_mg_L_1": "cod_out", "BOD_mg_L_1": "bod_out", "Ammonia_mg_L_1": "nh3_out", "Nitrogentotal_mg_L_1": "tn_out"}
for c in ren:
    eq[c] = pd.to_numeric(eq[c].str.replace("<", "", regex=False).replace("NULL", np.nan), errors="coerce")
eq = eq.rename(columns=ren).groupby("date")[list(ren.values())].mean()

# ---------------------------------------------------------------- 6. assemble
cal = pd.date_range(el.index.min(), el.index.max(), name="date")
assert len(cal) == len(el)          # the electricity series has no missing day
daily = pd.DataFrame(index=cal).join(el).join(qin).join(qout[["qout_mean", "qout_nh"]]).join(iq).join(eq)
# Mass-balance screen for the effluent meter: a whole-day mean below 10 % of the same-day influent
# mean cannot be a real operating state for more than a few hours of storage (the record contains,
# e.g., 17 consecutive days of exactly 0.00 in March 2019 while inflow and electricity are normal).
# Such days are set to missing; the unscreened series is kept as qout_mean_raw for sensitivity runs.
daily["qout_mean_raw"] = daily.qout_mean
bad = daily.qout_mean.notna() & daily.qin_mean.notna() & (daily.qout_mean < 0.10 * daily.qin_mean)
daily["qout_implausible"] = bad.astype(int)
daily.loc[bad, "qout_mean"] = np.nan
pd.Series({"days with effluent and influent flow": int((daily.qout_mean_raw.notna() & daily.qin_mean.notna()).sum()),
           "days failing the mass-balance screen (effluent < 10 % of influent)": int(bad.sum()),
           "of which in the hold-out period (after 29 Jan 2018)": int(bad[daily.index > "2018-01-29"].sum()),
           "of which exactly zero": int((daily.qout_mean_raw[bad] == 0).sum())}, name="value").to_csv(TAB / "T0b_effluent_meter_screen.csv", header=True)
daily["cod_load_t_d"] = daily.cod_in * daily.qin_ML_d / 1000.0     # mg/L * ML/d = kg/d -> t/d
daily["nh3_load_t_d"] = daily.nh3_in * daily.qin_ML_d / 1000.0
daily.to_csv(PROC / "etp_daily_reconstructed.csv", float_format="%.5g")

# benchmark label D (a UTC date) -> local day X = D + 1 that its energy and quality values describe;
# the weather columns under label D are therefore the weather of the day before X
mt = bm.rename(columns={"date": "bm_label_date"})
mt["date"] = mt.bm_label_date + pd.Timedelta(days=1)
mt = mt.set_index("date")
mt = mt.rename(columns={"avg_inflow": "bm_qin", "avg_outflow": "bm_qout", "Am": "bm_nh3", "BOD": "bm_bod",
                        "COD": "bm_cod", "TN": "bm_tn", "SLP": "bm_SLP", "VV": "bm_VV", "VG": "bm_VG"})
mt["bm_grid_MWh"] = mt.total_grid / 1000.0
mt = mt.drop(columns=["total_grid", "year", "month", "day"]).join(daily)
mt.to_csv(PROC / "etp_model_table.csv", float_format="%.6g")

print("daily:", daily.shape, daily.index.min().date(), "->", daily.index.max().date())
print("model table:", mt.shape)
print(daily[["grid_MWh", "gen_MWh", "total_MWh", "qin_mean", "qin_ML_d", "qin_peak", "qout_mean", "nh3_in", "bod_in",
             "cod_in", "tn_in", "cod_out", "nh3_out"]].describe().T[["count", "mean", "std", "min", "max"]].round(2))
print("non-missing reconstructed predictors on the 1,382 benchmark dates:")
print(mt[["grid_MWh", "gen_MWh", "qin_mean", "qout_mean", "nh3_in", "bod_in", "cod_in", "tn_in", "cod_out"]].notna().sum().to_dict())
