"""Step 2 - provenance audit of the public benchmark file (Data-Melbourne_F.csv) against the
primary Melbourne Water tables. Writes tables T1-T6 and Figures 1-3."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RAW, PROC, TAB, TZ, C, SPLIT_DATE, set_style, savefig

set_style()
mt = pd.read_csv(PROC / "etp_model_table.csv", parse_dates=["date", "bm_label_date"]).set_index("date")
daily = pd.read_csv(PROC / "etp_daily_reconstructed.csv", parse_dates=["date"]).set_index("date")
n = len(mt)

# ------------------------------------------------------------ A1 target definition
d = (mt.bm_grid_MWh - mt.grid_MWh).abs()
ben = daily.loc[:mt.index.max()]
a1 = {
    "benchmark rows": n,
    "rows where benchmark 'total_grid' == feeder1 + feeder2 of the matching source record": int((d < 1e-6).sum()),
    "max abs difference in remaining rows (MWh)": float(d[d >= 1e-6].max()) if (d >= 1e-6).any() else 0.0,
    "mean grid import, benchmark dates (MWh/d)": mt.grid_MWh.mean(),
    "mean on-site generation, benchmark dates (MWh/d)": mt.gen_MWh.mean(),
    "mean total supply = grid + generation (MWh/d)": mt.total_MWh.mean(),
    "generation share of total supply, mean (%)": 100 * (mt.gen_MWh / mt.total_MWh).mean(),
    "Pearson r(grid, generation), daily": mt.grid_MWh.corr(mt.gen_MWh),
    "Pearson r(grid, total), daily": mt.grid_MWh.corr(mt.total_MWh),
    "share of var(grid) from var(generation) [var(gen)/var(grid)]": mt.gen_MWh.var() / mt.grid_MWh.var(),
    "days with generation < 50 MWh (all 2014-2021 days)": int((daily.gen_MWh < 50).sum()),
}
pd.Series(a1, name="value").round(4).to_csv(TAB / "T1_target_definition.csv", header=True)
yr = daily.assign(year=daily.index.year).groupby("year")[["grid_MWh", "gen_MWh", "total_MWh"]].mean()
yr["gen_share_%"] = 100 * yr.gen_MWh / yr.total_MWh
yr["n_days"] = daily.groupby(daily.index.year).size()
yr.round(2).to_csv(TAB / "T1b_energy_by_year.csv")

# ------------------------------------------------------------ A2 flow columns are mis-dated
def utc_daily(fn, col):
    h = pd.read_csv(RAW / fn)
    h["ts"] = pd.to_datetime(h.start_epoch_hour * 3600, unit="s", utc=True)
    return h.groupby(h.ts.dt.tz_localize(None).dt.normalize())[col].mean()

rows = []
src = {"inflow": (utc_daily("MWC_ETP_Hourly_InfluentFlow_From2009_compact.csv", "ETP_Inflow_kL_s_1"), "bm_qin", "qin_mean"),
       "outflow": (utc_daily("MWC_ETP_Hourly_EffluentFlow_From2009_compact.csv", "ETP_Outflow_kL_s_1"), "bm_qout", "qout_mean")}
aligned = {}
for name, (s, bcol, rcol) in src.items():
    b = mt[bcol].values
    lab_d = pd.DatetimeIndex(mt.bm_label_date)
    same_utc = s.reindex(lab_d).values                   # UTC-day mean on the benchmark's date label
    same_lab = daily[rcol].reindex(lab_d).values         # local-day mean on the benchmark's date label
    same_loc = mt[rcol].values                           # local-day mean on the day the energy record covers
    idx = s.values[:n]                                   # i-th benchmark row <-> i-th source day from 2008-12-31
    aligned[name] = (idx, s.index[:n])
    def r(x, y):
        ok = ~np.isnan(x) & ~np.isnan(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())
    for lab, x in [("label date, UTC-day mean", same_utc), ("label date, local-day mean", same_lab),
                   ("energy day (label + 1 d), local-day mean", same_loc),
                   ("row position: i-th row <-> i-th source day from 2008-12-31", idx)]:
        rr, k = r(b, x)
        ok = ~np.isnan(x)
        rows.append({"variable": name, "benchmark value compared with": lab, "n": k, "pearson_r": rr,
                     "median_abs_diff_kL_s": float(np.nanmedian(np.abs(b - x))),
                     "rows_equal_to_3_decimals": int((np.abs(b[ok] - x[ok]) < 1.5e-3).sum())})
    for a, z in [(0, 350), (350, 700), (700, 1050), (1050, n)]:
        rr, k = r(b[a:z], idx[a:z])
        rows.append({"variable": name, "benchmark value compared with": f"row position, rows {a}-{z - 1}", "n": k, "pearson_r": rr,
                     "median_abs_diff_kL_s": float(np.nanmedian(np.abs(b[a:z] - idx[a:z]))),
                     "rows_equal_to_3_decimals": int((np.abs(b[a:z] - idx[a:z]) < 1.5e-3).sum())})
# ---- exact reconstruction rule: mean of the hourly records of the i-th UTC day counted from 31 Dec 2008,
#      leaving out the record stamped 00:00 UTC (duplicated hour stamps are NOT removed)
recon, rec_rows = {}, []
for name, fn, col, bcol in [("inflow", "MWC_ETP_Hourly_InfluentFlow_From2009_compact.csv", "ETP_Inflow_kL_s_1", "bm_qin"),
                            ("outflow", "MWC_ETP_Hourly_EffluentFlow_From2009_compact.csv", "ETP_Outflow_kL_s_1", "bm_qout")]:
    h = pd.read_csv(RAW / fn)
    h["ts"] = pd.to_datetime(h.start_epoch_hour * 3600, unit="s", utc=True)
    h = h[h.ts.dt.hour != 0]
    s_ = h.groupby(h.ts.dt.tz_localize(None).dt.normalize())[col].mean()
    b = mt[bcol].values; sv = s_.values
    by_pos = int((np.abs(sv[:n] - b) < 5.1e-4).sum())
    i = j = hit = 0; skipped = []; x = np.full(n, np.nan)          # ordered alignment that may skip source days
    while i < n and j < len(sv):
        if abs(sv[j] - b[i]) < 5.1e-4:
            x[i] = sv[j]; hit += 1; i += 1; j += 1
        else:
            k = next((k for k in range(1, 6) if j + k < len(sv) and abs(sv[j + k] - b[i]) < 5.1e-4), None)
            if k is None:
                x[i] = sv[j]; i += 1; j += 1
            else:
                skipped += [str(dd.date()) for dd in s_.index[j:j + k]]; j += k
    recon[name] = x
    rec_rows.append({"variable": name, "benchmark_rows": n, "equal_to_3_decimals_by_row_position": by_pos,
                     "equal_to_3_decimals_in_order_allowing_skipped_source_days": hit, "source_days_skipped": "; ".join(skipped) or "none",
                     "first_source_day": str(s_.index[0].date()), "last_source_day_used": str(s_.index[j - 1].date()),
                     "pearson_r_in_order": float(np.corrcoef(x, b)[0, 1])})
pd.DataFrame(rec_rows).round(6).to_csv(TAB / "T2b_flow_exact_reconstruction.csv", index=False)
t2 = pd.DataFrame(rows)
t2.round(4).to_csv(TAB / "T2_flow_misdating.csv", index=False)
src_span = (aligned["inflow"][1][0].date(), aligned["inflow"][1][-1].date())

# ------------------------------------------------------------ A3 silently imputed quality values
rows = []
for lab, b, s in [("NH3-N", "bm_nh3", "nh3_in"), ("BOD", "bm_bod", "bod_in"), ("COD", "bm_cod", "cod_in"), ("TN", "bm_tn", "tn_in")]:
    meas = mt[s].notna()
    rows.append({"variable": lab, "rows": n, "measured_in_source": int(meas.sum()),
                 "benchmark_equals_source_where_measured": int((mt.loc[meas, b] == mt.loc[meas, s]).sum()),
                 "not_measured_but_filled_in_benchmark": int((~meas).sum()),
                 "filled_share_%": round(100 * (~meas).mean(), 1),
                 "filled_rows_test_period_%": round(100 * (~meas)[mt.index > SPLIT_DATE].mean(), 1)})
t3 = pd.DataFrame(rows)
miss = mt.bod_in.isna()
t3["note"] = ["", f"all {int((mt.loc[miss, 'bm_bod'] == mt.loc[miss, 'bm_cod'] / 2).sum())} filled values equal COD/2 exactly", "", "fill rule not documented"]
t3.to_csv(TAB / "T3_imputed_quality_values.csv", index=False)

# ------------------------------------------------------------ A4 weather-field defects
hw = mt.set_index("bm_label_date").loc["2014-01-14":"2014-01-17", "TM"]
t4 = pd.Series({
    "SLP (sea-level pressure): rows equal to 0": int((mt.bm_SLP == 0).sum()),
    "SLP: non-zero rows": int((mt.bm_SLP != 0).sum()),
    "VV (visibility, km): max": float(mt.bm_VV.max()),
    "VV: rows > 50 km": int((mt.bm_VV > 50).sum()),
    "VG (max gust): rows equal to 0": int((mt.bm_VG == 0).sum()),
    "VG: zero share (%)": round(100 * (mt.bm_VG == 0).mean(), 1),
    "TM on 14-17 Jan 2014 heatwave days in file (deg C)": "; ".join(f"{k.date()}={v}" for k, v in hw.items()),
    "mean T Dec-Feb (deg C)": round(mt.loc[mt.bm_label_date.dt.month.isin([12, 1, 2]), "T"].mean(), 2),
    "mean T Jun-Aug (deg C)": round(mt.loc[mt.bm_label_date.dt.month.isin([6, 7, 8]), "T"].mean(), 2),
}, name="value")
t4.to_csv(TAB / "T4_weather_field_checks.csv", header=True)

# ------------------------------------------------------------ A5 which 24 h does an electricity record cover?
el = pd.read_csv(RAW / "MWC_ETP_Daily_ElectricityUse_From2014.csv")
el["t"] = pd.to_datetime(el.recorddate, format="%d/%m/%Y %H:%M").dt.tz_localize("UTC")
el["grid"] = el.feeder1_kwh + el.feeder2_kwh
el["total"] = el.grid + el.Generation_kWh
rows = []
for flow, fn, col in [("inflow", "MWC_ETP_Hourly_InfluentFlow_From2009_compact.csv", "ETP_Inflow_kL_s_1"),
                      ("outflow", "MWC_ETP_Hourly_EffluentFlow_From2009_compact.csv", "ETP_Outflow_kL_s_1")]:
    hq = pd.read_csv(RAW / fn)
    hq["ts"] = pd.to_datetime(hq.start_epoch_hour * 3600, unit="s", utc=True)
    roll = hq.groupby("ts")[col].mean().asfreq("h").rolling(24, min_periods=20).mean()
    for off in range(-48, 37, 3):                               # 24-h flow window starts 'off' hours after the stamp
        q = roll.reindex(el.t + pd.Timedelta(hours=off + 23)).values
        for tgt in ("total", "grid"):
            y = el[tgt].values; ok = ~np.isnan(q); dq, dy = np.diff(q), np.diff(y); k = ~np.isnan(dq)
            rows.append({"flow": flow, "target": tgt, "window_start_offset_h": off, "n": int(ok.sum()),
                         "pearson_r_levels": np.corrcoef(q[ok], y[ok])[0, 1],
                         "pearson_r_first_differences": np.corrcoef(dq[k], dy[k])[0, 1]})
t5 = pd.DataFrame(rows)
t5.round(4).to_csv(TAB / "T5_electricity_day_alignment.csv", index=False)
fig, ax = plt.subplots(figsize=(3.8, 2.5))
for flow, c in [("outflow", C["black"]), ("inflow", C["blue"])]:
    x = t5[(t5.target == "total") & (t5.flow == flow)]
    ax.plot(x.window_start_offset_h, x.pearson_r_levels, marker="o", ms=2.5, lw=1, color=c, label=flow)
ax.axvline(0, color=C["red"], lw=0.7, ls="--"); ax.axvline(-24, color="0.5", lw=0.7, ls=":")
ax.set(xlabel="Start of 24-h flow window relative to record stamp (h)", ylabel="r with total electricity supply")
ax.legend(frameon=False, title="hourly flow series"); savefig(fig, "FigS1_electricity_day_alignment"); plt.close(fig)

# ------------------------------------------------------------ A6 row gaps (what a row-wise 'lag 1' really is)
gap = mt.index.to_series().diff().dt.days.dropna()
t6 = gap.value_counts().sort_index().rename("rows").to_frame()
t6["share_%"] = (100 * t6.rows / len(gap)).round(1)
t6.index.name = "days_since_previous_row"
t6.to_csv(TAB / "T6_row_gaps.csv")

# ------------------------------------------------------------ Figure 1: grid import vs total supply
fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for col, lab, c in [("total_MWh", "Total electricity supply (grid + on-site generation)", C["black"]),
                    ("grid_MWh", "Grid import (benchmark target 'total_grid')", C["blue"]),
                    ("gen_MWh", "On-site generation", C["orange"])]:
    ax[0].plot(daily.index, daily[col], lw=0.3, color=c, alpha=0.35)
    ax[0].plot(daily.index, daily[col].rolling(30, center=True, min_periods=15).mean(), lw=1.4, color=c, label=lab)
ax[0].axvspan(mt.index.max(), daily.index.max(), color="0.9", zorder=0)
ax[0].axvline(SPLIT_DATE, color=C["red"], ls="--", lw=0.8)
ax[0].text(SPLIT_DATE, 40, " train | test", color=C["red"], va="bottom", fontsize=7)
ax[0].text(mt.index.max(), 660, " beyond benchmark file", fontsize=7, color="0.35")
ax[0].set_ylabel("MWh d$^{-1}$"); ax[0].set_ylim(0, 700); ax[0].legend(loc="upper left", frameon=False, ncol=1, fontsize=7)
share = 100 * daily.gen_MWh / daily.total_MWh
ax[1].plot(daily.index, share, lw=0.3, color=C["orange"], alpha=0.4)
ax[1].plot(daily.index, share.rolling(30, center=True, min_periods=15).mean(), lw=1.4, color=C["orange"])
ax[1].axvspan(mt.index.max(), daily.index.max(), color="0.9", zorder=0)
ax[1].set_ylabel("Generation share\nof supply (%)"); ax[1].set_ylim(0, 60)
fig.align_ylabels(); savefig(fig, "Fig1_grid_vs_total_supply"); plt.close(fig)

# ------------------------------------------------------------ Figure 2: flow mis-dating
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
ok = mt.qin_mean.notna()
r_same = np.corrcoef(mt.bm_qin[ok], mt.qin_mean[ok])[0, 1]
n_exact = int((np.abs(recon["inflow"] - mt.bm_qin.values) < 5.1e-4).sum())
lim = (2, 9)
ax[0].scatter(mt.qin_mean, mt.bm_qin, s=4, color=C["blue"], alpha=0.5, lw=0)
ax[0].set(xlabel="Source inflow, same day (kL s$^{-1}$)", ylabel="Benchmark 'avg_inflow' (kL s$^{-1}$)", xlim=lim, ylim=lim,
          title=f"(a) same day, r = {r_same:.2f}")
ax[1].scatter(recon["inflow"], mt.bm_qin, s=4, color=C["red"], alpha=0.5, lw=0)
ax[1].set(xlabel="Source inflow, i-th day from\n31 Dec 2008 (kL s$^{-1}$)", xlim=lim, ylim=lim, title=f"(b) row position: {n_exact:,}/{n:,} equal")
for a in ax[:2]:
    a.plot(lim, lim, color="0.4", lw=0.6, ls=":")
w = slice(0, 120)
ax[2].plot(range(120), mt.bm_qin.values[w], color=C["black"], lw=1.0, label="benchmark, rows 1-120\n(labelled Jan-Jun 2014)")
ax[2].plot(range(120), recon["inflow"][w], color=C["red"], lw=0.8, ls="--", label="source, 31 Dec 2008 +")
ax[2].plot(range(120), mt.qin_mean.values[w], color=C["blue"], lw=0.8, label="source, labelled dates")
ax[2].set(xlabel="Row number", title="(c) first 120 rows", ylim=(2.2, 8)); ax[2].legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(); savefig(fig, "Fig2_flow_misdating"); plt.close(fig)

# ------------------------------------------------------------ Figure 3: measured vs filled quality values
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.barh(t3.variable, t3.measured_in_source, color=C["green"], label="measured in source")
ax.barh(t3.variable, t3.not_measured_but_filled_in_benchmark, left=t3.measured_in_source, color=C["red"], label="filled in benchmark")
for i, r_ in t3.iterrows():
    ax.text(n + 15, i, f"{r_['filled_share_%']:.0f}% filled", va="center", fontsize=7)
ax.set(xlabel="Benchmark rows", xlim=(0, n * 1.3)); ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.grid(axis="y", visible=False); savefig(fig, "Fig3_filled_quality_values"); plt.close(fig)

for name in ["T1_target_definition", "T1b_energy_by_year", "T2_flow_misdating", "T3_imputed_quality_values", "T4_weather_field_checks",
             "T5_electricity_day_alignment", "T6_row_gaps"]:
    print("\n==", name); print(pd.read_csv(TAB / f"{name}.csv").to_string(index=False))
print("\nsource days matched by row position span:", src_span)
