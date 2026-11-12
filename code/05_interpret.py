"""Step 5 - what drives the two targets? Permutation importance on the hold-out, partial
dependence on inflow, a two-parameter engineering baseline and specific-energy indicators.
Writes T11-T12 and Figure 6."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, partial_dependence
from common import TAB, PROC, SEED, C, set_style, savefig, metrics, block_bootstrap_ci
from features import FEATURES, TARGETS, load, chrono_split, make_model, inner_time

set_style()
GROUPS = {"Influent flow (mean, max, s.d.)": ["qin_mean", "qin_max", "qin_std"], "Effluent flow": ["qout_mean"],
          "Influent quality (NH3-N, COD, BOD)": ["nh3_in", "cod_in", "bod_in"],
          "Previous-day weather": ["T", "TM", "Tm", "H", "PP", "V", "VM"], "Season": ["month_sin", "month_cos"]}
rows, pdp, ols = [], {}, []
rng = np.random.default_rng(SEED)
for target in TARGETS:
    df, _ = load(target)
    tr, te = chrono_split(df)
    cols = FEATURES["RC"]; X, y = df[cols].values, df.y.values
    m = make_model("HGB", inner_time()).fit(X[tr], y[tr])
    base = metrics(y[te], m.predict(X[te]))["RMSE"]
    # grouped permutation importance: all columns of a group are permuted with the same row order
    for g, members in GROUPS.items():
        j = [cols.index(c) for c in members]; inc = []
        for _ in range(50):
            Xp = X[te].copy(); perm = rng.permutation(len(te)); Xp[:, j] = Xp[perm][:, j]
            inc.append(metrics(y[te], m.predict(Xp))["RMSE"] - base)
        rows.append({"target": target, "group": g, "RMSE_increase_mean": np.mean(inc), "RMSE_increase_sd": np.std(inc), "baseline_RMSE": base})
    pi = permutation_importance(m, X[te], y[te], n_repeats=30, random_state=SEED, scoring="neg_root_mean_squared_error")
    for c, a, s in zip(cols, pi.importances_mean, pi.importances_std):
        rows.append({"target": target, "group": f"single: {c}", "RMSE_increase_mean": a, "RMSE_increase_sd": s, "baseline_RMSE": base})
    for f in ("qout_mean", "qin_mean"):
        j = cols.index(f); ok = ~np.isnan(X[tr][:, j])
        pd_ = partial_dependence(m.best_estimator_, X[tr][ok], [j], grid_resolution=40, percentiles=(0.02, 0.98))
        pdp[(target, f)] = (pd_["grid_values"][0], pd_["average"][0])
    # engineering baseline: E = a + b * Q  (base load + flow-proportional load), least squares on the training period
    for flow, qcol in [("effluent flow", "qout_mean"), ("influent flow", "qin_mean")]:
        q_tr, q_te = 86.4 * df[qcol].values[tr], 86.4 * df[qcol].values[te]          # ML/d
        k = ~np.isnan(q_tr); A = np.c_[np.ones(k.sum()), q_tr[k]]
        coef = np.linalg.lstsq(A, y[tr][k], rcond=None)[0]
        bs = []
        for _ in range(2000):                                    # moving-block bootstrap (14 rows) for a and b
            s_ = rng.integers(0, k.sum() - 14, int(np.ceil(k.sum() / 14))); idx = (s_[:, None] + np.arange(14)).ravel()[:k.sum()]
            bs.append(np.linalg.lstsq(A[idx], y[tr][k][idx], rcond=None)[0])
        bs = np.array(bs); kt = ~np.isnan(q_te)
        mm = metrics(y[te][kt], coef[0] + coef[1] * q_te[kt])
        ols.append({"target": target, "regressor": flow, "base_load_a_MWh_d": coef[0], "a_ci_lo": np.quantile(bs[:, 0], .025),
                    "a_ci_hi": np.quantile(bs[:, 0], .975), "marginal_b_kWh_m3": coef[1], "b_ci_lo": np.quantile(bs[:, 1], .025),
                    "b_ci_hi": np.quantile(bs[:, 1], .975),
                    "base_load_share_at_mean_flow_%": 100 * coef[0] / (coef[0] + coef[1] * np.nanmean(q_tr)),
                    "train_R2": metrics(y[tr][k], A @ coef)["R2"],
                    "holdout_RMSE": mm["RMSE"], "holdout_R2": mm["R2"], "holdout_MAPE_%": mm["MAPE_%"], "n_test": mm["n"]})
t11 = pd.DataFrame(rows); t11.round(4).to_csv(TAB / "T11_permutation_importance.csv", index=False)
pd.DataFrame(ols).round(4).to_csv(TAB / "T11b_flow_baseline.csv", index=False)

# ---- specific energy
d = pd.read_csv(PROC / "etp_daily_reconstructed.csv", parse_dates=["date"]).set_index("date")
d = d[d.qin_ML_d.notna()]
d["sec_grid_kWh_m3"] = d.grid_MWh / d.qin_ML_d                   # MWh/ML == kWh/m3
d["sec_total_kWh_m3"] = d.total_MWh / d.qin_ML_d
rem = (d.cod_in - d.cod_out) * d.qin_ML_d                        # kg COD removed per day
d["total_kWh_per_kgCODrem"] = 1000 * d.total_MWh / rem.where(rem > 0)
d["grid_kWh_per_kgCODrem"] = 1000 * d.grid_MWh / rem.where(rem > 0)
g = d.groupby(d.index.year)
t12 = pd.DataFrame({"days_with_flow": g.size(), "inflow_ML_d_median": g.qin_ML_d.median(),
                    "grid_kWh_m3_median": g.sec_grid_kWh_m3.median(), "total_kWh_m3_median": g.sec_total_kWh_m3.median(),
                    "understatement_by_grid_only_%": 100 * (1 - g.sec_grid_kWh_m3.median() / g.sec_total_kWh_m3.median()),
                    "days_with_COD_in_and_out": g.total_kWh_per_kgCODrem.count(),
                    "grid_kWh_per_kgCODrem_median": g.grid_kWh_per_kgCODrem.median(),
                    "total_kWh_per_kgCODrem_median": g.total_kWh_per_kgCODrem.median()})
t12.loc["all"] = [len(d), d.qin_ML_d.median(), d.sec_grid_kWh_m3.median(), d.sec_total_kWh_m3.median(),
                  100 * (1 - d.sec_grid_kWh_m3.median() / d.sec_total_kWh_m3.median()), d.total_kWh_per_kgCODrem.count(),
                  d.grid_kWh_per_kgCODrem.median(), d.total_kWh_per_kgCODrem.median()]
t12.index.name = "year"; t12.round(3).to_csv(TAB / "T12_specific_energy.csv")

# ---- Figure 6
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={"width_ratios": [1.2, 1, 1]}, layout="constrained")
gi = t11[~t11.group.str.startswith("single")]
yy = np.arange(len(GROUPS)); h = 0.38
for i, (target, c) in enumerate([("total", C["black"]), ("grid", C["blue"])]):
    x = gi[gi.target == target].set_index("group").reindex(GROUPS.keys())
    ax[0].barh(yy + (i - 0.5) * h, x.RMSE_increase_mean, h, xerr=x.RMSE_increase_sd, color=c, label=f"{target}", error_kw={"lw": 0.6})
ax[0].set_yticks(yy); ax[0].set_yticklabels([k.replace(" (", "\n(") for k in GROUPS], fontsize=7); ax[0].invert_yaxis()
ax[0].set(xlabel="Increase in hold-out RMSE when\npermuted (MWh d$^{-1}$)", title="(a) permutation importance")
ax[0].legend(frameon=False, title="target"); ax[0].grid(axis="y", visible=False)
for target, c in [("total", C["black"]), ("grid", C["blue"])]:
    ax[1].plot(pdp[(target, "qout_mean")][0] * 86.4, pdp[(target, "qout_mean")][1], color=c, lw=1.4, label=f"{target}, effluent")
    ax[1].plot(pdp[(target, "qin_mean")][0] * 86.4, pdp[(target, "qin_mean")][1], color=c, lw=1.0, ls=":", label=f"{target}, influent")
ax[1].set(xlabel="Mean daily flow (ML d$^{-1}$)", ylabel="Partial dependence (MWh d$^{-1}$)", title="(b) response to flow")
ax[1].legend(frameon=False, fontsize=6)
yrs = [y_ for y_ in t12.index if y_ != "all"]
ax[2].plot(yrs, t12.loc[yrs, "total_kWh_m3_median"], marker="o", color=C["black"], label="total supply")
ax[2].plot(yrs, t12.loc[yrs, "grid_kWh_m3_median"], marker="s", color=C["blue"], label="grid import only")
ax[2].set(xlabel="Year", ylabel="Specific electricity (kWh m$^{-3}$)", title="(c) specific electricity use", ylim=(0, 1.3))
ax[2].legend(frameon=False, loc="lower left")
savefig(fig, "Fig6_drivers_and_specific_energy"); plt.close(fig)
print(gi.round(3).to_string(index=False)); print(pd.DataFrame(ols).round(3).T.to_string()); print(t12.round(3).to_string())
