"""Step 8 - summary figure of the re-evaluation (Figure 4) and a compact results table (T15)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import TAB, PROC, C, set_style, savefig

set_style()
t7 = pd.read_csv(TAB / "T7_holdout_results.csv"); t9 = pd.read_csv(TAB / "T9_validation_protocols.csv")
t13 = pd.read_csv(TAB / "T13_augmentation_leakage.csv")
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6))

# (a) hold-out RMSE by feature set
sets = [("BM", "BM"), ("BM_flowfix", "BM,\nflows\nre-dated"), ("RC", "RC"),
        ("LAG_only", "lags\nonly"), ("BM_rowlag", "BM +\nprev.\nrow"), ("RC_dyn", "RC +\ntrue\nlags")]
x = np.arange(len(sets)); w = 0.38
for i, (tg, c) in enumerate([("grid", C["blue"]), ("total", C["black"])]):
    h = t7[t7.target == tg].set_index("model")
    v = [h.loc[f"{s}|HGB", "RMSE"] for s, _ in sets]
    lo = [h.loc[f"{s}|HGB", "RMSE"] - h.loc[f"{s}|HGB", "RMSE_ci_lo"] for s, _ in sets]
    hi = [h.loc[f"{s}|HGB", "RMSE_ci_hi"] - h.loc[f"{s}|HGB", "RMSE"] for s, _ in sets]
    ax[0, 0].bar(x + (i - 0.5) * w, v, w, yerr=[lo, hi], color=c, label=f"{tg}", error_kw={"lw": 0.6, "capsize": 1.5})
    ax[0, 0].axhline(h.loc["Persistence (previous day)", "RMSE"], color=c, ls="--", lw=0.8)
ax[0, 0].text(len(sets) - 0.45, 41.6, "persistence", fontsize=6.5, ha="right", color="0.25")
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels([l for _, l in sets], fontsize=7)
ax[0, 0].set(ylabel="Hold-out RMSE (MWh d$^{-1}$)", title="(a) inputs (BM = benchmark file, RC = reconstructed)", ylim=(0, 62))
ax[0, 0].legend(frameon=False, ncol=2, loc="upper right", fontsize=7); ax[0, 0].grid(axis="x", visible=False)

# (b) validation protocol
protos = [("shuffled 5-fold CV", "shuffled\n5-fold"), ("rolling origin (5 expanding folds)", "rolling\norigin"), ("chronological hold-out (75/25)", "chrono.\nhold-out")]
g = t9[(t9.target == "grid") & (t9.model == "HGB")]
xx = np.arange(len(protos)); w = 0.2
for i, (fs, lab, c) in enumerate([("BM", "benchmark file", C["red"]), ("BM_rowlag", "benchmark + previous row", C["orange"]),
                                  ("RC", "reconstructed", C["sky"]), ("RC_dyn", "reconstructed + true lags", C["blue"])]):
    v = [g[(g.features == fs) & (g.protocol == p)].R2.iloc[0] for p, _ in protos]
    ax[0, 1].bar(xx + (i - 1.5) * w, v, w, color=c, label=lab)
ax[0, 1].axhline(0, color="0.2", lw=0.6); ax[0, 1].set_xticks(xx); ax[0, 1].set_xticklabels([l for _, l in protos], fontsize=7)
ax[0, 1].set(ylabel="R$^2$ (grid import)", title="(b) validation protocol, HGB", ylim=(-0.25, 1.22))
ax[0, 1].legend(frameon=False, fontsize=6, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2); ax[0, 1].grid(axis="x", visible=False)

# (c) augmentation order
s = t13[t13.lagged_energy_input].reset_index(drop=True)
labs = ["no\naugmentation", "augment all,\nthen split", "split, then\naugment train"]
xx = np.arange(3)
ax[1, 0].bar(xx - 0.2, s["MAPE_%"], 0.4, color=C["blue"], label="HGB with lagged energy")
ax[1, 0].bar(xx + 0.2, s["persistence_MAPE_%"], 0.4, color="0.6", label="copy previous row")
for i in range(3):
    ax[1, 0].text(xx[i] - 0.2, s["MAPE_%"][i] + 0.25, f"{s['MAPE_%'][i]:.1f}", ha="center", fontsize=7)
    ax[1, 0].text(xx[i] + 0.2, s["persistence_MAPE_%"][i] + 0.25, f"{s['persistence_MAPE_%'][i]:.1f}", ha="center", fontsize=7)
ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels(labs, fontsize=7)
ax[1, 0].set(ylabel="Test MAPE (%)", title="(c) ten-fold spline augmentation, grid import", ylim=(0, 15))
ax[1, 0].legend(frameon=False, fontsize=7, loc="upper center"); ax[1, 0].grid(axis="x", visible=False)

# (d) observed vs predicted, last 150 hold-out rows
P = pd.read_csv(PROC / "holdout_predictions_total.csv", parse_dates=["date"]).set_index("date").iloc[-150:]
ax[1, 1].plot(P.index, P.y, color=C["black"], lw=1.0, label="observed")
ax[1, 1].plot(P.index, P["BM|HGB"], color=C["red"], lw=0.8, label="BM")
ax[1, 1].plot(P.index, P["RC_dyn|HGB"], color=C["blue"], lw=0.8, label="RC + true lags")
ax[1, 1].set(ylabel="Total supply (MWh d$^{-1}$)", title="(d) hold-out, last 150 rows", ylim=(280, 530))
ax[1, 1].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", columnspacing=0.8, handlelength=1.2); ax[1, 1].tick_params(axis="x", labelrotation=30)
fig.tight_layout(); savefig(fig, "Fig4_reevaluation_summary"); plt.close(fig)

keep = ["Persistence (previous day)", "Mean of previous 7 d", "Training mean", "BM|HGB", "BM_flowfix|HGB", "RC_noflow|HGB", "RC|HGB", "RC|RF", "RC|Ridge",
        "RC_unscreened|HGB", "RC_dyn_unscreened|HGB", "RC_qalign|HGB", "RC_dyn_qalign|HGB",
        "LAG_only|HGB", "BM_rowlag|HGB", "RC_dyn|HGB", "RC_dyn|RF", "RC_dyn|Ridge"]
t15 = t7[t7.model.isin(keep)][["target", "model", "RMSE", "RMSE_ci_lo", "RMSE_ci_hi", "MAE", "MAPE_%", "R2", "skill_vs_persistence_%", "MASE", "DM_p"]]
t15.round(3).to_csv(TAB / "T15_main_results_compact.csv", index=False)
print(t15.round(2).to_string(index=False))
