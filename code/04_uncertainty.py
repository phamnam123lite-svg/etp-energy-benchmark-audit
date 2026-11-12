"""Step 4 - prediction intervals on the chronological hold-out (nominal coverage 90 %).
Gaussian residual band, split conformal, conformalized quantile regression (CQR) and
adaptive conformal inference (ACI). Writes T10 and Figure 5."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from common import TAB, SEED, C, set_style, savefig
from features import FEATURES, TARGETS, load, chrono_split, make_model, inner_time

set_style()
ALPHA = 0.10


def score(y, lo, hi, a=ALPHA):
    cov = (y >= lo) & (y <= hi)
    w = hi - lo
    winkler = w + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)
    h = len(y) // 2
    return {"coverage_%": 100 * cov.mean(), "mean_width_MWh": w.mean(), "interval_score": winkler.mean(),
            "coverage_first_half_%": 100 * cov[:h].mean(), "coverage_second_half_%": 100 * cov[h:].mean()}


def cq(scores, a):
    n = len(scores)
    return np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - a)) / n), method="higher")


rows, keep = [], {}
for target in TARGETS:
    df, _ = load(target)
    tr, te = chrono_split(df)
    k = int(0.8 * len(tr)); fit, cal = tr[:k], tr[k:]            # proper training set | calibration set (most recent 20 %)
    for fs in ("RC", "RC_dyn"):
        X, y = df[FEATURES[fs]].values, df.y.values
        gs = make_model("HGB", inner_time()).fit(X[fit], y[fit]); bp = gs.best_params_
        mu_cal, mu_te = gs.predict(X[cal]), gs.predict(X[te])
        # (a) Gaussian band from in-sample residuals of a model fitted on all training data
        full = HistGradientBoostingRegressor(max_iter=200, early_stopping=False, random_state=SEED, **bp).fit(X[tr], y[tr])
        z = 1.6449 * np.std(y[tr] - full.predict(X[tr])); mf = full.predict(X[te])
        rows.append({"target": target, "features": fs, "method": "Gaussian band, in-sample residuals", **score(y[te], mf - z, mf + z)})
        # (b) split conformal
        q = cq(np.abs(y[cal] - mu_cal), ALPHA)
        rows.append({"target": target, "features": fs, "method": "Split conformal", **score(y[te], mu_te - q, mu_te + q)})
        # (c) CQR
        ql = HistGradientBoostingRegressor(loss="quantile", quantile=ALPHA / 2, max_iter=200, early_stopping=False, random_state=SEED, **bp).fit(X[fit], y[fit])
        qh = HistGradientBoostingRegressor(loss="quantile", quantile=1 - ALPHA / 2, max_iter=200, early_stopping=False, random_state=SEED, **bp).fit(X[fit], y[fit])
        e = cq(np.maximum(ql.predict(X[cal]) - y[cal], y[cal] - qh.predict(X[cal])), ALPHA)
        lo_c, hi_c = ql.predict(X[te]) - e, qh.predict(X[te]) + e
        rows.append({"target": target, "features": fs, "method": "CQR", **score(y[te], lo_c, hi_c)})
        # (d) ACI: online update of the miscoverage level, residual pool grows as test days are revealed
        gamma, a_t, pool = 0.02, ALPHA, list(np.abs(y[cal] - mu_cal)); lo_a, hi_a = [], []
        for i in range(len(te)):
            a_use = min(max(a_t, 0.001), 0.999)
            qa = np.quantile(pool[-250:], 1 - a_use, method="higher")
            lo_a.append(mu_te[i] - qa); hi_a.append(mu_te[i] + qa)
            err = float(not (lo_a[-1] <= y[te][i] <= hi_a[-1]))
            a_t += gamma * (ALPHA - err); pool.append(abs(y[te][i] - mu_te[i]))
        lo_a, hi_a = np.array(lo_a), np.array(hi_a)
        rows.append({"target": target, "features": fs, "method": "Adaptive conformal (ACI)", **score(y[te], lo_a, hi_a)})
        keep[(target, fs)] = dict(idx=df.index[te], y=y[te], mu=mu_te, sc=(mu_te - q, mu_te + q), aci=(lo_a, hi_a), z=(mf - z, mf + z))
        rows[-1]["n_fit"], rows[-1]["n_cal"], rows[-1]["n_test"] = len(fit), len(cal), len(te)
t10 = pd.DataFrame(rows)
t10.round(3).to_csv(TAB / "T10_prediction_intervals.csv", index=False)
print(t10.round(2).to_string(index=False))

# ---- Figure 5
k = keep[("total", "RC_dyn")]
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [1.7, 1]}, sharex=True)
ax[0].fill_between(k["idx"], *k["aci"], color=C["sky"], alpha=0.45, lw=0, label="90 % interval, adaptive conformal")
ax[0].plot(k["idx"], k["mu"], color=C["blue"], lw=0.8, label="prediction (HGB, reconstructed + lags)")
ax[0].plot(k["idx"], k["y"], ".", color=C["black"], ms=2.2, label="observed")
ax[0].set_ylabel("Total supply (MWh d$^{-1}$)"); ax[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
for key, lab, c in [("z", "Gaussian band", C["red"]), ("sc", "split conformal", C["orange"]), ("aci", "adaptive conformal", C["blue"])]:
    cov = pd.Series(((k["y"] >= k[key][0]) & (k["y"] <= k[key][1])).astype(float), index=k["idx"])
    ax[1].plot(cov.index, 100 * cov.rolling(60, min_periods=30).mean(), color=c, lw=1.1, label=lab)
ax[1].axhline(90, color="0.3", ls=":", lw=0.8); ax[1].set_ylim(40, 101)
ax[1].set_ylabel("Coverage, trailing\n60 rows (%)"); ax[1].legend(frameon=False, ncol=3, fontsize=7, loc="lower left")
fig.align_ylabels(); savefig(fig, "Fig5_prediction_intervals"); plt.close(fig)
