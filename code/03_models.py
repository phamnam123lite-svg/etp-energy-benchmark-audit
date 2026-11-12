"""Step 3 - re-evaluation. (i) chronological hold-out for every target x feature set x model,
with naive baselines, skill scores, Diebold-Mariano tests and block-bootstrap intervals;
(ii) the same models under shuffled K-fold and rolling-origin validation.
Writes T7-T9 and the hold-out predictions used by later steps."""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit
from common import TAB, PROC, SEED, metrics, dm_test, block_bootstrap_ci
from features import FEATURES, TARGETS, load, chrono_split, make_model, inner_time, inner_shuffled

t0 = time.time()
rows, preds = [], {}
for target in TARGETS:
    df, y_all = load(target)
    tr, te = chrono_split(df)
    y_tr, y_te = df.y.values[tr], df.y.values[te]
    P = pd.DataFrame({"y": y_te}, index=df.index[te])
    # ---- naive references (no fitting beyond training means)
    clim = df.iloc[tr].groupby(df.index[tr].month).y.mean()
    base = {"Persistence (previous day)": df.y_lag1.values[te], "Seasonal naive (7 d earlier)": df.y_lag7.values[te],
            "Mean of previous 7 d": df.y_roll7.values[te], "Training mean": np.full(len(te), y_tr.mean()),
            "Training monthly mean": clim.reindex(df.index[te].month).values,
            "Previous row of file": df.y_prev_row.values[te]}
    for k, v in base.items():
        P[k] = v
    pers = base["Persistence (previous day)"]
    mase_den = np.mean(np.abs(y_all.diff().loc[:df.index[tr][-1]].dropna()))     # in-sample one-day naive MAE
    for fs, cols in FEATURES.items():
        for kind in ("Ridge", "RF", "HGB"):
            m = make_model(kind, inner_time()).fit(df[cols].values[tr], y_tr)
            P[f"{fs}|{kind}"] = m.predict(df[cols].values[te])
    for name in P.columns[1:]:
        f = P[name].values
        r = {"target": target, "model": name, **metrics(y_te, f)}
        r["skill_vs_persistence_%"] = 100 * (1 - r["RMSE"] / metrics(y_te, pers)["RMSE"])
        r["MASE"] = r["MAE"] / mase_den
        lo, hi = block_bootstrap_ci(y_te, f)
        r["RMSE_ci_lo"], r["RMSE_ci_hi"] = lo, hi
        if name != "Persistence (previous day)":
            r["DM_stat_vs_persistence"], r["DM_p"] = dm_test(y_te, f, pers)
            dlo, dhi = block_bootstrap_ci(y_te, f, pers)
            r["dRMSE_vs_pers_ci_lo"], r["dRMSE_vs_pers_ci_hi"] = dlo, dhi
        rows.append(r)
    preds[target] = P
    P.to_csv(PROC / f"holdout_predictions_{target}.csv", float_format="%.4f")
    print(f"[{target}] hold-out done, n_train={len(tr)}, n_test={len(te)}, {time.time() - t0:.0f}s", flush=True)
t7 = pd.DataFrame(rows)
t7.round(4).to_csv(TAB / "T7_holdout_results.csv", index=False)

# ---------------------------------------------------------------- flow-correction contrasts (paired)
rows = []
for target, P in preds.items():
    y = P.y.values
    for kind in ("Ridge", "RF", "HGB"):
        for a, b in [("BM", "BM_flowfix"), ("RC_noflow", "RC"), ("BM", "RC"), ("BM_rowlag", "RC_dyn"), ("LAG_only", "RC_dyn"),
                     ("RC_unscreened", "RC"), ("RC_dyn_unscreened", "RC_dyn"), ("RC", "RC_qalign"), ("RC_dyn", "RC_dyn_qalign")]:
            fa, fb = P[f"{a}|{kind}"].values, P[f"{b}|{kind}"].values
            lo, hi = block_bootstrap_ci(y, fb, fa)
            st, p = dm_test(y, fb, fa)
            rows.append({"target": target, "model": kind, "A": a, "B": b, "RMSE_A": metrics(y, fa)["RMSE"], "RMSE_B": metrics(y, fb)["RMSE"],
                         "dRMSE_B_minus_A": metrics(y, fb)["RMSE"] - metrics(y, fa)["RMSE"], "ci_lo": lo, "ci_hi": hi, "DM_stat": st, "DM_p": p})
pd.DataFrame(rows).round(4).to_csv(TAB / "T8_feature_set_contrasts.csv", index=False)

# ---------------------------------------------------------------- validation-protocol effect
rows = []
for target in TARGETS:
    df, _ = load(target)
    n = len(df)
    for fs in ("BM", "RC", "BM_rowlag", "RC_dyn"):
        X, y = df[FEATURES[fs]].values, df.y.values
        for kind in ("RF", "HGB"):
            for proto, outer, inner in [("shuffled 5-fold CV", KFold(5, shuffle=True, random_state=SEED), inner_shuffled()),
                                        ("rolling origin (5 expanding folds)", TimeSeriesSplit(5), inner_time())]:
                oof = np.full(n, np.nan); fold_r2 = []
                for a, b in outer.split(X):
                    m = make_model(kind, inner).fit(X[a], y[a])
                    oof[b] = m.predict(X[b]); fold_r2.append(metrics(y[b], oof[b])["R2"])
                ok = ~np.isnan(oof)
                r = {"target": target, "features": fs, "model": kind, "protocol": proto, **metrics(y[ok], oof[ok]),
                     "mean_fold_R2": float(np.mean(fold_r2)), "min_fold_R2": float(np.min(fold_r2))}
                # persistence on exactly the same evaluated rows
                r["RMSE_persistence_same_rows"] = metrics(y[ok], df.y_lag1.values[ok])["RMSE"]
                rows.append(r)
        print(f"[{target}] protocols for {fs} done, {time.time() - t0:.0f}s", flush=True)
    h = t7[(t7.target == target)]
    for fs in ("BM", "RC", "BM_rowlag", "RC_dyn"):
        for kind in ("RF", "HGB"):
            x = h[h.model == f"{fs}|{kind}"].iloc[0]
            rows.append({"target": target, "features": fs, "model": kind, "protocol": "chronological hold-out (75/25)",
                         **{k: x[k] for k in ("RMSE", "MAE", "MAPE_%", "R2", "bias", "n")},
                         "RMSE_persistence_same_rows": h[h.model == "Persistence (previous day)"].RMSE.iloc[0]})
t9 = pd.DataFrame(rows).sort_values(["target", "features", "model", "protocol"])
t9.round(4).to_csv(TAB / "T9_validation_protocols.csv", index=False)
print("total time %.0fs" % (time.time() - t0))
