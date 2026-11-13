"""Step 9 - same-day explanation versus day-ahead forecasting, on the COMPLETE daily calendar
(every day from 8 Jan 2014 to 28 Jun 2019, so the series is regularly spaced).

  EXPLAIN  : flows of the SAME day + target lags      (not available before the day starts)
  FORECAST : only information known at 00:00 of the day: target lags and flows of earlier days
  LAGS     : target lags only

Weather and influent quality exist only on sampling days and are not used here. Writes T16."""
import numpy as np
import pandas as pd
from common import TAB, PROC, SPLIT_DATE, metrics, dm_test, block_bootstrap_ci
from features import MODELS, make_model, inner_time

d = pd.read_csv(PROC / "etp_daily_reconstructed.csv", parse_dates=["date"]).set_index("date")
END = pd.Timestamp("2019-06-28")
rows = []
for tname, col in [("grid", "grid_MWh"), ("total", "total_MWh")]:
    y = d[col]
    X = pd.DataFrame({"y_lag1": y.shift(1), "y_lag2": y.shift(2), "y_lag7": y.shift(7), "y_roll7": y.shift(1).rolling(7).mean()})
    ang = 2 * np.pi * (X.index.dayofyear - 1) / 365.25
    X["sin"], X["cos"] = np.sin(ang), np.cos(ang)
    for c in ("qout_mean", "qin_mean", "qin_max", "qin_std"):
        X[c] = d[c]                                    # same day
        X[c + "_lag1"] = d[c].shift(1)                 # previous day
    X["qout_lag2"] = d.qout_mean.shift(2); X["qout_roll7"] = d.qout_mean.shift(1).rolling(7, min_periods=4).mean()
    lags = ["y_lag1", "y_lag2", "y_lag7", "y_roll7", "sin", "cos"]
    sets = {"LAGS": lags,
            "FORECAST (day-ahead)": lags + ["qout_mean_lag1", "qout_lag2", "qout_roll7", "qin_mean_lag1", "qin_max_lag1", "qin_std_lag1"],
            "EXPLAIN (same-day flows)": lags + ["qout_mean", "qin_mean", "qin_max", "qin_std"]}
    ok = X[lags].notna().all(axis=1) & y.notna() & (X.index <= END)
    tr = np.where(ok & (X.index <= SPLIT_DATE + pd.Timedelta(days=1)))[0]; te = np.where(ok & (X.index > SPLIT_DATE + pd.Timedelta(days=1)))[0]
    k = int(0.8 * len(tr)); fit, cal = tr[:k], tr[k:]
    yt = y.values[te]; pers = X.y_lag1.values[te]; tidx = X.index[te]
    P = {"Persistence (previous day)": pers, "Mean of previous 7 d": X.y_roll7.values[te]}
    conf = {}
    for sname, cols in sets.items():
        for kind in MODELS:
            m = make_model(kind, inner_time()).fit(X[cols].values[tr], y.values[tr]); P[f"{sname}|{kind}"] = m.predict(X[cols].values[te])
            if kind in ("Ridge", "HGB"):               # split-conformal 90 % interval from a model fitted without the calibration block
                mc = make_model(kind, inner_time()).fit(X[cols].values[fit], y.values[fit])
                q = np.quantile(np.abs(y.values[cal] - mc.predict(X[cols].values[cal])), 0.9, method="higher")
                conf[f"{sname}|{kind}"] = (100 * np.mean(np.abs(yt - mc.predict(X[cols].values[te])) <= q), 2 * q)
    for name, f in P.items():
        r = {"target": tname, "inputs": name.split("|")[0], "model": name.split("|")[1] if "|" in name else "-", **metrics(yt, f)}
        r["skill_vs_persistence_%"] = 100 * (1 - r["RMSE"] / metrics(yt, pers)["RMSE"])
        r["RMSE_ci_lo"], r["RMSE_ci_hi"] = block_bootstrap_ci(yt, f, t=tidx)
        if name != "Persistence (previous day)":
            r["DM_p_vs_persistence"] = dm_test(yt, f, pers, t=tidx)[1]
        if name in conf:
            r["conformal90_coverage_%"], r["conformal90_width_MWh"] = conf[name]
        rows.append(r)
    for kind in MODELS:                                # does same-day flow add to what is known the day before?
        st, p = dm_test(yt, P[f"EXPLAIN (same-day flows)|{kind}"], P[f"FORECAST (day-ahead)|{kind}"], t=tidx)
        rows.append({"target": tname, "inputs": "EXPLAIN minus FORECAST", "model": kind,
                     "RMSE": metrics(yt, P[f"EXPLAIN (same-day flows)|{kind}"])["RMSE"] - metrics(yt, P[f"FORECAST (day-ahead)|{kind}"])["RMSE"],
                     "DM_p_vs_persistence": p, "n": len(te)})
    print(tname, "n_train", len(tr), "n_test", len(te), flush=True)
t16 = pd.DataFrame(rows); t16.round(3).to_csv(TAB / "T16_explain_vs_forecast.csv", index=False)
print(t16[["target", "inputs", "model", "RMSE", "MAPE_%", "R2", "skill_vs_persistence_%", "DM_p_vs_persistence", "conformal90_coverage_%", "conformal90_width_MWh"]].round(2).to_string(index=False))
