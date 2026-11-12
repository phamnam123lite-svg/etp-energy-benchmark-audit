"""Step 7 - out-of-time check on the 19 months of electricity data that are NOT in the benchmark
file (29 Jun 2019 - 31 Jan 2021; only electricity is published for this period). Univariate
lag models are trained on days up to 29 Jan 2018 and evaluated on two later windows. Writes T14."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from common import TAB, PROC, SEED, SPLIT_DATE, metrics, dm_test

d = pd.read_csv(PROC / "etp_daily_reconstructed.csv", parse_dates=["date"]).set_index("date")
END_BM = pd.Timestamp("2019-06-28")                            # last energy day in the benchmark file
rows = []
for tname, col in [("grid", "grid_MWh"), ("total", "total_MWh"), ("generation", "gen_MWh")]:
    y = d[col]
    X = pd.DataFrame({"lag1": y.shift(1), "lag2": y.shift(2), "lag7": y.shift(7), "roll7": y.shift(1).rolling(7).mean(),
                      "roll28": y.shift(1).rolling(28).mean()})
    ang = 2 * np.pi * (X.index.dayofyear - 1) / 365.25
    X["sin"], X["cos"] = np.sin(ang), np.cos(ang)
    ok = X.notna().all(axis=1)
    tr = ok & (X.index <= SPLIT_DATE + pd.Timedelta(days=1))
    k = int(0.8 * tr.sum()); fit_idx, cal_idx = np.where(tr)[0][:k], np.where(tr)[0][k:]
    models = {"Ridge (lags)": RidgeCV(alphas=np.logspace(-2, 3, 16)),
              "HGB (lags)": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=8, min_samples_leaf=20,
                                                          early_stopping=False, random_state=SEED)}
    fitted = {m: mod.fit(X.values[fit_idx], y.values[fit_idx]) for m, mod in models.items()}
    for wname, w in [("in-benchmark test window (30 Jan 2018 - 28 Jun 2019)", (X.index > SPLIT_DATE + pd.Timedelta(days=1)) & (X.index <= END_BM)),
                     ("post-benchmark window (29 Jun 2019 - 31 Jan 2021)", X.index > END_BM)]:
        yt = y[w].values; pers = X.lag1[w].values
        for name, f in [("Persistence (previous day)", pers), ("Mean of previous 7 d", X.roll7[w].values)] + \
                       [(m, mod.predict(X.values[w])) for m, mod in fitted.items()]:
            r = {"target": tname, "window": wname, "model": name, **metrics(yt, f)}
            r["skill_vs_persistence_%"] = 100 * (1 - r["RMSE"] / metrics(yt, pers)["RMSE"])
            if name != "Persistence (previous day)":
                r["DM_p_vs_persistence"] = dm_test(yt, f, pers)[1]
            if name in fitted:                                  # split-conformal 90 % interval, calibrated before the split date
                q = np.quantile(np.abs(y.values[cal_idx] - fitted[name].predict(X.values[cal_idx])), 0.9, method="higher")
                r["conformal90_coverage_%"] = 100 * np.mean(np.abs(yt - f) <= q); r["conformal90_width"] = 2 * q
            rows.append(r)
t14 = pd.DataFrame(rows); t14.round(3).to_csv(TAB / "T14_out_of_time.csv", index=False)
print(t14.drop(columns=["bias"]).round(2).to_string(index=False))
ac = {c: [d[c].autocorr(l) for l in (1, 2, 7, 30)] for c in ("grid_MWh", "total_MWh", "gen_MWh")}
pd.DataFrame(ac, index=["lag1", "lag2", "lag7", "lag30"]).round(3).to_csv(TAB / "T14b_autocorrelation.csv")
print(pd.DataFrame(ac, index=["lag1", "lag2", "lag7", "lag30"]).round(3))
