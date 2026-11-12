"""Step 6 - controlled experiment on interpolation-based augmentation.
Earlier work on this file enlarged the 1,382 rows ten-fold by cubic-spline interpolation BEFORE
the chronological 75/25 split and then used lagged energy as an input. Here the same order of
operations is emulated with the benchmark columns, and compared with (i) no augmentation and
(ii) augmentation of the training period only. Writes T13."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from sklearn.ensemble import HistGradientBoostingRegressor
from common import TAB, SEED, metrics
from features import FEATURES, load, chrono_split

K = 10                                                         # ten-fold enlargement
df, _ = load("grid")
cols = FEATURES["BM"]; raw = df[cols + ["y"]].copy()
n = len(raw); t = np.arange(n, dtype=float)
tr, te = chrono_split(df); n_tr = len(tr)


def upsample(block):
    tt = np.arange(len(block), dtype=float)
    fine = np.arange(0, len(block) - 1 + 1e-9, 1.0 / K)
    out = pd.DataFrame({c: CubicSpline(tt, block[c].values)(fine) for c in block.columns})
    out["is_original"] = np.isclose(fine, np.round(fine))
    out["t"] = fine
    return out


def fit_eval(train, test, label, lag=True):
    f = cols + (["lag"] if lag else [])
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=8, early_stopping=False, random_state=SEED)
    m.fit(train[f].values, train.y.values)
    p = m.predict(test[f].values)
    r = {"scenario": label, "lagged_energy_input": lag, "n_train": len(train), "n_test": len(test), **metrics(test.y.values, p)}
    r["persistence_MAPE_%"] = metrics(test.y.values, test.lag.values)["MAPE_%"]
    r["persistence_RMSE"] = metrics(test.y.values, test.lag.values)["RMSE"]
    o = test.is_original.values
    r["MAPE_%_original_rows_only"] = metrics(test.y.values[o], p[o])["MAPE_%"]
    r["RMSE_original_rows_only"] = metrics(test.y.values[o], p[o])["RMSE"]
    return r


rows = []
# S0: no augmentation
a = raw.copy(); a["lag"] = a.y.shift(1); a["is_original"] = True; a = a.dropna(subset=["lag"])
for lag in (False, True):
    rows.append(fit_eval(a.iloc[:n_tr - 1], a.iloc[n_tr - 1:], "S0 no augmentation, chronological split", lag))
# S1: augment the whole series, THEN split chronologically; lag = previous (augmented) row
b = upsample(raw); b["lag"] = b.y.shift(1); b = b.dropna(subset=["lag"])
cut = (n_tr - 1) + 0.5 / K                                     # same calendar boundary as S0
for lag in (False, True):
    rows.append(fit_eval(b[b.t < cut], b[b.t >= cut], "S1 augment all rows, then split (order used in earlier work)", lag))
# S2: split first, augment the training period only, evaluate on the untouched test rows
c_tr = upsample(raw.iloc[:n_tr]); c_tr["lag"] = c_tr.y.shift(1); c_tr = c_tr.dropna(subset=["lag"])
for lag in (False, True):
    rows.append(fit_eval(c_tr, a.iloc[n_tr - 1:], "S2 split, then augment training rows only", lag))
t13 = pd.DataFrame(rows)
t13.round(3).to_csv(TAB / "T13_augmentation_leakage.csv", index=False)
print(t13.round(3).to_string(index=False))
