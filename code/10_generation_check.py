"""Step 10 - is 'grid import + on-site generation' the plant's electricity demand?
The publisher's metadata says that the two feeders plus generation give total consumption. If that
is so (no export), three things should hold in the data and are tested here:
  (a) daily generation never exceeds daily total supply and grid import never approaches zero;
  (b) a change in generation is offset one-for-one by grid import (slope of d(grid) on d(gen) = -1);
  (c) total supply does not depend on generation once effluent flow is accounted for.
Writes T17."""
import numpy as np
import pandas as pd
from common import TAB, PROC

d = pd.read_csv(PROC / "etp_daily_reconstructed.csv", parse_dates=["date"]).set_index("date")


def ols_hac(y, X, L=7):
    """OLS with Newey-West (Bartlett, L lags) standard errors; rows must be consecutive days."""
    X = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
    Xu = X * u[:, None]; S = Xu.T @ Xu
    for k in range(1, L + 1):
        G = Xu[k:].T @ Xu[:-k]; S += (1 - k / (L + 1)) * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X); V = XtXi @ S @ XtXi
    return b, np.sqrt(np.diag(V)), 1 - u.var() / y.var()


out = {}
out["days in electricity record"] = len(d)
out["min daily grid import (MWh)"] = d.grid_MWh.min()
out["max daily generation (MWh)"] = d.gen_MWh.max()
out["min daily total supply (MWh)"] = d.total_MWh.min()
out["days with generation > 50 % of total supply"] = int((d.gen_MWh > 0.5 * d.total_MWh).sum())
out["max generation share of total supply (%)"] = 100 * (d.gen_MWh / d.total_MWh).max()
# (b) first differences, all consecutive days 2014-2021
dg, dgen = d.grid_MWh.diff().values[1:], d.gen_MWh.diff().values[1:]
b, se, r2 = ols_hac(dg, dgen[:, None])
out["(b) slope of d(grid) on d(generation), all days 2014-2021"] = b[1]
out["(b) 95 % CI low"] = b[1] - 1.96 * se[1]; out["(b) 95 % CI high"] = b[1] + 1.96 * se[1]; out["(b) n"] = len(dg)
# (b') controlling for the change in effluent flow (days with screened flow on both days)
f = d[["grid_MWh", "gen_MWh", "total_MWh", "qout_mean"]].copy(); df_ = f.diff(); ok = df_.notna().all(axis=1) & f.qout_mean.shift(1).notna()
b2, se2, _ = ols_hac(df_.grid_MWh[ok].values, np.c_[df_.gen_MWh[ok].values, 86.4 * df_.qout_mean[ok].values])
out["(b') slope on d(generation), controlling for d(effluent flow)"] = b2[1]
out["(b') 95 % CI low"] = b2[1] - 1.96 * se2[1]; out["(b') 95 % CI high"] = b2[1] + 1.96 * se2[1]; out["(b') n"] = int(ok.sum())
# (c) levels: total ~ effluent flow + generation
ok = f.notna().all(axis=1)
b3, se3, r23 = ols_hac(f.total_MWh[ok].values, np.c_[86.4 * f.qout_mean[ok].values, f.gen_MWh[ok].values])
out["(c) coefficient of generation in total ~ flow + generation"] = b3[2]
out["(c) 95 % CI low"] = b3[2] - 1.96 * se3[2]; out["(c) 95 % CI high"] = b3[2] + 1.96 * se3[2]; out["(c) n"] = int(ok.sum())
# outage days: generation below 20 MWh; compare total supply with the mean of the 7 preceding days
low = d.gen_MWh < 20
ref = d.total_MWh.where(~low).shift(1).rolling(7, min_periods=3).mean()
diff = (d.total_MWh - ref)[low].dropna(); gdiff = (d.grid_MWh - d.grid_MWh.where(~low).shift(1).rolling(7, min_periods=3).mean())[low].dropna()
out["generation-outage days (< 20 MWh)"] = int(low.sum())
out["mean change in TOTAL supply on outage days vs previous week (MWh)"] = diff.mean()
out["mean change in GRID import on outage days vs previous week (MWh)"] = gdiff.mean()
out["mean generation on ordinary days (MWh)"] = d.gen_MWh[~low].mean()
t17 = pd.Series(out, name="value"); t17.round(3).to_csv(TAB / "T17_generation_no_export_check.csv", header=True)
print(t17.round(3).to_string())
