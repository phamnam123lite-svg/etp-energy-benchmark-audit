"""Shared helpers for the ETP energy-benchmark audit (paths, metrics, tests, plotting style)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
for _p in (PROC, TAB, FIG):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261112
TZ = "Australia/Melbourne"
SPLIT_DATE = pd.Timestamp("2018-01-28")  # last training day, as in Bagherzadeh et al. (2021) / Alali et al. (2023)

RAW_SHA256 = {
    "Data-Melbourne_F.csv": "323dcf40bdc91ea1e8ba250aa0c3b3565b81cd914c40ad8ad36a876bdb4f5f94",
    "MWC_ETP_Daily_ElectricityUse_From2014.csv": "65969ab6f60be476e0467e204dffca6bac20aea7eccb6d35be872a18242c34e5",
    "MWC_ETP_Daily_InfluentQuality_From2014.csv": "754c7cd6fac4705a7c89d35cd8c9af6b5d25843e26ffc365faf9765c8b7f0752",
    "MWC_ETP_Daily_EffluentQuality_From2014.csv": "26680dc276c81b02410e6320dacf180319c4edfdfa115faca6ed69b31979fb32",
    "MWC_ETP_Hourly_InfluentFlow_From2009_compact.csv": "bfece373bf5091379aa04d37199d136fe7c9ece17236f47a1ddbb621b5d9ef28",
    "MWC_ETP_Hourly_EffluentFlow_From2009_compact.csv": "4e0cae76e2097aa5e78903b87701ab47f40d64d49d138ff25e7fbf1cfe5d3c86",
}


def metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    e = y - yhat
    return {
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "MAPE_%": float(100 * np.mean(np.abs(e[y != 0]) / np.abs(y[y != 0]))),   # days with y = 0 are skipped
        "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
        "bias": float(np.mean(-e)),
        "n": int(len(y)),
    }


def dm_test(y, f1, f2, t=None, h=1, bandwidth_days=14):
    """Diebold-Mariano test on squared-error loss with the Harvey-Leybourne-Newbold small-sample
    correction. Negative statistic => forecast 1 has lower loss than forecast 2.
    If `t` (dates of the observations) is given, the long-run variance uses a Bartlett kernel on the
    CALENDAR distance between observations (bandwidth in days), which is valid for irregularly spaced
    rows; otherwise a Newey-West estimator on row lags is used (regular daily series).
    Returns (statistic, two-sided p-value from Student-t with n-1 d.o.f.)."""
    from scipy import stats
    y, f1, f2 = (np.asarray(a, float) for a in (y, f1, f2))
    d = (y - f1) ** 2 - (y - f2) ** 2
    n = len(d); dc = d - d.mean()
    if t is not None:
        days = (pd.DatetimeIndex(t) - pd.DatetimeIndex(t)[0]).days.values.astype(float)
        w = np.clip(1.0 - np.abs(days[:, None] - days[None, :]) / (bandwidth_days + 1.0), 0.0, None)
        lrv = float(dc @ w @ dc) / n
    else:
        L = max(h - 1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
        lrv = np.sum(dc * dc) / n
        for k in range(1, L + 1):
            lrv += 2 * (1 - k / (L + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    if lrv <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(lrv / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


def block_bootstrap_ci(y, f1, f2=None, t=None, stat="RMSE", block=14, block_days=21, B=2000, seed=SEED, alpha=0.05):
    """Block-bootstrap CI for a metric of f1 (or for metric(f1) - metric(f2)).
    With `t`: non-overlapping CALENDAR blocks of `block_days` days are resampled (irregular rows keep
    their true spacing inside a block). Without `t`: moving blocks of `block` consecutive rows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); f1 = np.asarray(f1, float); f2 = None if f2 is None else np.asarray(f2, float)
    n = len(y)

    def m(idx, f):
        e = y[idx] - f[idx]
        return np.sqrt(np.mean(e ** 2)) if stat == "RMSE" else np.mean(np.abs(e))

    if t is not None:
        days = (pd.DatetimeIndex(t) - pd.DatetimeIndex(t)[0]).days.values
        blocks = [np.where(days // block_days == k)[0] for k in np.unique(days // block_days)]
        draw = lambda: np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
    else:
        nb = int(np.ceil(n / block))
        draw = lambda: (rng.integers(0, n - block + 1, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n]
    out = np.empty(B)
    for b in range(B):
        idx = draw()
        out[b] = m(idx, f1) - (m(idx, f2) if f2 is not None else 0.0)
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float); order = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5, "font.family": "DejaVu Sans",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


# colour-blind-safe palette (Okabe-Ito)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#666666", "black": "#000000"}


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
