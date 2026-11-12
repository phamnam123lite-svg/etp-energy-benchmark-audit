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


def dm_test(y, f1, f2, h=1):
    """Diebold-Mariano test on squared-error loss with the Harvey-Leybourne-Newbold
    small-sample correction. Negative statistic => forecast 1 has lower loss than forecast 2.
    Returns (statistic, two-sided p-value from Student-t with n-1 d.o.f.)."""
    from scipy import stats
    y, f1, f2 = (np.asarray(a, float) for a in (y, f1, f2))
    d = (y - f1) ** 2 - (y - f2) ** 2
    n = len(d); dbar = d.mean()
    # Newey-West long-run variance; truncation lag grows with n so that serial
    # correlation in the loss differential is not ignored for one-step forecasts
    L = max(h - 1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    dc = d - dbar
    lrv = np.sum(dc * dc) / n
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        lrv += 2 * w * np.sum(dc[k:] * dc[:-k]) / n
    if lrv <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(lrv / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)


def block_bootstrap_ci(y, f1, f2=None, stat="RMSE", block=14, B=2000, seed=SEED, alpha=0.05):
    """Moving-block bootstrap CI for a metric of f1 (or for metric(f1) - metric(f2))."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); f1 = np.asarray(f1, float)
    n = len(y); nb = int(np.ceil(n / block))
    starts_max = n - block + 1

    def m(idx, f):
        e = y[idx] - f[idx]
        return np.sqrt(np.mean(e ** 2)) if stat == "RMSE" else np.mean(np.abs(e))

    out = np.empty(B)
    for b in range(B):
        s = rng.integers(0, starts_max, nb)
        idx = (s[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[b] = m(idx, f1) - (m(idx, np.asarray(f2, float)) if f2 is not None else 0.0)
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


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
