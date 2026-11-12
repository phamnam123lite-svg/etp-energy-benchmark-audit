"""Feature sets, model factories and validation protocols shared by the modelling scripts."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import PROC, SEED, SPLIT_DATE

WX = ["T", "TM", "Tm", "H", "PP", "V", "VM"]                 # previous-day weather (see audit)
CAL = ["month_sin", "month_cos"]
FEATURES = {
    # the 16 predictors of the public file + season, exactly as distributed
    "BM": ["bm_qin", "bm_qout", "bm_nh3", "bm_bod", "bm_cod", "bm_tn", "bm_SLP", "bm_VV", "bm_VG"] + WX + CAL,
    # identical, except that the two flow columns are replaced by the correctly dated daily means
    "BM_flowfix": ["qin_mean", "qout_mean", "bm_nh3", "bm_bod", "bm_cod", "bm_tn", "bm_SLP", "bm_VV", "bm_VG"] + WX + CAL,
    # reconstructed: dated flows incl. within-day variability, measured quality only, defective weather fields dropped
    "RC": ["qin_mean", "qin_max", "qin_std", "qout_mean", "nh3_in", "cod_in", "bod_in"] + WX + CAL,
    "RC_noflow": ["nh3_in", "cod_in", "bod_in"] + WX + CAL,
}
LAGS = ["y_lag1", "y_lag7", "y_roll7"]                       # true calendar lags of the target
FEATURES["RC_dyn"] = FEATURES["RC"] + LAGS
FEATURES["LAG_only"] = LAGS + CAL
FEATURES["RC_unscreened"] = [c.replace("qout_mean", "qout_mean_raw") for c in FEATURES["RC"]]        # sensitivity runs
FEATURES["RC_dyn_unscreened"] = FEATURES["RC_unscreened"] + LAGS
# sensitivity: influent quality taken from the composite sample stamped 07:00 on the NEXT day, which covers
# 07:00 of the energy day to 07:00 of the following day (17 of the 24 hours of the energy day)
FEATURES["RC_qalign"] = [{"nh3_in": "nh3_next", "cod_in": "cod_next", "bod_in": "bod_next"}.get(c, c) for c in FEATURES["RC"]]
FEATURES["RC_dyn_qalign"] = FEATURES["RC_qalign"] + LAGS
FEATURES["BM_rowlag"] = FEATURES["BM"] + ["y_prev_row"]      # 'lag 1' = previous ROW of the file, as in earlier studies
TARGETS = {"grid": "grid_MWh", "total": "total_MWh"}


def load(target):
    """Model table for one target, with calendar-true and row-wise lags. Index = local energy day."""
    mt = pd.read_csv(PROC / "etp_model_table.csv", parse_dates=["date", "bm_label_date"]).set_index("date")
    daily = pd.read_csv(PROC / "etp_daily_reconstructed.csv", parse_dates=["date"]).set_index("date")
    y = daily[TARGETS[target]]
    df = mt.copy()
    df["y"] = df[TARGETS[target]]
    df["y_lag1"] = y.shift(1).reindex(df.index)
    df["y_lag7"] = y.shift(7).reindex(df.index)
    df["y_roll7"] = y.shift(1).rolling(7).mean().reindex(df.index)
    df["y_prev_row"] = df["y"].shift(1)
    for c in ("nh3", "cod", "bod"):
        df[f"{c}_next"] = daily[f"{c}_in"].shift(-1).reindex(df.index)
    ang = 2 * np.pi * (df.index.dayofyear - 1) / 365.25
    df["month_sin"], df["month_cos"] = np.sin(ang), np.cos(ang)
    df = df[df.y_roll7.notna() & df.y_prev_row.notna()]        # drops the first week of 2014
    return df, y


def chrono_split(df):
    tr = df.index <= SPLIT_DATE + pd.Timedelta(days=1)         # label date <= 28 Jan 2018
    return np.where(tr)[0], np.where(~tr)[0]


HGB_GRID = {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [4, 8, 16], "min_samples_leaf": [20], "l2_regularization": [0.0, 1.0]}


def make_model(kind, inner_cv):
    if kind == "Ridge":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 3, 16)))
    if kind == "RF":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                                   random_state=SEED, n_jobs=2))
    if kind == "HGB":   # handles missing values natively; tuned on the training data only
        base = HistGradientBoostingRegressor(max_iter=200, early_stopping=False, random_state=SEED)
        return GridSearchCV(base, HGB_GRID, cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=2)
    raise ValueError(kind)


def inner_time():
    return TimeSeriesSplit(n_splits=3)


def inner_shuffled():
    return KFold(n_splits=3, shuffle=True, random_state=SEED)
