from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import pointbiserialr
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x, y)
    if tab.empty:
        return np.nan
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    if n == 0:
        return np.nan
    r, k = tab.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return float(np.sqrt(chi2 / denom))


def build_dataset(meas_path: str, bs_path: str) -> pd.DataFrame:
    meas = pd.read_csv(meas_path)
    bs = pd.read_csv(bs_path)

    bs = bs.rename(
        columns={
            "lat": "bs_lat",
            "lon": "bs_lon",
            "dem_alt_m": "bs_dem_alt_m",
            "dem_file": "bs_dem_file",
        }
    )
    keep_bs = [
        c
        for c in [
            "cell_id",
            "bs_lat",
            "bs_lon",
            "bs_dem_alt_m",
            "bs_dem_file",
            "plmn",
            "name",
            "bs_is_5g",
            "bs_has_dss",
            "bs_band",
            "bs_sector",
            "bs_band_num",
            "bs_sector_num",
            "bs_bands_all",
            "source_file",
        ]
        if c in bs.columns
    ]
    bs = bs[keep_bs].drop_duplicates(subset=["cell_id"])

    df = meas.merge(bs, on="cell_id", how="left")
    df["dist_bs_m"] = haversine_m(
        df["lat"].to_numpy(dtype=float),
        df["lon"].to_numpy(dtype=float),
        df["bs_lat"].to_numpy(dtype=float),
        df["bs_lon"].to_numpy(dtype=float),
    )
    if "dem_alt_m" in df.columns and "bs_dem_alt_m" in df.columns:
        df["delta_alt_m"] = df["dem_alt_m"] - df["bs_dem_alt_m"]
    return df


def correlation_with_signal(df: pd.DataFrame, out_path: str) -> None:
    base_features = [
        "dist_bs_m",
        "delta_alt_m",
        "dem_alt_m",
        "bs_dem_alt_m",
        "lat",
        "lon",
        "bs_band_num",
        "bs_sector_num",
        "bs_is_5g",
        "bs_has_dss",
        "accuracy",
        "speed",
        "ta",
    ]
    features = [f for f in base_features if f in df.columns]
    rows = []
    y = pd.to_numeric(df["signal"], errors="coerce")
    for feat in features:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) < 10:
            continue
        xv = x[m].to_numpy()
        yv = y[m].to_numpy()
        pr, pp = pearsonr(xv, yv)
        sr, sp = spearmanr(xv, yv)
        rows.append(
            {
                "feature": feat,
                "n": int(m.sum()),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "abs_spearman_r": float(abs(sr)),
            }
        )
    out = pd.DataFrame(rows).sort_values("abs_spearman_r", ascending=False)
    out.to_csv(out_path, index=False)


def correlation_with_coverage(df: pd.DataFrame, threshold_dbm: float, out_num_path: str, out_cat_path: str) -> None:
    work = df.copy()
    work["covered"] = (pd.to_numeric(work["signal"], errors="coerce") >= float(threshold_dbm)).astype(int)

    num_features = [
        f
        for f in [
            "dist_bs_m",
            "delta_alt_m",
            "dem_alt_m",
            "bs_dem_alt_m",
            "lat",
            "lon",
            "bs_band_num",
            "bs_sector_num",
            "bs_is_5g",
            "bs_has_dss",
            "accuracy",
            "speed",
            "ta",
        ]
        if f in work.columns
    ]
    num_rows = []
    for feat in num_features:
        x = pd.to_numeric(work[feat], errors="coerce")
        y = work["covered"]
        m = x.notna() & y.notna()
        if int(m.sum()) < 10:
            continue
        xv = x[m].to_numpy()
        yv = y[m].to_numpy(dtype=int)
        pbr, pbp = pointbiserialr(yv, xv)
        f_val, f_p = f_classif(xv.reshape(-1, 1), yv)
        mi = mutual_info_classif(xv.reshape(-1, 1), yv, random_state=42)
        num_rows.append(
            {
                "feature": feat,
                "n": int(m.sum()),
                "pointbiserial_r": float(pbr),
                "pointbiserial_p": float(pbp),
                "anova_f": float(f_val[0]),
                "anova_p": float(f_p[0]),
                "mutual_info": float(mi[0]),
                "abs_pointbiserial_r": float(abs(pbr)),
            }
        )
    pd.DataFrame(num_rows).sort_values(["mutual_info", "abs_pointbiserial_r"], ascending=False).to_csv(out_num_path, index=False)

    cat_features = [c for c in ["mode", "plmn", "bs_band", "bs_sector", "source_file"] if c in work.columns]
    cat_rows = []
    for feat in cat_features:
        x = work[feat].fillna("NA")
        y = work["covered"]
        if x.nunique() < 2:
            continue
        tab = pd.crosstab(x, y)
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(tab, correction=False)
        cv = cramers_v(x, y)
        cat_rows.append(
            {
                "feature": feat,
                "n": int(len(work)),
                "n_levels": int(x.nunique()),
                "chi2": float(chi2),
                "chi2_p": float(p),
                "cramers_v": float(cv),
            }
        )
    pd.DataFrame(cat_rows).sort_values("cramers_v", ascending=False).to_csv(out_cat_path, index=False)


def parse_thresholds(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation/association report for engineered coverage features.")
    parser.add_argument("--meas", default="data/connectivity/derived/measurement_altitudes_dem.csv")
    parser.add_argument("--bs", default="data/connectivity/derived/bs_altitudes_dem_matched.csv")
    parser.add_argument("--thresholds", default="-110")
    parser.add_argument("--out-dir", default="results/feature_reports")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("[INIT] Loading and merging measurement + BS files...")
    df = build_dataset(args.meas, args.bs)
    print(f"[INIT] Rows: {len(df)} | rows with matched BS: {int(df['bs_lat'].notna().sum())}")

    out_signal = os.path.join(args.out_dir, "feature_corr_with_signal.csv")
    correlation_with_signal(df, out_signal)
    print(f"[SAVE] {out_signal}")

    for thr in parse_thresholds(args.thresholds):
        thr_tag = str(int(thr)) if float(thr).is_integer() else str(thr).replace(".", "p")
        out_num = os.path.join(args.out_dir, f"feature_assoc_with_coverage_{thr_tag}dbm_numeric.csv")
        out_cat = os.path.join(args.out_dir, f"feature_assoc_with_coverage_{thr_tag}dbm_categorical.csv")
        correlation_with_coverage(df, thr, out_num, out_cat)
        print(f"[SAVE] {out_num}")
        print(f"[SAVE] {out_cat}")

    print("[DONE]")


if __name__ == "__main__":
    main()
