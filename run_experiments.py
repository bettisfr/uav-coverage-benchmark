from __future__ import annotations

import argparse
import glob
import itertools
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from evaluation.metrics import BinaryMetrics
from methods.alpha_shape import AlphaShapeCoverageModel
from methods.convex_hull import ConvexHullCoverageModel
from methods.gpr_model import GPRCoverageModel
from methods.idw_model import IDWCoverageModel
from methods.kriging_model import KrigingCoverageModel
from methods.ml_classifier import MLCoverageModel

try:
    from sklearn.cluster import DBSCAN
except Exception:  # pragma: no cover
    DBSCAN = None


STD_COLS = ["lat", "lon", "cell_id", "signal", "measured_at", "mode", "source_file"]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _append_tag_to_path(path: str, tag: str | None) -> str:
    if not tag:
        return path
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(tag))
    base, ext = os.path.splitext(path)
    ext = ext or ".csv"
    return f"{base}_{safe}{ext}"


def _load_new_csv(path: str, skip_ocid: bool) -> pd.DataFrame:
    base = os.path.basename(path)
    if skip_ocid and "ocid" in base:
        return pd.DataFrame(columns=STD_COLS)

    df = pd.read_csv(path)
    required = {"lat", "lon", "cell_id", "net_type", "rssi"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=STD_COLS)

    signal = np.where(df["net_type"] == "LTE", df.get("rsrp"), df.get("rssi"))
    out = pd.DataFrame(
        {
            "lat": df.get("lat"),
            "lon": df.get("lon"),
            "cell_id": df.get("cell_id"),
            "signal": signal,
            "measured_at": df.get("measured_at"),
            "mode": "new",
            "source_file": base,
        }
    )
    return out


def _load_old_csv(path: str, mode: str) -> pd.DataFrame:
    base = os.path.basename(path)
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 8:
        return pd.DataFrame(columns=STD_COLS)

    date_str = base.replace("signal-", "").replace(".csv", "")
    measured_at = f"{date_str} 12:00:00"

    out = pd.DataFrame(
        {
            "lat": df.iloc[:, 0],
            "lon": df.iloc[:, 1],
            "cell_id": df.iloc[:, 6],
            "signal": df.iloc[:, 7],
            "measured_at": measured_at,
            "mode": mode,
            "source_file": base,
        }
    )
    return out


def load_all_data(cfg: dict) -> pd.DataFrame:
    dc = cfg["data"]
    root = dc["root_dir"]
    skip_ocid = bool(dc.get("skip_ocid", True))

    frames: List[pd.DataFrame] = []

    for path in glob.glob(os.path.join(root, "dataset", "*.csv")):
        frames.append(_load_new_csv(path, skip_ocid=skip_ocid))

    for mode in ["bike", "car", "train", "walk"]:
        pattern = os.path.join(root, "dataset-old", mode, "*.csv")
        for path in glob.glob(pattern):
            frames.append(_load_old_csv(path, mode=mode))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=STD_COLS)

    df = pd.concat(frames, ignore_index=True)
    for col in ["lat", "lon", "cell_id", "signal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep measured_at optional: old/new sources can be mixed and timezone formats may differ.
    # We still parse it (UTC) so temporal split remains available when desired.
    df["measured_at"] = pd.to_datetime(df["measured_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["lat", "lon", "cell_id", "signal"]).copy()

    df = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]
    df["cell_id"] = df["cell_id"].astype(int)
    return df


def apply_common_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pc = cfg["preprocess"]
    df = df.copy()

    min_signal = float(pc.get("min_signal_dbm", -150))
    max_signal = float(pc.get("max_signal_dbm", -30))
    df = df[(df["signal"] >= min_signal) & (df["signal"] <= max_signal)]

    counts = df.groupby("cell_id").size()
    keep_ids = counts[counts >= int(pc.get("min_total_samples_per_cell", 10))].index
    df = df[df["cell_id"].isin(keep_ids)]

    max_cells = cfg.get("experiment", {}).get("max_cells")
    if max_cells:
        top_ids = counts.sort_values(ascending=False).head(int(max_cells)).index
        df = df[df["cell_id"].isin(top_ids)]

    return df


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371000.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def enrich_with_derived_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    dc = cfg.get("data", {})
    meas_path = dc.get("measurement_features_csv", "data/connectivity/derived/measurement_altitudes_dem.csv")
    bs_path = dc.get("bs_features_csv", "data/connectivity/derived/bs_altitudes_dem_matched.csv")

    if os.path.exists(meas_path):
        meas = pd.read_csv(meas_path, usecols=lambda c: c in {"lat", "lon", "dem_alt_m"})
        meas["lat"] = pd.to_numeric(meas["lat"], errors="coerce")
        meas["lon"] = pd.to_numeric(meas["lon"], errors="coerce")
        meas["dem_alt_m"] = pd.to_numeric(meas["dem_alt_m"], errors="coerce")
        meas = meas.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"], keep="first")
        out = out.merge(meas, on=["lat", "lon"], how="left")

    if os.path.exists(bs_path):
        bs = pd.read_csv(bs_path)
        if "lat" in bs.columns and "bs_lat" not in bs.columns:
            bs = bs.rename(columns={"lat": "bs_lat"})
        if "lon" in bs.columns and "bs_lon" not in bs.columns:
            bs = bs.rename(columns={"lon": "bs_lon"})
        keep_cols = [
            c
            for c in [
                "cell_id",
                "bs_lat",
                "bs_lon",
                "dem_alt_m",
                "bs_dem_alt_m",
                "bs_is_5g",
                "bs_has_dss",
                "bs_band_num",
                "bs_sector_num",
                "plmn",
                "bs_band",
                "bs_sector",
            ]
            if c in bs.columns
        ]
        bs = bs[keep_cols].copy()
        if "dem_alt_m" in bs.columns and "bs_dem_alt_m" not in bs.columns:
            bs = bs.rename(columns={"dem_alt_m": "bs_dem_alt_m"})
        bs["cell_id"] = pd.to_numeric(bs["cell_id"], errors="coerce")
        bs = bs.dropna(subset=["cell_id"]).drop_duplicates(subset=["cell_id"], keep="first")
        bs["cell_id"] = bs["cell_id"].astype(int)
        out = out.merge(bs, on="cell_id", how="left")

    if {"lat", "lon", "bs_lat", "bs_lon"}.issubset(out.columns):
        m = out["bs_lat"].notna() & out["bs_lon"].notna()
        out.loc[m, "dist_bs_m"] = _haversine_m(
            out.loc[m, "lat"].to_numpy(dtype=float),
            out.loc[m, "lon"].to_numpy(dtype=float),
            out.loc[m, "bs_lat"].to_numpy(dtype=float),
            out.loc[m, "bs_lon"].to_numpy(dtype=float),
        )
    if {"dem_alt_m", "bs_dem_alt_m"}.issubset(out.columns):
        out["delta_alt_m"] = out["dem_alt_m"] - out["bs_dem_alt_m"]
    return out


def split_data(df: pd.DataFrame, cfg: dict):
    if df.empty:
        return df, df, {"strategy": "none", "split_point": None}

    scfg = cfg.get("split", {})
    strategy = str(scfg.get("strategy", "random")).lower()
    test_fraction = float(scfg.get("test_fraction", scfg.get("temporal_test_fraction", 0.3)))
    test_fraction = min(max(test_fraction, 0.01), 0.99)

    if strategy == "temporal":
        if "measured_at" not in df.columns or df["measured_at"].notna().sum() == 0:
            # Fallback: if no valid timestamps exist, use deterministic random split.
            strategy = "random"
        else:
            split_time = df["measured_at"].quantile(1 - test_fraction)
            train_df = df[df["measured_at"] <= split_time].copy()
            test_df = df[df["measured_at"] > split_time].copy()
            return train_df, test_df, {"strategy": "temporal", "split_point": split_time}

    # Default split: random (time-independent), reproducible via random_state.
    random_state = int(scfg.get("random_state", 42))
    rng = np.random.default_rng(random_state)
    n = len(df)
    n_test = int(round(n * test_fraction))
    n_test = min(max(n_test, 1), n - 1)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    return train_df, test_df, {"strategy": "random", "split_point": None}


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return BinaryMetrics(tp=tp, fp=fp, tn=tn, fn=fn).to_dict()


def _flatten_params(prefix: str, obj) -> Dict[str, object]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            out.update(_flatten_params(key, v))
    else:
        out[prefix] = obj
    return out


def _expand_grid(method_cfg: dict) -> List[dict]:
    """
    Expand a method config where any key can be:
    - scalar -> single value
    - list   -> options for grid search
    - dict   -> recursively expanded
    """
    keys = list(method_cfg.keys())
    option_lists = []

    for k in keys:
        v = method_cfg[k]
        if isinstance(v, dict):
            nested = _expand_grid(v)
            option_lists.append([(k, x) for x in nested])
        elif isinstance(v, list):
            option_lists.append([(k, x) for x in v])
        else:
            option_lists.append([(k, v)])

    variants = []
    for combo in itertools.product(*option_lists):
        d = {}
        for k, v in combo:
            d[k] = v
        variants.append(d)
    return variants


def _method_summary(
    method_name: str,
    method_group: str,
    variant: int,
    threshold_dbm: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_s: float,
    pred_s: float,
    method_cfg: dict | None = None,
    n_train_global: int | None = None,
    n_train_used: int | None = None,
) -> Dict[str, float]:
    metrics = _evaluate_predictions(y_true, y_pred)
    params_flat = _flatten_params("param", method_cfg or {})
    metrics.update(
        {
            "method": method_name,
            "method_group": method_group,
            "variant": int(variant),
            "threshold_dbm": float(threshold_dbm),
            "n_train_global": (None if n_train_global is None else int(n_train_global)),
            "n_train_used": (None if n_train_used is None else int(n_train_used)),
            "n_test": int(len(y_true)),
            "fit_seconds": float(fit_s),
            "predict_seconds": float(pred_s),
            "total_seconds": float(fit_s + pred_s),
        }
    )
    metrics.update(params_flat)
    return metrics


def _order_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    first_cols = [
        "method",
        "method_group",
        "variant",
        "threshold_dbm",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "n_train_global",
        "n_train_used",
        "n_test",
        "fit_seconds",
        "predict_seconds",
        "total_seconds",
    ]
    ordered = [c for c in first_cols if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def _format_summary_for_output(df: pd.DataFrame) -> pd.DataFrame:
    return _order_summary_columns(df.copy())


def print_compact_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No summary rows.")
        return

    preferred_cols = [
        "method",
        "method_group",
        "variant",
        "threshold_dbm",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "fit_seconds",
        "predict_seconds",
        "total_seconds",
        "param_min_samples",
        "param_outlier_strategy",
        "param_outlier_percentile",
        "param_alpha",
        "param_variogram_model",
        "param_power",
        "param_k_neighbors",
        "param_length_scale",
        "param_noise_level",
        "param_model",
        "param_feature_set",
        "param_n_estimators",
        "param_max_depth",
    ]
    cols = [c for c in preferred_cols if c in summary.columns]
    sort_by = ["f1", "precision"]
    ascending = [False, False]
    if "threshold_dbm" in cols:
        sort_by = ["threshold_dbm"] + sort_by
        ascending = [False] + ascending
    compact = summary[cols].sort_values(by=sort_by, ascending=ascending).copy()
    compact = compact.rename(
        columns={
            "threshold_dbm": "thr_dbm",
            "fit_seconds": "fit_sec",
            "predict_seconds": "predict_sec",
            "total_seconds": "total_sec",
            "param_min_samples": "min_samp",
            "param_outlier_strategy": "outlier",
            "param_outlier_percentile": "outlier_pct",
            "param_alpha": "alpha",
            "param_variogram_model": "variogram",
            "param_power": "idw_power",
            "param_k_neighbors": "k_neighbors",
            "param_length_scale": "gpr_ls",
            "param_noise_level": "gpr_noise",
            "param_model": "ml_model",
            "param_feature_set": "feature_set",
            "param_n_estimators": "n_estimators",
            "param_max_depth": "max_depth",
        }
    )
    print(compact.to_string(index=False))


def save_per_method_csv(summary: pd.DataFrame, summary_path: str) -> List[str]:
    if summary.empty or "method_group" not in summary.columns:
        return []

    base, ext = os.path.splitext(summary_path)
    ext = ext or ".csv"
    written = []
    name_map = {
        "0_CH": "convex_hull",
        "1_alpha_shape": "alpha_shape",
        "2_kriging": "kriging",
        "2_kriging_missing_dep": "kriging_missing_dep",
        "3_idw": "idw",
        "5_gpr": "gpr",
        "4_ml": "ml",
        "4_ml_missing_dep": "ml_missing_dep",
    }
    for method_group, group_df in summary.groupby("method_group"):
        safe = name_map.get(str(method_group), str(method_group))
        safe = safe.replace("/", "_").replace(" ", "_")
        out_path = f"{base}_{safe}{ext}"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        group_df.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def append_per_method_csv_rows(rows_df: pd.DataFrame, summary_path: str) -> List[str]:
    if not summary_path or rows_df.empty or "method_group" not in rows_df.columns:
        return []
    rows_df = _format_summary_for_output(rows_df)
    base, ext = os.path.splitext(summary_path)
    ext = ext or ".csv"
    written = []
    name_map = {
        "0_CH": "convex_hull",
        "1_alpha_shape": "alpha_shape",
        "2_kriging": "kriging",
        "2_kriging_missing_dep": "kriging_missing_dep",
        "3_idw": "idw",
        "5_gpr": "gpr",
        "4_ml": "ml",
        "4_ml_missing_dep": "ml_missing_dep",
    }
    for method_group, group_df in rows_df.groupby("method_group"):
        safe = name_map.get(str(method_group), str(method_group))
        safe = safe.replace("/", "_").replace(" ", "_")
        out_path = f"{base}_{safe}{ext}"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        write_header = not os.path.exists(out_path)
        group_df.to_csv(out_path, mode="a", header=write_header, index=False)
        written.append(out_path)
    return written


def clear_output_files(summary_path: str, split_path: str, methods_to_run: List[str] | None) -> None:
    # split info is always regenerated for the current run
    if os.path.exists(split_path):
        os.remove(split_path)

    # remove only per-method files that are part of this run
    method_name_map = {
        "convex_hull": ["convex_hull"],
        "alpha_shape": ["alpha_shape"],
        "kriging": ["kriging", "kriging_missing_dep"],
        "idw": ["idw"],
        "gpr": ["gpr", "gpr_missing_dep"],
        "ml": ["ml", "ml_missing_dep"],
    }
    selected = methods_to_run or ["convex_hull", "kriging", "idw", "ml"]

    base, ext = os.path.splitext(summary_path)
    ext = ext or ".csv"
    for mk in selected:
        for suffix in method_name_map.get(mk, []):
            p = f"{base}_{suffix}{ext}"
            if os.path.exists(p):
                os.remove(p)


def _outlier_cfg(method_cfg: dict) -> dict:
    return method_cfg.get("outlier", {"strategy": "none"})


def _filter_group_outliers(group: pd.DataFrame, out_cfg: dict) -> pd.DataFrame:
    strategy = str(out_cfg.get("strategy", "none")).lower()
    if strategy == "none" or len(group) < 4:
        return group

    pts = group[["lon", "lat"]].to_numpy(dtype=float)
    center = np.median(pts, axis=0)
    dist = np.sqrt(np.sum((pts - center) ** 2, axis=1))

    if strategy == "percentile":
        q = float(out_cfg.get("percentile", 97.0))
        q = min(max(q, 1.0), 100.0)
        thr = np.percentile(dist, q)
        keep = dist <= thr
    elif strategy == "mad":
        k = float(out_cfg.get("mad_k", 3.5))
        med = np.median(dist)
        mad = np.median(np.abs(dist - med))
        if mad <= 0:
            return group
        robust_sigma = 1.4826 * mad
        keep = dist <= (med + k * robust_sigma)
    elif strategy == "dbscan":
        if DBSCAN is None:
            return group
        eps = float(out_cfg.get("dbscan_eps", 0.0007))
        min_samples = int(out_cfg.get("dbscan_min_samples", 5))
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
        valid = labels[labels >= 0]
        if len(valid) == 0:
            return group
        counts = np.bincount(valid)
        main_cluster = int(np.argmax(counts))
        keep = labels == main_cluster
    else:
        return group

    min_keep = int(out_cfg.get("min_keep", 3))
    if int(np.sum(keep)) < min_keep:
        return group
    return group.iloc[np.where(keep)[0]]


def preprocess_for_method(train_df: pd.DataFrame, method_cfg: dict) -> pd.DataFrame:
    out_cfg = _outlier_cfg(method_cfg)
    filtered_groups = []
    for _, g in train_df.groupby("cell_id"):
        filtered_groups.append(_filter_group_outliers(g, out_cfg))
    if not filtered_groups:
        return train_df.iloc[0:0].copy()
    return pd.concat(filtered_groups, ignore_index=True)


def run_all_methods(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: dict,
    methods_to_run: List[str] | None = None,
    summary_path_for_checkpoints: str | None = None,
) -> pd.DataFrame:
    test_df = test_df.reset_index(drop=True)
    thr_cfg = cfg["preprocess"]["signal_threshold_dbm"]
    thresholds = [float(x) for x in thr_cfg] if isinstance(thr_cfg, list) else [float(thr_cfg)]

    mcfg = cfg["methods"]
    methods_set = set(methods_to_run or ["convex_hull", "kriging", "idw", "ml"])

    test_cells = test_df["cell_id"].to_numpy(dtype=int)
    test_lons = test_df["lon"].to_numpy(dtype=float)
    test_lats = test_df["lat"].to_numpy(dtype=float)
    n_train_global = int(len(train_df))

    summaries = []
    total_thr = len(thresholds)
    method_sequence = [
        ("convex_hull", "CONVEX_HULL"),
        ("alpha_shape", "ALPHA_SHAPE"),
        ("kriging", "KRIGING"),
        ("idw", "IDW"),
        ("gpr", "GPR"),
        ("ml", "ML"),
    ]

    for method_key, method_label in method_sequence:
        if method_key not in methods_set:
            continue

        if method_key == "convex_hull":
            variants = _expand_grid(mcfg["convex_hull"])
            method_group = "0_CH"
            print(f"[{method_label}] Convex Hull variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting Convex Hull...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                model = ConvexHullCoverageModel(min_samples=int(method_cfg["min_samples"]))
                t0 = time.perf_counter()
                model.fit(train_m, "cell_id", "lon", "lat")
                fit_s = time.perf_counter() - t0
                print(f"[{method_label}][v{vidx}] Done Convex Hull (fit={fit_s:.3f}s)")
                fitted.append((vidx, method_cfg, model, fit_s, int(len(train_m))))

            for tidx, thr in enumerate(thresholds, start=1):
                print(f"[{method_label}] Threshold {tidx}/{total_thr}: {thr:.1f} dBm")
                y_true = (test_df["signal"].to_numpy(dtype=float) >= thr).astype(int)
                for vidx, method_cfg, model, fit_s, n_train_used in fitted:
                    t0 = time.perf_counter()
                    y_pred = np.array([int(model.predict_inside(cid, lo, la)) for cid, lo, la in zip(test_cells, test_lons, test_lats)], dtype=int)
                    pred_s = time.perf_counter() - t0
                    summaries.append(
                        _method_summary(
                            f"{method_group}_v{vidx}",
                            method_group,
                            vidx,
                            thr,
                            y_true,
                            y_pred,
                            fit_s,
                            pred_s,
                            method_cfg=method_cfg,
                            n_train_global=n_train_global,
                            n_train_used=n_train_used,
                        )
                    )
                    append_per_method_csv_rows(pd.DataFrame([summaries[-1]]), summary_path_for_checkpoints or "")
                    print(f"[{method_label}][v{vidx}] Done Convex Hull predict (thr={thr:.1f}, pred={pred_s:.3f}s)")

        elif method_key == "alpha_shape":
            variants = _expand_grid(mcfg["alpha_shape"])
            method_group = "1_alpha_shape"
            print(f"[{method_label}] Alpha-shape variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting Alpha-shape...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                model = AlphaShapeCoverageModel(
                    min_samples=int(method_cfg["min_samples"]),
                    alpha=float(method_cfg["alpha"]),
                )
                t0 = time.perf_counter()
                model.fit(train_m, "cell_id", "lon", "lat")
                fit_s = time.perf_counter() - t0
                print(f"[{method_label}][v{vidx}] Done Alpha-shape (fit={fit_s:.3f}s)")
                fitted.append((vidx, method_cfg, model, fit_s, int(len(train_m))))

            for tidx, thr in enumerate(thresholds, start=1):
                print(f"[{method_label}] Threshold {tidx}/{total_thr}: {thr:.1f} dBm")
                y_true = (test_df["signal"].to_numpy(dtype=float) >= thr).astype(int)
                for vidx, method_cfg, model, fit_s, n_train_used in fitted:
                    t0 = time.perf_counter()
                    y_pred = np.array([int(model.predict_inside(cid, lo, la)) for cid, lo, la in zip(test_cells, test_lons, test_lats)], dtype=int)
                    pred_s = time.perf_counter() - t0
                    summaries.append(
                        _method_summary(
                            f"{method_group}_v{vidx}",
                            method_group,
                            vidx,
                            thr,
                            y_true,
                            y_pred,
                            fit_s,
                            pred_s,
                            method_cfg=method_cfg,
                            n_train_global=n_train_global,
                            n_train_used=n_train_used,
                        )
                    )
                    append_per_method_csv_rows(pd.DataFrame([summaries[-1]]), summary_path_for_checkpoints or "")
                    print(f"[{method_label}][v{vidx}] Done Alpha-shape predict (thr={thr:.1f}, pred={pred_s:.3f}s)")

        elif method_key == "kriging":
            variants = _expand_grid(mcfg["kriging"])
            print(f"[{method_label}] Kriging variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting Kriging...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                model = KrigingCoverageModel(
                    min_samples=int(method_cfg["min_samples"]),
                    variogram_model=str(method_cfg.get("variogram_model", "linear")),
                    nlags=int(method_cfg.get("nlags", 6)),
                    jitter_epsilon=float(method_cfg.get("jitter_epsilon", 1e-2)),
                )
                t0 = time.perf_counter()
                model.fit(train_m, "cell_id", "lon", "lat", "signal")
                fit_s = time.perf_counter() - t0
                suffix = "" if model.available else "_missing_dep"
                method_group = f"2_kriging{suffix}"
                print(f"[{method_label}][v{vidx}] Done Kriging{suffix} (fit={fit_s:.3f}s)")
                fitted.append((vidx, method_cfg, method_group, model, fit_s, int(len(train_m))))

            test_groups = [
                (int(cell_id), np.asarray(list(idx), dtype=int))
                for cell_id, idx in test_df.groupby("cell_id").groups.items()
            ]
            cached_predictions = []
            for vidx, method_cfg, method_group, model, fit_s, n_train_used in fitted:
                t0 = time.perf_counter()
                sig_pred = np.full(shape=len(test_df), fill_value=np.nan, dtype=float)
                for cell_id, idx in test_groups:
                    sig_pred[idx] = model.predict_signal(cell_id, test_lons[idx], test_lats[idx])
                pred_signal_s = time.perf_counter() - t0
                cached_predictions.append(
                    (vidx, method_cfg, method_group, fit_s, n_train_used, sig_pred, pred_signal_s)
                )
                print(
                    f"[{method_label}][v{vidx}] Cached Kriging signal prediction "
                    f"(all test points, {pred_signal_s:.3f}s)"
                )

            for tidx, thr in enumerate(thresholds, start=1):
                print(f"[{method_label}] Threshold {tidx}/{total_thr}: {thr:.1f} dBm")
                y_true = (test_df["signal"].to_numpy(dtype=float) >= thr).astype(int)
                for vidx, method_cfg, method_group, fit_s, n_train_used, sig_pred, pred_signal_s in cached_predictions:
                    t0 = time.perf_counter()
                    y_pred = np.where(np.isnan(sig_pred), 0, (sig_pred >= thr).astype(int))
                    threshold_s = time.perf_counter() - t0
                    # Amortize threshold-independent Kriging signal prediction across thresholds.
                    pred_s = (pred_signal_s / float(total_thr)) + threshold_s
                    summaries.append(
                        _method_summary(
                            f"{method_group}_v{vidx}",
                            method_group,
                            vidx,
                            thr,
                            y_true,
                            y_pred,
                            fit_s,
                            pred_s,
                            method_cfg=method_cfg,
                            n_train_global=n_train_global,
                            n_train_used=n_train_used,
                        )
                    )
                    append_per_method_csv_rows(pd.DataFrame([summaries[-1]]), summary_path_for_checkpoints or "")
                    print(f"[{method_label}][v{vidx}] Done Kriging predict (thr={thr:.1f}, pred={pred_s:.3f}s)")

        elif method_key == "idw":
            variants = _expand_grid(mcfg["idw"])
            print(f"[{method_label}] IDW variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting IDW...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                model = IDWCoverageModel(
                    min_samples=int(method_cfg["min_samples"]),
                    k_neighbors=int(method_cfg.get("k_neighbors", 12)),
                    power=float(method_cfg.get("power", 2.0)),
                    epsilon=float(method_cfg.get("epsilon", 1e-12)),
                )
                t0 = time.perf_counter()
                model.fit(train_m, "cell_id", "lon", "lat", "signal")
                fit_s = time.perf_counter() - t0
                method_group = "3_idw"
                print(f"[{method_label}][v{vidx}] Done IDW (fit={fit_s:.3f}s)")
                fitted.append((vidx, method_cfg, method_group, model, fit_s, int(len(train_m))))

            for tidx, thr in enumerate(thresholds, start=1):
                print(f"[{method_label}] Threshold {tidx}/{total_thr}: {thr:.1f} dBm")
                y_true = (test_df["signal"].to_numpy(dtype=float) >= thr).astype(int)
                for vidx, method_cfg, method_group, model, fit_s, n_train_used in fitted:
                    t0 = time.perf_counter()
                    y_pred = np.zeros(shape=len(test_df), dtype=int)
                    for cell_id, idx in test_df.groupby("cell_id").groups.items():
                        idx = np.asarray(list(idx), dtype=int)
                        sig = model.predict_signal(int(cell_id), test_lons[idx], test_lats[idx])
                        y_pred[idx] = np.where(np.isnan(sig), 0, (sig >= thr).astype(int))
                    pred_s = time.perf_counter() - t0
                    summaries.append(
                        _method_summary(
                            f"{method_group}_v{vidx}",
                            method_group,
                            vidx,
                            thr,
                            y_true,
                            y_pred,
                            fit_s,
                            pred_s,
                            method_cfg=method_cfg,
                            n_train_global=n_train_global,
                            n_train_used=n_train_used,
                        )
                    )
                    append_per_method_csv_rows(pd.DataFrame([summaries[-1]]), summary_path_for_checkpoints or "")
                    print(f"[{method_label}][v{vidx}] Done IDW predict (thr={thr:.1f}, pred={pred_s:.3f}s)")

        elif method_key == "ml":
            variants = _expand_grid(mcfg["ml"])
            print(f"[{method_label}] ML variants: {len(variants)}")
            preprocessed_train = []
            for vidx, method_cfg in enumerate(variants, start=1):
                base_train_m = preprocess_for_method(train_df, method_cfg).copy()
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(base_train_m)}")
                preprocessed_train.append((vidx, method_cfg, base_train_m))

            groups = list(test_df.groupby("cell_id").groups.items())
            total_groups = len(groups)
            for tidx, thr in enumerate(thresholds, start=1):
                print(f"[{method_label}] Threshold {tidx}/{total_thr}: {thr:.1f} dBm")
                y_true = (test_df["signal"].to_numpy(dtype=float) >= thr).astype(int)
                for vidx, method_cfg, base_train_m in preprocessed_train:
                    feature_set = str(method_cfg.get("feature_set", "baseline")).lower()
                    if feature_set == "enriched":
                        feature_cols = [
                            c
                            for c in [
                                "lon",
                                "lat",
                                "dist_bs_m",
                                "delta_alt_m",
                                "bs_dem_alt_m",
                                "bs_is_5g",
                                "bs_band_num",
                            ]
                            if c in base_train_m.columns and c in test_df.columns
                        ]
                        if len(feature_cols) < 2:
                            feature_cols = ["lon", "lat"]
                    else:
                        feature_cols = ["lon", "lat"]
                    predict_progress_every = int(method_cfg.get("predict_progress_every", 50))
                    print(f"[{method_label}][v{vidx}] Fitting ML classifier... features={feature_set}:{','.join(feature_cols)}")
                    train_m = base_train_m.copy()
                    train_m["label"] = (train_m["signal"] >= thr).astype(int)
                    model = MLCoverageModel(
                        min_samples=int(method_cfg["min_samples"]),
                        model=str(method_cfg.get("model", "rf")),
                        random_state=int(method_cfg.get("random_state", 42)),
                        n_estimators=int(method_cfg.get("n_estimators", 120)),
                        max_depth=(None if method_cfg.get("max_depth") is None else int(method_cfg.get("max_depth"))),
                        min_samples_leaf=int(method_cfg.get("min_samples_leaf", 3)),
                        min_samples_split=int(method_cfg.get("min_samples_split", 2)),
                        learning_rate=float(method_cfg.get("learning_rate", 0.1)),
                        subsample=float(method_cfg.get("subsample", 1.0)),
                        colsample_bytree=float(method_cfg.get("colsample_bytree", 1.0)),
                        class_weight=method_cfg.get("class_weight"),
                        use_gpu=bool(method_cfg.get("use_gpu", False)),
                    )
                    t0 = time.perf_counter()
                    model.fit(
                        train_m,
                        "cell_id",
                        "lon",
                        "lat",
                        "label",
                        feature_cols=feature_cols,
                        progress_every=int(method_cfg.get("fit_progress_every", 50)),
                        log_fn=print,
                    )
                    fit_s = time.perf_counter() - t0
                    suffix = "" if model.available else "_missing_dep"
                    method_group = f"4_ml{suffix}"

                    t0 = time.perf_counter()
                    y_pred = np.zeros(shape=len(test_df), dtype=int)
                    for gidx, (cell_id, idx) in enumerate(groups, start=1):
                        idx = np.asarray(list(idx), dtype=int)
                        pred = model.predict_on_frame(int(cell_id), test_df.iloc[idx][feature_cols]).astype(int)
                        y_pred[idx] = pred
                        if gidx % predict_progress_every == 0 or gidx == total_groups:
                            print(f"[{method_label}][v{vidx}] ML predict progress: {gidx}/{total_groups} cells")
                    pred_s = time.perf_counter() - t0
                    summaries.append(
                        _method_summary(
                            f"{method_group}_v{vidx}",
                            method_group,
                            vidx,
                            thr,
                            y_true,
                            y_pred,
                            fit_s,
                            pred_s,
                            method_cfg=method_cfg,
                            n_train_global=n_train_global,
                            n_train_used=int(len(base_train_m)),
                        )
                    )
                    append_per_method_csv_rows(pd.DataFrame([summaries[-1]]), summary_path_for_checkpoints or "")
                    print(f"[{method_label}][v{vidx}] Done ML{suffix} (fit={fit_s:.3f}s, thr={thr:.1f})")

        elif method_key == "gpr":
            variants = _expand_grid(mcfg["gpr"])
            print(f"[{method_label}] GPR variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting GPR...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                model = GPRCoverageModel(
                    min_samples=int(method_cfg["min_samples"]),
                    length_scale=float(method_cfg.get("length_scale", 0.001)),
                    noise_level=float(method_cfg.get("noise_level", 1.0)),
                    alpha=float(method_cfg.get("alpha", 1e-6)),
                    normalize_y=bool(method_cfg.get("normalize_y", True)),
                    max_train=int(method_cfg.get("max_train", 300)),
                    random_state=int(method_cfg.get("random_state", 42)),
                )
                t0 = time.perf_counter()
                model.fit(train_m, "cell_id", "lon", "lat", "signal")
                fit_s = time.perf_counter() - t0
                suffix = "" if model.available else "_missing_dep"
                method_group = f"5_gpr{suffix}"
                print(f"[{method_label}][v{vidx}] Done GPR{suffix} (fit={fit_s:.3f}s)")
                fitted.append((vidx, method_cfg, method_group, model, fit_s, int(len(train_m))))

            for tidx, thr in enumerate(thresholds, start=1):
                print(f"[{method_label}] Threshold {tidx}/{total_thr}: {thr:.1f} dBm")
                y_true = (test_df["signal"].to_numpy(dtype=float) >= thr).astype(int)
                for vidx, method_cfg, method_group, model, fit_s, n_train_used in fitted:
                    t0 = time.perf_counter()
                    y_pred = np.zeros(shape=len(test_df), dtype=int)
                    for cell_id, idx in test_df.groupby("cell_id").groups.items():
                        idx = np.asarray(list(idx), dtype=int)
                        sig = model.predict_signal(int(cell_id), test_lons[idx], test_lats[idx])
                        y_pred[idx] = np.where(np.isnan(sig), 0, (sig >= thr).astype(int))
                    pred_s = time.perf_counter() - t0
                    summaries.append(
                        _method_summary(
                            f"{method_group}_v{vidx}",
                            method_group,
                            vidx,
                            thr,
                            y_true,
                            y_pred,
                            fit_s,
                            pred_s,
                            method_cfg=method_cfg,
                            n_train_global=n_train_global,
                            n_train_used=n_train_used,
                        )
                    )
                    append_per_method_csv_rows(pd.DataFrame([summaries[-1]]), summary_path_for_checkpoints or "")
                    print(f"[{method_label}][v{vidx}] Done GPR predict (thr={thr:.1f}, pred={pred_s:.3f}s)")

        # Method-level checkpoint save (all thresholds done for this method)
        if summary_path_for_checkpoints:
            print("[CHECKPOINT] Method completed (rows already appended).")

    return pd.DataFrame(summaries)

def main():
    parser = argparse.ArgumentParser(description="Coverage benchmark runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset: convex_hull,alpha_shape,kriging,idw,gpr,ml (all=convex_hull,kriging,idw,ml)",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix tag for output filenames to avoid overwriting (e.g. FULL_20260317).",
    )
    parser.add_argument(
        "--include-ocid",
        action="store_true",
        help="Include OCID files from data/connectivity/dataset (override data.skip_ocid=true).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_cells is not None:
        cfg.setdefault("experiment", {})["max_cells"] = int(args.max_cells)
    if args.include_ocid:
        cfg.setdefault("data", {})["skip_ocid"] = False

    print("[INIT] Loading data...")
    df = load_all_data(cfg)
    print(f"[INIT] Loaded rows: {len(df)}")
    print("[INIT] Enriching with derived BS/DEM features (if available)...")
    df = enrich_with_derived_features(df, cfg)
    print("[INIT] Applying common filters...")
    df = apply_common_filters(df, cfg)
    print(f"[INIT] Rows after filters: {len(df)} | cells: {df['cell_id'].nunique() if not df.empty else 0}")
    print("[INIT] Building shared split...")
    train_df, test_df, split_meta = split_data(df, cfg)
    print(
        f"[INIT] Split ready | strategy={split_meta.get('strategy')} "
        f"train={len(train_df)} test={len(test_df)} split_point={split_meta.get('split_point')}"
    )

    if train_df.empty or test_df.empty:
        print("No data available after preprocessing/split.")
        return

    out_summary = _append_tag_to_path(cfg["output"]["summary_csv"], args.run_tag.strip() or None)
    out_split = _append_tag_to_path(cfg["output"]["split_info_csv"], args.run_tag.strip() or None)
    for out_path in [out_summary, out_split]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    methods_to_run = None
    if args.methods.strip().lower() != "all":
        methods_to_run = [m.strip() for m in args.methods.split(",") if m.strip()]
    clear_output_files(out_summary, out_split, methods_to_run)

    summary = run_all_methods(
        train_df,
        test_df,
        cfg,
        methods_to_run=methods_to_run,
        summary_path_for_checkpoints=out_summary,
    )
    print("[SAVE] Writing summary...")
    summary = _format_summary_for_output(summary)
    per_method_paths = save_per_method_csv(summary, out_summary)

    split_info = pd.DataFrame(
        [
            {
                "n_total": len(df),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "split_strategy": split_meta.get("strategy"),
                "split_point": split_meta.get("split_point"),
                "n_cells_total": df["cell_id"].nunique(),
                "n_cells_train": train_df["cell_id"].nunique(),
                "n_cells_test": test_df["cell_id"].nunique(),
            }
        ]
    )
    print("[SAVE] Writing split info...")
    split_info.to_csv(out_split, index=False)

    if per_method_paths:
        print("Saved per-method CSV files:")
        for p in per_method_paths:
            print(f" - {p}")
    print(f"Saved split info to {out_split}")
    print("[RESULT] Compact table (sorted by F1):")
    print_compact_summary(summary)


if __name__ == "__main__":
    main()
