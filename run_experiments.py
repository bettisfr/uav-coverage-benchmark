from __future__ import annotations

import argparse
import copy
import glob
import itertools
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from methods.alpha_shape import AlphaShapeCoverageModel
from methods.convex_hull import ConvexHullCoverageModel
from methods.gpr_model import GPRCoverageModel
from methods.idw_model import IDWCoverageModel
from methods.kriging_model import KrigingCoverageModel
from methods.ml_regressor import MLSignalModel

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
    test_fraction = float(scfg.get("test_fraction", 0.3))
    test_fraction = min(max(test_fraction, 0.01), 0.99)

    # Default split: stratified random by cell_id (time-independent),
    # reproducible via random_state.
    random_state = int(scfg.get("random_state", 42))
    rng = np.random.default_rng(random_state)

    test_idx_parts = []
    train_idx_parts = []

    # Preserve per-cell proportions to avoid cell imbalance between train/test.
    for _, g in df.groupby("cell_id"):
        idx = g.index.to_numpy(dtype=int)
        n_cell = len(idx)
        if n_cell <= 1:
            train_idx_parts.append(idx)
            continue

        n_test_cell = int(round(n_cell * test_fraction))
        n_test_cell = min(max(n_test_cell, 1), n_cell - 1)

        perm = rng.permutation(idx)
        test_idx_parts.append(perm[:n_test_cell])
        train_idx_parts.append(perm[n_test_cell:])

    if test_idx_parts:
        test_idx = np.concatenate(test_idx_parts)
    else:
        test_idx = np.array([], dtype=int)

    if train_idx_parts:
        train_idx = np.concatenate(train_idx_parts)
    else:
        train_idx = np.array([], dtype=int)

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Safety fallback: if something degenerates, revert to global random split.
    if train_df.empty or test_df.empty:
        n = len(df)
        n_test = int(round(n * test_fraction))
        n_test = min(max(n_test, 1), n - 1)
        perm = rng.permutation(n)
        test_pos = perm[:n_test]
        train_pos = perm[n_test:]
        train_df = df.iloc[train_pos].copy()
        test_df = df.iloc[test_pos].copy()

    return train_df, test_df, {"strategy": "random", "split_point": None}


def build_stratified_kfold_splits(df: pd.DataFrame, n_splits: int, random_state: int) -> List[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    if df.empty:
        return []
    n_splits = int(max(2, n_splits))
    rng = np.random.default_rng(int(random_state))

    # For each cell_id, partition its samples into K disjoint chunks.
    grouped_chunks: Dict[int, List[np.ndarray]] = {}
    for cell_id, g in df.groupby("cell_id"):
        idx = g.index.to_numpy(dtype=int)
        perm = rng.permutation(idx)
        grouped_chunks[int(cell_id)] = [chunk.astype(int) for chunk in np.array_split(perm, n_splits)]

    splits: List[tuple[pd.DataFrame, pd.DataFrame, dict]] = []
    for fold_id in range(n_splits):
        test_parts = []
        train_parts = []
        for chunks in grouped_chunks.values():
            test_chunk = chunks[fold_id]
            train_chunks = [c for i, c in enumerate(chunks) if i != fold_id and len(c) > 0]
            if len(test_chunk) > 0:
                test_parts.append(test_chunk)
            if train_chunks:
                train_parts.append(np.concatenate(train_chunks))

        test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()

        if train_df.empty or test_df.empty:
            continue

        splits.append(
            (
                train_df,
                test_df,
                {
                    "strategy": "stratified_kfold",
                    "split_point": None,
                    "fold_id": int(fold_id),
                    "n_splits": int(n_splits),
                },
            )
        )

    return splits


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_valid = int(np.sum(mask))
    if n_valid == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "n_valid": 0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum((yt - yp) ** 2))
    yt_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - yt_mean) ** 2))
    r2 = float("nan") if ss_tot <= 1e-12 else float(1.0 - (ss_res / ss_tot))

    if len(yt) < 2:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
        yt_rank = pd.Series(yt).rank(method="average").to_numpy(dtype=float)
        yp_rank = pd.Series(yp).rank(method="average").to_numpy(dtype=float)
        spearman = float(np.corrcoef(yt_rank, yp_rank)[0, 1])

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
        "n_valid": n_valid,
    }


def _evaluate_coverage_from_signal(y_true: np.ndarray, y_pred: np.ndarray, threshold_dbm: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_valid = int(np.sum(mask))
    if n_valid == 0:
        return {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
            "n_valid": 0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    ytb = yt >= float(threshold_dbm)
    ypb = yp >= float(threshold_dbm)

    tp = int(np.sum(ypb & ytb))
    fp = int(np.sum(ypb & (~ytb)))
    tn = int(np.sum((~ypb) & (~ytb)))
    fn = int(np.sum((~ypb) & ytb))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / n_valid) if n_valid > 0 else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n_valid": n_valid,
    }


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
    y_true_signal: np.ndarray,
    y_pred_signal: np.ndarray,
    fit_s: float,
    pred_s: float,
    method_cfg: dict | None = None,
    n_train_global: int | None = None,
    n_train_used: int | None = None,
) -> Dict[str, float]:
    metrics = _evaluate_regression(y_true_signal, y_pred_signal)
    params_flat = _flatten_params("param", method_cfg or {})
    metrics.update(
        {
            "method": method_name,
            "method_group": method_group,
            "variant": int(variant),
            "n_train_global": (None if n_train_global is None else int(n_train_global)),
            "n_train_used": (None if n_train_used is None else int(n_train_used)),
            "n_test": int(len(y_true_signal)),
            "fit_seconds": float(fit_s),
            "predict_seconds": float(pred_s),
            "total_seconds": float(fit_s + pred_s),
        }
    )
    metrics.update(params_flat)
    return metrics


def _method_coverage_summaries(
    base_row: Dict[str, object],
    y_true_signal: np.ndarray,
    y_pred_signal: np.ndarray,
    thresholds_dbm: List[float],
) -> List[Dict[str, object]]:
    keep_keys = [
        "method",
        "method_group",
        "variant",
        "n_train_global",
        "n_train_used",
        "n_test",
        "fit_seconds",
        "predict_seconds",
        "total_seconds",
    ]
    keep_keys.extend([k for k in base_row.keys() if str(k).startswith("param_")])
    prefix = {k: base_row.get(k) for k in keep_keys if k in base_row}

    rows: List[Dict[str, object]] = []
    for tau in thresholds_dbm:
        m = _evaluate_coverage_from_signal(y_true_signal, y_pred_signal, float(tau))
        row = dict(prefix)
        row.update(
            {
                "threshold_dbm": float(tau),
                "tp": int(m["tp"]),
                "fp": int(m["fp"]),
                "tn": int(m["tn"]),
                "fn": int(m["fn"]),
                "precision": float(m["precision"]),
                "recall": float(m["recall"]),
                "f1": float(m["f1"]),
                "accuracy": float(m["accuracy"]),
                "n_valid": int(m["n_valid"]),
            }
        )
        rows.append(row)
    return rows


def _order_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    first_cols = [
        "method",
        "method_group",
        "variant",
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
        "n_valid",
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


def _order_coverage_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    first_cols = [
        "method",
        "method_group",
        "variant",
        "threshold_dbm",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "tp",
        "fp",
        "tn",
        "fn",
        "n_valid",
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


def _format_coverage_for_output(df: pd.DataFrame) -> pd.DataFrame:
    return _order_coverage_columns(df.copy())


def _aggregate_repeated_summaries(summary_raw: pd.DataFrame) -> pd.DataFrame:
    if summary_raw.empty:
        return summary_raw
    if "repeat_id" not in summary_raw.columns:
        return summary_raw

    # Aggregate per method/variant/params and report mean+std across repeats.
    group_cols = [
        c
        for c in summary_raw.columns
        if c
        in {
            "method",
            "method_group",
            "variant",
            "n_train_global",
            "n_train_used",
            "n_test",
        }
        or c.startswith("param_")
    ]
    group_cols = [c for c in group_cols if c in summary_raw.columns]

    metric_cols = [
        c
        for c in [
            "mae",
            "rmse",
            "r2",
            "pearson",
            "spearman",
            "n_valid",
            "fit_seconds",
            "predict_seconds",
            "total_seconds",
        ]
        if c in summary_raw.columns
    ]

    grouped = summary_raw.groupby(group_cols, dropna=False, as_index=False)
    mean_df = grouped[metric_cols].mean(numeric_only=True)
    out = mean_df.copy()
    out["n_repeats"] = grouped.size()["size"]

    std_df = grouped[metric_cols].std(numeric_only=True, ddof=0).fillna(0.0)
    for c in metric_cols:
        out[f"{c}_std"] = std_df[c]

    # Keep a stable column order.
    out = _order_summary_columns(out)
    std_cols = [c for c in out.columns if c.endswith("_std")]
    base_cols = [c for c in out.columns if c not in std_cols]
    return out[base_cols + std_cols]


def _aggregate_repeated_coverages(coverage_raw: pd.DataFrame) -> pd.DataFrame:
    if coverage_raw.empty:
        return coverage_raw
    if "repeat_id" not in coverage_raw.columns:
        return coverage_raw

    group_cols = [
        c
        for c in coverage_raw.columns
        if c
        in {
            "method",
            "method_group",
            "variant",
            "threshold_dbm",
            "n_train_global",
            "n_train_used",
            "n_test",
        }
        or c.startswith("param_")
    ]
    group_cols = [c for c in group_cols if c in coverage_raw.columns]

    metric_cols = [
        c
        for c in ["precision", "recall", "f1", "accuracy", "tp", "fp", "tn", "fn", "n_valid", "fit_seconds", "predict_seconds", "total_seconds"]
        if c in coverage_raw.columns
    ]

    grouped = coverage_raw.groupby(group_cols, dropna=False, as_index=False)
    mean_df = grouped[metric_cols].mean(numeric_only=True)
    out = mean_df.copy()
    out["n_repeats"] = grouped.size()["size"]

    std_df = grouped[metric_cols].std(numeric_only=True, ddof=0).fillna(0.0)
    for c in metric_cols:
        out[f"{c}_std"] = std_df[c]

    out = _order_coverage_columns(out)
    std_cols = [c for c in out.columns if c.endswith("_std")]
    base_cols = [c for c in out.columns if c not in std_cols]
    return out[base_cols + std_cols]


def print_compact_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No summary rows.")
        return

    preferred_cols = [
        "method",
        "method_group",
        "variant",
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
        "n_valid",
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
    sort_by = ["rmse", "mae"]
    ascending = [True, True]
    compact = summary[cols].sort_values(by=sort_by, ascending=ascending).copy()
    compact = compact.rename(
        columns={
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


def print_compact_coverage(coverage: pd.DataFrame) -> None:
    if coverage.empty:
        print("No coverage rows.")
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
        "tp",
        "fp",
        "tn",
        "fn",
        "n_valid",
    ]
    cols = [c for c in preferred_cols if c in coverage.columns]
    compact = coverage[cols].sort_values(by=["threshold_dbm", "f1"], ascending=[False, False]).copy()
    compact = compact.rename(
        columns={
            "threshold_dbm": "tau_dbm",
            "precision": "prec",
            "recall": "rec",
            "accuracy": "acc",
        }
    )
    print(compact.to_string(index=False))


def derive_coverage_output_path(summary_path: str) -> str:
    base, ext = os.path.splitext(summary_path)
    ext = ext or ".csv"
    if "metrics_summary" in base:
        return base.replace("metrics_summary", "coverage_summary") + ext
    return f"{base}_coverage{ext}"


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
    selected = methods_to_run or ["convex_hull", "alpha_shape", "kriging", "idw", "gpr", "ml"]

    base, ext = os.path.splitext(summary_path)
    ext = ext or ".csv"
    for mk in selected:
        for suffix in method_name_map.get(mk, []):
            p = f"{base}_{suffix}{ext}"
            if os.path.exists(p):
                os.remove(p)

    coverage_path = derive_coverage_output_path(summary_path)
    cbase, cext = os.path.splitext(coverage_path)
    cext = cext or ".csv"
    for mk in selected:
        for suffix in method_name_map.get(mk, []):
            p = f"{cbase}_{suffix}{cext}"
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
    coverage_path_for_checkpoints: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_df = test_df.reset_index(drop=True)
    mcfg = cfg["methods"]
    methods_set = set(methods_to_run or ["convex_hull", "alpha_shape", "kriging", "idw", "gpr", "ml"])

    test_cells = test_df["cell_id"].to_numpy(dtype=int)
    test_lons = test_df["lon"].to_numpy(dtype=float)
    test_lats = test_df["lat"].to_numpy(dtype=float)
    y_true_signal = test_df["signal"].to_numpy(dtype=float)
    n_train_global = int(len(train_df))
    thresholds_dbm = [float(x) for x in cfg.get("preprocess", {}).get("signal_threshold_dbm", [-105, -110, -115, -120])]

    summaries = []
    coverage_rows = []
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
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    train_m = train_m.copy()
                    train_m["cell_id"] = 0
                model = ConvexHullCoverageModel(min_samples=int(method_cfg["min_samples"]))
                t0 = time.perf_counter()
                model.fit(
                    train_m,
                    "cell_id",
                    "lon",
                    "lat",
                    "signal",
                    dist_col="dist_bs_m",
                    bs_lon_col="bs_lon",
                    bs_lat_col="bs_lat",
                )
                fit_s = time.perf_counter() - t0
                print(f"[{method_label}][v{vidx}] Done Convex Hull (fit={fit_s:.3f}s)")
                fitted.append((vidx, method_cfg, model, fit_s, int(len(train_m))))

            cached_predictions = []
            for vidx, method_cfg, model, fit_s, n_train_used in fitted:
                t0 = time.perf_counter()
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    sig_pred = np.array(
                        [float(model.predict_signal(0, lo, la)) for lo, la in zip(test_lons, test_lats)],
                        dtype=float,
                    )
                else:
                    sig_pred = np.array(
                        [float(model.predict_signal(cid, lo, la)) for cid, lo, la in zip(test_cells, test_lons, test_lats)],
                        dtype=float,
                    )
                pred_signal_s = time.perf_counter() - t0
                cached_predictions.append((vidx, method_cfg, fit_s, n_train_used, sig_pred, pred_signal_s))
                print(
                    f"[{method_label}][v{vidx}] Cached Convex Hull-LR signal prediction "
                    f"(all test points, {pred_signal_s:.3f}s)"
                )

            for vidx, method_cfg, fit_s, n_train_used, sig_pred, pred_signal_s in cached_predictions:
                row = _method_summary(
                        f"{method_group}_v{vidx}",
                        method_group,
                        vidx,
                        y_true_signal,
                        sig_pred,
                        fit_s,
                        pred_signal_s,
                        method_cfg=method_cfg,
                        n_train_global=n_train_global,
                        n_train_used=n_train_used,
                    )
                summaries.append(row)
                coverage_rows.extend(_method_coverage_summaries(row, y_true_signal, sig_pred, thresholds_dbm))
                append_per_method_csv_rows(pd.DataFrame([row]), summary_path_for_checkpoints or "")
                if coverage_path_for_checkpoints:
                    append_per_method_csv_rows(pd.DataFrame(_method_coverage_summaries(row, y_true_signal, sig_pred, thresholds_dbm)), coverage_path_for_checkpoints)
                print(f"[{method_label}][v{vidx}] Done Convex Hull-LR regression (pred={pred_signal_s:.3f}s)")

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
                model.fit(
                    train_m,
                    "cell_id",
                    "lon",
                    "lat",
                    "signal",
                    dist_col="dist_bs_m",
                    bs_lon_col="bs_lon",
                    bs_lat_col="bs_lat",
                )
                fit_s = time.perf_counter() - t0
                print(f"[{method_label}][v{vidx}] Done Alpha-shape (fit={fit_s:.3f}s)")
                fitted.append((vidx, method_cfg, model, fit_s, int(len(train_m))))

            cached_predictions = []
            for vidx, method_cfg, model, fit_s, n_train_used in fitted:
                t0 = time.perf_counter()
                sig_pred = np.array(
                    [float(model.predict_signal(cid, lo, la)) for cid, lo, la in zip(test_cells, test_lons, test_lats)],
                    dtype=float,
                )
                pred_signal_s = time.perf_counter() - t0
                cached_predictions.append((vidx, method_cfg, fit_s, n_train_used, sig_pred, pred_signal_s))
                print(
                    f"[{method_label}][v{vidx}] Cached Alpha-Shape-LR signal prediction "
                    f"(all test points, {pred_signal_s:.3f}s)"
                )

            for vidx, method_cfg, fit_s, n_train_used, sig_pred, pred_signal_s in cached_predictions:
                row = _method_summary(
                        f"{method_group}_v{vidx}",
                        method_group,
                        vidx,
                        y_true_signal,
                        sig_pred,
                        fit_s,
                        pred_signal_s,
                        method_cfg=method_cfg,
                        n_train_global=n_train_global,
                        n_train_used=n_train_used,
                    )
                summaries.append(row)
                coverage_rows.extend(_method_coverage_summaries(row, y_true_signal, sig_pred, thresholds_dbm))
                append_per_method_csv_rows(pd.DataFrame([row]), summary_path_for_checkpoints or "")
                if coverage_path_for_checkpoints:
                    append_per_method_csv_rows(pd.DataFrame(_method_coverage_summaries(row, y_true_signal, sig_pred, thresholds_dbm)), coverage_path_for_checkpoints)
                print(f"[{method_label}][v{vidx}] Done Alpha-Shape-LR regression (pred={pred_signal_s:.3f}s)")

        elif method_key == "kriging":
            variants = _expand_grid(mcfg["kriging"])
            print(f"[{method_label}] Kriging variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting Kriging...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    train_m = train_m.copy()
                    train_m["cell_id"] = 0
                    max_train_global = int(method_cfg.get("max_train_global", 2000))
                    if max_train_global > 0 and len(train_m) > max_train_global:
                        train_m = train_m.sample(
                            n=max_train_global,
                            random_state=int(method_cfg.get("random_state", 42)),
                        ).reset_index(drop=True)
                model = KrigingCoverageModel(
                    min_samples=int(method_cfg["min_samples"]),
                    variogram_model=str(method_cfg.get("variogram_model", "spherical")),
                    nlags=int(method_cfg.get("nlags", 10)),
                    jitter_epsilon=float(method_cfg.get("jitter_epsilon", 1e-2)),
                    min_signal_dbm=float(cfg.get("preprocess", {}).get("min_signal_dbm", -150.0)),
                    max_signal_dbm=float(cfg.get("preprocess", {}).get("max_signal_dbm", -30.0)),
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
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    sig_pred[:] = model.predict_signal(0, test_lons, test_lats)
                else:
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

            for vidx, method_cfg, method_group, fit_s, n_train_used, sig_pred, pred_signal_s in cached_predictions:
                sig_pred_clean = np.where(np.isnan(sig_pred), float("-inf"), sig_pred)
                row = _method_summary(
                        f"{method_group}_v{vidx}",
                        method_group,
                        vidx,
                        y_true_signal,
                        sig_pred_clean,
                        fit_s,
                        pred_signal_s,
                        method_cfg=method_cfg,
                        n_train_global=n_train_global,
                        n_train_used=n_train_used,
                    )
                summaries.append(row)
                coverage_rows.extend(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm))
                append_per_method_csv_rows(pd.DataFrame([row]), summary_path_for_checkpoints or "")
                if coverage_path_for_checkpoints:
                    append_per_method_csv_rows(pd.DataFrame(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm)), coverage_path_for_checkpoints)
                print(f"[{method_label}][v{vidx}] Done Kriging regression (pred={pred_signal_s:.3f}s)")

        elif method_key == "idw":
            variants = _expand_grid(mcfg["idw"])
            print(f"[{method_label}] IDW variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting IDW...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    train_m = train_m.copy()
                    train_m["cell_id"] = 0
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

            for vidx, method_cfg, method_group, model, fit_s, n_train_used in fitted:
                t0 = time.perf_counter()
                sig_pred = np.full(shape=len(test_df), fill_value=float("nan"), dtype=float)
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    sig_pred[:] = model.predict_signal(0, test_lons, test_lats)
                else:
                    for cell_id, idx in test_df.groupby("cell_id").groups.items():
                        idx = np.asarray(list(idx), dtype=int)
                        sig_pred[idx] = model.predict_signal(int(cell_id), test_lons[idx], test_lats[idx])
                pred_s = time.perf_counter() - t0
                sig_pred_clean = np.where(np.isnan(sig_pred), float("-inf"), sig_pred)
                row = _method_summary(
                        f"{method_group}_v{vidx}",
                        method_group,
                        vidx,
                        y_true_signal,
                        sig_pred_clean,
                        fit_s,
                        pred_s,
                        method_cfg=method_cfg,
                        n_train_global=n_train_global,
                        n_train_used=n_train_used,
                    )
                summaries.append(row)
                coverage_rows.extend(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm))
                append_per_method_csv_rows(pd.DataFrame([row]), summary_path_for_checkpoints or "")
                if coverage_path_for_checkpoints:
                    append_per_method_csv_rows(pd.DataFrame(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm)), coverage_path_for_checkpoints)
                print(f"[{method_label}][v{vidx}] Done IDW regression (pred={pred_s:.3f}s)")

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
            for vidx, method_cfg, base_train_m in preprocessed_train:
                feature_set = str(method_cfg.get("feature_set", "baseline")).lower()
                scope = str(method_cfg.get("scope", "per_cell")).lower()

                train_work = base_train_m.copy()
                test_work = test_df.copy()

                # Optional engineered features for ML-only experiments.
                for frame in (train_work, test_work):
                    if "dist_bs_m" in frame.columns:
                        d = pd.to_numeric(frame["dist_bs_m"], errors="coerce").astype(float)
                        d = d.clip(lower=0.0)
                        frame["dist_bs_km"] = d / 1000.0
                        frame["dist_bs_log"] = np.log1p(d)
                        frame["dist_bs_sq"] = d * d
                    if {"lat", "lon", "bs_lat", "bs_lon"}.issubset(frame.columns):
                        lat = pd.to_numeric(frame["lat"], errors="coerce").astype(float)
                        lon = pd.to_numeric(frame["lon"], errors="coerce").astype(float)
                        bslat = pd.to_numeric(frame["bs_lat"], errors="coerce").astype(float)
                        bslon = pd.to_numeric(frame["bs_lon"], errors="coerce").astype(float)
                        dlat = lat - bslat
                        dlon = lon - bslon
                        bearing = np.arctan2(dlon, dlat)
                        frame["bearing_sin"] = np.sin(bearing)
                        frame["bearing_cos"] = np.cos(bearing)
                    if {"delta_alt_m", "dist_bs_km"}.issubset(frame.columns):
                        da = pd.to_numeric(frame["delta_alt_m"], errors="coerce").astype(float)
                        dk = pd.to_numeric(frame["dist_bs_km"], errors="coerce").astype(float)
                        frame["delta_alt_x_dist"] = da * dk
                    if {"bs_band_num", "dist_bs_km"}.issubset(frame.columns):
                        bb = pd.to_numeric(frame["bs_band_num"], errors="coerce").astype(float)
                        dk = pd.to_numeric(frame["dist_bs_km"], errors="coerce").astype(float)
                        frame["band_x_dist"] = bb * dk

                # Keep original cell identity as feature when requested, even in global scope.
                if "cell_id" in train_work.columns:
                    train_work["cell_id_feat"] = pd.to_numeric(train_work["cell_id"], errors="coerce")
                if "cell_id" in test_work.columns:
                    test_work["cell_id_feat"] = pd.to_numeric(test_work["cell_id"], errors="coerce")

                feature_candidates = {
                    "baseline": ["lon", "lat"],
                    # historical "enriched" kept for backward compatibility
                    "enriched": ["lon", "lat", "dist_bs_m", "delta_alt_m"],
                    "dist_only": ["dist_bs_m"],
                    "geo_dist_alt": ["lon", "lat", "dist_bs_m", "delta_alt_m"],
                    "bs_context": [
                        "lon",
                        "lat",
                        "dist_bs_m",
                        "delta_alt_m",
                        "bs_lat",
                        "bs_lon",
                        "bs_is_5g",
                        "bs_has_dss",
                        "bs_band_num",
                        "bs_sector_num",
                    ],
                    # Strong global setting: add cell identity as numeric feature.
                    "global_strong": [
                        "lon",
                        "lat",
                        "dist_bs_m",
                        "dist_bs_log",
                        "dist_bs_sq",
                        "dist_bs_km",
                        "delta_alt_m",
                        "delta_alt_x_dist",
                        "bs_lat",
                        "bs_lon",
                        "bearing_sin",
                        "bearing_cos",
                        "bs_is_5g",
                        "bs_has_dss",
                        "bs_band_num",
                        "bs_sector_num",
                        "band_x_dist",
                        "cell_id_feat",
                    ],
                    # One-shot richer global feature set for fast experimentation.
                    "global_poly": [
                        "lon",
                        "lat",
                        "dist_bs_m",
                        "dist_bs_log",
                        "dist_bs_sq",
                        "dist_bs_km",
                        "delta_alt_m",
                        "delta_alt_x_dist",
                        "bs_lat",
                        "bs_lon",
                        "bearing_sin",
                        "bearing_cos",
                        "bs_is_5g",
                        "bs_has_dss",
                        "bs_band_num",
                        "bs_sector_num",
                        "band_x_dist",
                        "cell_id_feat",
                    ],
                }
                wanted = feature_candidates.get(feature_set, feature_candidates["baseline"])
                feature_cols = [c for c in wanted if c in train_work.columns and c in test_work.columns]
                if len(feature_cols) < 1:
                    feature_cols = ["lon", "lat"]

                predict_progress_every = int(method_cfg.get("predict_progress_every", 50))
                print(f"[{method_label}][v{vidx}] Fitting ML regressor... features={feature_set}:{','.join(feature_cols)}")
                train_m = train_work
                if scope == "global":
                    train_m["_model_group"] = 0
                    cell_col_fit = "_model_group"
                else:
                    cell_col_fit = "cell_id"
                model = MLSignalModel(
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
                    use_gpu=bool(method_cfg.get("use_gpu", False)),
                )
                t0 = time.perf_counter()
                model.fit(
                    train_m,
                    cell_col_fit,
                    "lon",
                    "lat",
                    "signal",
                    feature_cols=feature_cols,
                    progress_every=int(method_cfg.get("fit_progress_every", 50)),
                    log_fn=print,
                )
                fit_s = time.perf_counter() - t0
                suffix = "" if model.available else "_missing_dep"
                method_group = f"4_ml{suffix}"

                t0 = time.perf_counter()
                sig_pred = np.full(shape=len(test_df), fill_value=float("nan"), dtype=float)
                if scope == "global":
                    sig_pred[:] = model.predict_on_frame(0, test_work[feature_cols]).astype(float)
                else:
                    for gidx, (cell_id, idx) in enumerate(groups, start=1):
                        idx = np.asarray(list(idx), dtype=int)
                        pred = model.predict_on_frame(int(cell_id), test_work.iloc[idx][feature_cols]).astype(float)
                        sig_pred[idx] = pred
                        if gidx % predict_progress_every == 0 or gidx == total_groups:
                            print(f"[{method_label}][v{vidx}] ML predict progress: {gidx}/{total_groups} cells")
                pred_s = time.perf_counter() - t0
                sig_pred_clean = np.where(np.isnan(sig_pred), float("-inf"), sig_pred)
                row = _method_summary(
                        f"{method_group}_v{vidx}",
                        method_group,
                        vidx,
                        y_true_signal,
                        sig_pred_clean,
                        fit_s,
                        pred_s,
                        method_cfg=method_cfg,
                        n_train_global=n_train_global,
                        n_train_used=int(len(base_train_m)),
                    )
                summaries.append(row)
                coverage_rows.extend(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm))
                append_per_method_csv_rows(pd.DataFrame([row]), summary_path_for_checkpoints or "")
                if coverage_path_for_checkpoints:
                    append_per_method_csv_rows(pd.DataFrame(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm)), coverage_path_for_checkpoints)
                print(f"[{method_label}][v{vidx}] Done ML{suffix} regression (fit={fit_s:.3f}s, pred={pred_s:.3f}s)")

        elif method_key == "gpr":
            variants = _expand_grid(mcfg["gpr"])
            print(f"[{method_label}] GPR variants: {len(variants)}")
            fitted = []
            for vidx, method_cfg in enumerate(variants, start=1):
                print(f"[{method_label}][v{vidx}] Fitting GPR...")
                train_m = preprocess_for_method(train_df, method_cfg)
                print(f"[{method_label}][v{vidx}] Train rows after outlier filter: {len(train_m)}")
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    train_m = train_m.copy()
                    train_m["cell_id"] = 0
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

            for vidx, method_cfg, method_group, model, fit_s, n_train_used in fitted:
                t0 = time.perf_counter()
                sig_pred = np.full(shape=len(test_df), fill_value=float("nan"), dtype=float)
                scope = str(method_cfg.get("scope", "per_cell")).lower()
                if scope == "global":
                    sig_pred[:] = model.predict_signal(0, test_lons, test_lats)
                else:
                    for cell_id, idx in test_df.groupby("cell_id").groups.items():
                        idx = np.asarray(list(idx), dtype=int)
                        sig_pred[idx] = model.predict_signal(int(cell_id), test_lons[idx], test_lats[idx])
                pred_s = time.perf_counter() - t0
                sig_pred_clean = np.where(np.isnan(sig_pred), float("-inf"), sig_pred)
                row = _method_summary(
                        f"{method_group}_v{vidx}",
                        method_group,
                        vidx,
                        y_true_signal,
                        sig_pred_clean,
                        fit_s,
                        pred_s,
                        method_cfg=method_cfg,
                        n_train_global=n_train_global,
                        n_train_used=n_train_used,
                    )
                summaries.append(row)
                coverage_rows.extend(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm))
                append_per_method_csv_rows(pd.DataFrame([row]), summary_path_for_checkpoints or "")
                if coverage_path_for_checkpoints:
                    append_per_method_csv_rows(pd.DataFrame(_method_coverage_summaries(row, y_true_signal, sig_pred_clean, thresholds_dbm)), coverage_path_for_checkpoints)
                print(f"[{method_label}][v{vidx}] Done GPR regression (pred={pred_s:.3f}s)")

        # Method-level checkpoint save
        if summary_path_for_checkpoints:
            print("[CHECKPOINT] Method completed (rows already appended).")

    return pd.DataFrame(summaries), pd.DataFrame(coverage_rows)

def main():
    parser = argparse.ArgumentParser(description="Coverage benchmark runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset: convex_hull,alpha_shape,kriging,idw,gpr,ml (all=all six methods)",
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
    parser.add_argument(
        "--split-repeats",
        type=int,
        default=None,
        help="Number of repeated stratified random splits by cell_id (1 = single split).",
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=None,
        help="Enable stratified K-fold split by cell_id with the given number of folds.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_cells is not None:
        cfg.setdefault("experiment", {})["max_cells"] = int(args.max_cells)
    if args.include_ocid:
        cfg.setdefault("data", {})["skip_ocid"] = False
    if args.split_repeats is not None:
        cfg.setdefault("split", {})["repeats"] = int(args.split_repeats)
    if args.kfolds is not None:
        cfg.setdefault("split", {})["strategy"] = "stratified_kfold"
        cfg.setdefault("split", {})["n_splits"] = int(args.kfolds)

    print("[INIT] Loading data...")
    df = load_all_data(cfg)
    print(f"[INIT] Loaded rows: {len(df)}")
    print("[INIT] Enriching with derived BS/DEM features (if available)...")
    df = enrich_with_derived_features(df, cfg)
    print("[INIT] Applying common filters...")
    df = apply_common_filters(df, cfg)
    print(f"[INIT] Rows after filters: {len(df)} | cells: {df['cell_id'].nunique() if not df.empty else 0}")
    out_summary = _append_tag_to_path(cfg["output"]["summary_csv"], args.run_tag.strip() or None)
    out_coverage = derive_coverage_output_path(out_summary)
    out_split = _append_tag_to_path(cfg["output"]["split_info_csv"], args.run_tag.strip() or None)
    for out_path in [out_summary, out_coverage, out_split]:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    methods_to_run = None
    if args.methods.strip().lower() != "all":
        methods_to_run = [m.strip() for m in args.methods.split(",") if m.strip()]
    clear_output_files(out_summary, out_split, methods_to_run)

    repeats = int(cfg.get("split", {}).get("repeats", 1))
    repeats = max(repeats, 1)
    base_seed = int(cfg.get("split", {}).get("random_state", 42))
    strategy = str(cfg.get("split", {}).get("strategy", "random")).lower()
    n_splits = int(cfg.get("split", {}).get("n_splits", 5))
    print(f"[INIT] Split strategy={strategy} | repeats={repeats} | n_splits={n_splits} | base_seed={base_seed}")

    all_summaries = []
    all_coverages = []
    split_rows = []

    split_jobs: List[tuple[int, pd.DataFrame, pd.DataFrame, dict]] = []
    if strategy == "stratified_kfold":
        kfold_splits = build_stratified_kfold_splits(df, n_splits=n_splits, random_state=base_seed)
        for ridx, (train_df, test_df, split_meta) in enumerate(kfold_splits):
            split_jobs.append((ridx, train_df, test_df, split_meta))
    else:
        print(f"[INIT] Using stratified random split by cell_id with repeats={repeats} (repeats=1 => single split).")
        for ridx in range(repeats):
            cfg_rep = copy.deepcopy(cfg)
            if strategy == "random":
                cfg_rep.setdefault("split", {})["random_state"] = base_seed + ridx
            train_df, test_df, split_meta = split_data(df, cfg_rep)
            split_jobs.append((ridx, train_df, test_df, split_meta))

    total_runs = len(split_jobs)
    for run_idx, (ridx, train_df, test_df, split_meta) in enumerate(split_jobs, start=1):
        print(f"[RUN] Split {run_idx}/{total_runs}...")
        print(
            f"[RUN] Split ready | run={ridx} strategy={split_meta.get('strategy')} "
            f"train={len(train_df)} test={len(test_df)} split_point={split_meta.get('split_point')} "
            f"fold={split_meta.get('fold_id')}"
        )
        if train_df.empty or test_df.empty:
            print(f"[WARN] Empty split at run={ridx}; skipping.")
            continue

        cfg_rep = copy.deepcopy(cfg)
        if strategy == "random":
            cfg_rep.setdefault("split", {})["random_state"] = base_seed + ridx

        summary_r, coverage_r = run_all_methods(
            train_df,
            test_df,
            cfg_rep,
            methods_to_run=methods_to_run,
            # For repeated runs we aggregate at the end; avoid mixed per-row checkpoint appends.
            summary_path_for_checkpoints=(out_summary if total_runs == 1 else None),
            coverage_path_for_checkpoints=(out_coverage if total_runs == 1 else None),
        )
        if not summary_r.empty:
            summary_r = summary_r.copy()
            summary_r["repeat_id"] = int(ridx)
            all_summaries.append(summary_r)
        if not coverage_r.empty:
            coverage_r = coverage_r.copy()
            coverage_r["repeat_id"] = int(ridx)
            all_coverages.append(coverage_r)

        split_rows.append(
            {
                "repeat_id": int(ridx),
                "n_total": len(df),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "split_strategy": split_meta.get("strategy"),
                "split_point": split_meta.get("split_point"),
                "fold_id": split_meta.get("fold_id"),
                "n_splits": split_meta.get("n_splits"),
                "n_cells_total": df["cell_id"].nunique(),
                "n_cells_train": train_df["cell_id"].nunique(),
                "n_cells_test": test_df["cell_id"].nunique(),
            }
        )

    if not all_summaries:
        print("No data available after preprocessing/split.")
        return

    summary_raw = pd.concat(all_summaries, ignore_index=True)
    summary = _aggregate_repeated_summaries(summary_raw) if total_runs > 1 else summary_raw
    coverage_raw = pd.concat(all_coverages, ignore_index=True) if all_coverages else pd.DataFrame()
    coverage = _aggregate_repeated_coverages(coverage_raw) if (total_runs > 1 and not coverage_raw.empty) else coverage_raw
    print("[SAVE] Writing summary...")
    summary = _format_summary_for_output(summary)
    per_method_paths = save_per_method_csv(summary, out_summary)
    if not coverage.empty:
        print("[SAVE] Writing coverage summary...")
        coverage = _format_coverage_for_output(coverage)
        per_method_cov_paths = save_per_method_csv(coverage, out_coverage)
    else:
        per_method_cov_paths = []

    split_info = pd.DataFrame(split_rows)
    print("[SAVE] Writing split info...")
    split_info.to_csv(out_split, index=False)

    if per_method_paths:
        print("Saved per-method CSV files:")
        for p in per_method_paths:
            print(f" - {p}")
    if per_method_cov_paths:
        print("Saved per-method coverage CSV files:")
        for p in per_method_cov_paths:
            print(f" - {p}")
    print(f"Saved split info to {out_split}")
    print("[RESULT] Compact table (sorted by RMSE):")
    print_compact_summary(summary)
    if not coverage.empty:
        print("[RESULT] Compact coverage table (sorted by tau,f1):")
        print_compact_coverage(coverage)


if __name__ == "__main__":
    main()
