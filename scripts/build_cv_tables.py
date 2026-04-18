#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


REGRESSOR_FILES: Dict[str, str] = {
    "CH": "metrics_summary_convex_hull.csv",
    "AS": "metrics_summary_alpha_shape.csv",
    "K": "metrics_summary_kriging.csv",
    "IDW": "metrics_summary_idw.csv",
    "GPR": "metrics_summary_gpr.csv",
    "ML": "metrics_summary_ml.csv",
}

COVERAGE_FILES: Dict[str, str] = {
    "CH": "coverage_summary_convex_hull.csv",
    "AS": "coverage_summary_alpha_shape.csv",
    "K": "coverage_summary_kriging.csv",
    "IDW": "coverage_summary_idw.csv",
    "GPR": "coverage_summary_gpr.csv",
    "ML": "coverage_summary_ml.csv",
}


def _resolve_candidates(metrics_dir: Path, base_name: str, run_tag: str = "") -> List[Path]:
    direct = metrics_dir / base_name
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix

    candidates: List[Path] = []
    if direct.exists():
        candidates.append(direct)

    if stem.startswith("coverage_summary_"):
        method = stem[len("coverage_summary_") :]
        tagged = sorted(metrics_dir.glob(f"coverage_summary_*_{method}{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(tagged)
    elif stem.startswith("metrics_summary_"):
        method = stem[len("metrics_summary_") :]
        tagged = sorted(metrics_dir.glob(f"metrics_summary_*_{method}{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
        candidates.extend(tagged)

    if not candidates:
        candidates = sorted(metrics_dir.glob(f"{stem}_*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        # Keep newest first and remove duplicates while preserving order.
        candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    if run_tag:
        candidates = [p for p in candidates if run_tag in p.name]

    return candidates


def _resolve_file(metrics_dir: Path, base_name: str, run_tag: str = "") -> Path:
    candidates = _resolve_candidates(metrics_dir, base_name, run_tag=run_tag)
    if candidates:
        return candidates[0]
    direct = metrics_dir / base_name
    tag_note = f" for run_tag={run_tag}" if run_tag else ""
    raise FileNotFoundError(f"Missing file: {direct} (and no tagged fallback found{tag_note})")


def _fmt(df: pd.DataFrame, digits: int) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if c != "Met."]
    for c in num_cols:
        out[c] = out[c].astype(float).round(digits)
    return out


def build_regressor_table(metrics_dir: Path, digits: int, run_tag: str = "") -> pd.DataFrame:
    rows: List[dict] = []
    for label, filename in REGRESSOR_FILES.items():
        path = _resolve_file(metrics_dir, filename, run_tag=run_tag)
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Empty file: {path}")
        rows.append(
            {
                "Met.": label,
                "MAE": df["mae"].mean(),
                "RMSE": df["rmse"].mean(),
                "R2": df["r2"].mean(),
                "Pearson": df["pearson"].mean(),
                "Spearman": df["spearman"].mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return _fmt(out, digits)


def build_coverage_table(metrics_dir: Path, threshold: float, digits: int, run_tag: str = "") -> pd.DataFrame:
    rows: List[dict] = []
    for label, filename in COVERAGE_FILES.items():
        picked_path = None
        picked_rows = None
        for path in _resolve_candidates(metrics_dir, filename, run_tag=run_tag):
            df = pd.read_csv(path)
            if df.empty or "threshold_dbm" not in df.columns:
                continue
            d = df[df["threshold_dbm"] == threshold]
            if not d.empty:
                picked_path = path
                picked_rows = d
                break
        if picked_rows is None:
            path0 = _resolve_file(metrics_dir, filename, run_tag=run_tag)
            df0 = pd.read_csv(path0)
            ths = sorted(df0["threshold_dbm"].dropna().unique().tolist()) if "threshold_dbm" in df0.columns else []
            raise ValueError(
                f"No rows with threshold_dbm={threshold} for method {label}. "
                f"Latest file checked: {path0}. Available thresholds there: {ths}"
            )
        rows.append(
            {
                "Met.": label,
                "Prec": picked_rows["precision"].mean(),
                "Rec": picked_rows["recall"].mean(),
                "F1": picked_rows["f1"].mean(),
                "Acc": picked_rows["accuracy"].mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return _fmt(out, digits)


def _to_text(df: pd.DataFrame, fmt: str) -> str:
    if fmt == "plain":
        return df.to_string(index=False)
    if fmt == "latex":
        return df.to_latex(index=False, escape=False)
    if fmt == "csv":
        return df.to_csv(index=False)
    raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CV-aggregated regressor/coverage tables from results/metrics CSV files."
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/metrics",
        help="Directory containing metrics_summary_*.csv and coverage_summary_*.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-100.0,
        help="Coverage threshold (dBm) for the derived classification table.",
    )
    parser.add_argument(
        "--mode",
        choices=["regressor", "coverage", "both"],
        default="both",
        help="Which table(s) to generate.",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "latex", "csv"],
        default="plain",
        help="Output format.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal digits for numeric values.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional tag filter to force consistent file selection (e.g., CV5_TAU100).",
    )
    parser.add_argument(
        "--out-reg",
        default="",
        help="Optional output path for regressor table.",
    )
    parser.add_argument(
        "--out-cov",
        default="",
        help="Optional output path for coverage table.",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)

    if args.mode in ("regressor", "both"):
        reg = build_regressor_table(metrics_dir, args.digits, run_tag=args.run_tag)
        reg_text = _to_text(reg, args.format)
        if args.out_reg:
            Path(args.out_reg).write_text(reg_text + ("\n" if not reg_text.endswith("\n") else ""), encoding="utf-8")
        else:
            print("Regressor (signal prediction)")
            print(reg_text)
            if args.mode == "both":
                print()

    if args.mode in ("coverage", "both"):
        cov = build_coverage_table(metrics_dir, args.threshold, args.digits, run_tag=args.run_tag)
        cov_text = _to_text(cov, args.format)
        if args.out_cov:
            Path(args.out_cov).write_text(cov_text + ("\n" if not cov_text.endswith("\n") else ""), encoding="utf-8")
        else:
            print(f"Derived classifier (tau = {args.threshold:.0f} dBm)")
            print(cov_text)


if __name__ == "__main__":
    main()
