#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


METHOD_FILES: Dict[str, str] = {
    "CH": "metrics_summary_convex_hull.csv",
    "AS": "metrics_summary_alpha_shape.csv",
    "K": "metrics_summary_kriging.csv",
    "IDW": "metrics_summary_idw.csv",
    "GPR": "metrics_summary_gpr.csv",
    "ML": "metrics_summary_ml.csv",
}


def _pick_best_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty metrics file: {csv_path}")
    if "rmse" in df.columns:
        return df.sort_values("rmse", ascending=True).iloc[0]
    return df.iloc[0]


def build_table(metrics_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for label, filename in METHOD_FILES.items():
        path = metrics_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        row = _pick_best_row(path)
        rows.append(
            {
                "Met.": label,
                "MAE": float(row.get("mae", float("nan"))),
                "RMSE": float(row.get("rmse", float("nan"))),
                "R2": float(row.get("r2", float("nan"))),
                "Pearson": float(row.get("pearson", float("nan"))),
                "Spearman": float(row.get("spearman", float("nan"))),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("RMSE", ascending=True).reset_index(drop=True)
    for c in ["MAE", "RMSE", "R2", "Pearson", "Spearman"]:
        out[c] = out[c].round(2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final regressor metrics table from per-method summary CSV files."
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/metrics",
        help="Directory containing metrics_summary_<method>.csv files.",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "latex", "csv"],
        default="plain",
        help="Output format.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output file path. If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    table = build_table(Path(args.metrics_dir))

    if args.format == "plain":
        text = table.to_string(index=False)
    elif args.format == "latex":
        text = table.to_latex(index=False, escape=False)
    else:
        text = table.to_csv(index=False)

    if args.out:
        Path(args.out).write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

