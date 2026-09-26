#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


METHOD_FILES: Dict[str, str] = {
    "CH": "convex_hull",
    "AS": "alpha_shape",
    "K": "kriging",
    "IDW": "idw",
    "GPR": "gpr",
    "ML": "ml",
}


def _resolve(metrics_dir: Path, stem: str, suffix: str, run_tag: str) -> Path | None:
    tagged = metrics_dir / f"{stem}_{run_tag}_{suffix}.csv" if run_tag else None
    if tagged and tagged.exists():
        return tagged
    base = metrics_dir / f"{stem}_{suffix}.csv"
    return base if base.exists() else None


def load_quality(metrics_dir: Path, run_tag: str, digits: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for m, s in METHOD_FILES.items():
        p = _resolve(metrics_dir, "quality_summary", s, run_tag)
        if p is None:
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        d = df.iloc[[0]].copy()
        d["Met."] = m
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[["Met.", "quality_macro_f1", "quality_weighted_f1", "quality_bal_acc", "quality_qwk", "quality_mae_class"]]
    out = out.rename(
        columns={
            "quality_macro_f1": "MacroF1",
            "quality_weighted_f1": "WeightedF1",
            "quality_bal_acc": "BalAcc",
            "quality_qwk": "QWK",
            "quality_mae_class": "MAEcls",
        }
    )
    for c in ["MacroF1", "WeightedF1", "BalAcc", "QWK", "MAEcls"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(digits)
    return out.sort_values("Met.").reset_index(drop=True)


def load_binary(metrics_dir: Path, run_tag: str, tau: float, digits: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for m, s in METHOD_FILES.items():
        p = _resolve(metrics_dir, "coverage_summary", s, run_tag)
        if p is None:
            continue
        df = pd.read_csv(p)
        if df.empty or "threshold_dbm" not in df.columns:
            continue
        d = df[df["threshold_dbm"] == float(tau)]
        if d.empty:
            continue
        d = d.iloc[[0]].copy()
        d["Met."] = m
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[["Met.", "precision", "recall", "f1", "accuracy"]]
    out = out.rename(columns={"precision": "Prec", "recall": "Rec", "f1": "F1", "accuracy": "Acc"})
    for c in ["Prec", "Rec", "F1", "Acc"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(digits)
    return out.sort_values("Met.").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Primary (RSSI quality) + secondary (binary tau) tables.")
    ap.add_argument("--metrics-dir", default="results/metrics")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--tau", type=float, default=-95.0)
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--format", choices=["plain", "csv"], default="plain")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    q = load_quality(metrics_dir, args.run_tag, args.digits)
    b = load_binary(metrics_dir, args.run_tag, args.tau, args.digits)

    if args.format == "csv":
        if not q.empty:
            print("# primary_quality")
            print(q.to_csv(index=False).strip())
        if not b.empty:
            print("# secondary_binary")
            print(b.to_csv(index=False).strip())
        return

    print("Primary task: RSSI quality (ordinal, 5 classes)")
    print(q.to_string(index=False) if not q.empty else "No quality rows.")
    print("")
    print(f"Secondary analysis: binary thresholding (tau = {args.tau:.0f} dBm)")
    print(b.to_string(index=False) if not b.empty else "No binary rows for selected tau.")


if __name__ == "__main__":
    main()

