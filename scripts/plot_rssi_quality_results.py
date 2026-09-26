#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


METHOD_FILES: Dict[str, str] = {
    "CH": "convex_hull",
    "AS": "alpha_shape",
    "K": "kriging",
    "IDW": "idw",
    "GPR": "gpr",
    "ML": "ml",
}

CLASS_ORDER = ["very_poor", "poor", "fair", "good", "excellent"]


def _resolve_file(metrics_dir: Path, stem: str, suffix: str, run_tag: str) -> Path | None:
    tagged = metrics_dir / f"{stem}_{run_tag}_{suffix}.csv"
    if tagged.exists():
        return tagged
    fallback = metrics_dir / f"{stem}_{suffix}.csv"
    if fallback.exists():
        return fallback
    return None


def load_quality_summary(metrics_dir: Path, run_tag: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for label, suffix in METHOD_FILES.items():
        path = _resolve_file(metrics_dir, "quality_summary", suffix, run_tag)
        if path is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["method_label"] = label
        rows.append(df)
    if not rows:
        raise ValueError("No quality_summary rows found.")
    out = pd.concat(rows, ignore_index=True)
    out["method_label"] = pd.Categorical(out["method_label"], categories=list(METHOD_FILES.keys()), ordered=True)
    return out.sort_values("method_label")


def load_quality_confusion(metrics_dir: Path, run_tag: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for label, suffix in METHOD_FILES.items():
        path = _resolve_file(metrics_dir, "quality_confusion", suffix, run_tag)
        if path is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["method_label"] = label
        rows.append(df)
    if not rows:
        raise ValueError("No quality_confusion rows found.")
    return pd.concat(rows, ignore_index=True)


def plot_quality_bar(summary_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    cols = [
        ("quality_macro_f1", "Macro-F1"),
        ("quality_bal_acc", "Bal. Acc"),
        ("quality_qwk", "QWK"),
    ]
    methods = summary_df["method_label"].astype(str).tolist()
    x = np.arange(len(methods))
    w = 0.24

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (c, name) in enumerate(cols):
        y = pd.to_numeric(summary_df[c], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (i - 1) * w, y, width=w, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("RSSI Quality Metrics by Method", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(ncol=3, frameon=False, loc="upper center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_confusion_heatmaps(cm_df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    methods = [m for m in METHOD_FILES.keys() if m in set(cm_df["method_label"].astype(str).unique())]
    n = len(methods)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    for idx, m in enumerate(methods):
        ax = axes[idx // ncols, idx % ncols]
        d = cm_df[cm_df["method_label"] == m].copy()
        mat = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=float)
        for _, r in d.iterrows():
            ti = CLASS_ORDER.index(str(r["true_class_name"]))
            pi = CLASS_ORDER.index(str(r["pred_class_name"]))
            mat[ti, pi] += float(r["count"])
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        matn = mat / row_sum
        im = ax.imshow(matn, vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_title(m)
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_yticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels(CLASS_ORDER, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_ORDER, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for j in range(n, nrows * ncols):
        ax = axes[j // ncols, j % ncols]
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("Row-normalized count")
    fig.suptitle("RSSI Quality Confusion Matrices (Row-normalized)", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot RSSI quality metrics and confusion matrices.")
    ap.add_argument("--metrics-dir", default="results/metrics")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--out-dir", default="results/plots")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_quality_summary(metrics_dir, args.run_tag)
    cm = load_quality_confusion(metrics_dir, args.run_tag)

    out_bar_png = out_dir / f"quality_metrics_bar_{args.run_tag or 'latest'}.png"
    out_bar_pdf = out_dir / f"quality_metrics_bar_{args.run_tag or 'latest'}.pdf"
    out_cm_png = out_dir / f"quality_confusion_heatmaps_{args.run_tag or 'latest'}.png"
    out_cm_pdf = out_dir / f"quality_confusion_heatmaps_{args.run_tag or 'latest'}.pdf"

    plot_quality_bar(summary, out_bar_png, out_bar_pdf)
    plot_confusion_heatmaps(cm, out_cm_png, out_cm_pdf)

    print(f"saved {out_bar_png}")
    print(f"saved {out_bar_pdf}")
    print(f"saved {out_cm_png}")
    print(f"saved {out_cm_pdf}")


if __name__ == "__main__":
    main()
