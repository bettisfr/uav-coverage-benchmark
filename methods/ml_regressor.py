from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesRegressor
except Exception:  # pragma: no cover
    ExtraTreesRegressor = None
try:
    from sklearn.ensemble import GradientBoostingRegressor
except Exception:  # pragma: no cover
    GradientBoostingRegressor = None
try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover
    RandomForestRegressor = None
try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


@dataclass
class CellModel:
    model: Optional[object]
    constant_value: Optional[float]
    feature_medians: Dict[str, float]


class MLSignalModel:
    def __init__(
        self,
        min_samples: int = 20,
        model: str = "rf",
        random_state: int = 42,
        n_estimators: int = 120,
        max_depth: Optional[int] = 12,
        min_samples_leaf: int = 3,
        min_samples_split: int = 2,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        use_gpu: bool = False,
    ) -> None:
        self.min_samples = min_samples
        self.model = model.lower()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_gpu = use_gpu
        self.models: Dict[int, CellModel] = {}
        self.feature_cols = ["lon", "lat"]
        self.available = self._is_backend_available()

    def _is_backend_available(self) -> bool:
        if self.model == "rf":
            return RandomForestRegressor is not None
        if self.model == "et":
            return ExtraTreesRegressor is not None
        if self.model == "gb":
            return GradientBoostingRegressor is not None
        if self.model == "xgb":
            return XGBRegressor is not None
        return False

    def _build_model(self):
        if self.model == "rf":
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if self.model == "et":
            return ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            )
        if self.model == "gb":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                subsample=self.subsample,
                random_state=self.random_state,
            )
        if self.model == "xgb":
            tree_method = "gpu_hist" if self.use_gpu else "hist"
            predictor = "gpu_predictor" if self.use_gpu else "cpu_predictor"
            return XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=(6 if self.max_depth is None else self.max_depth),
                min_child_weight=1,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="reg:squarederror",
                tree_method=tree_method,
                predictor=predictor,
                random_state=self.random_state,
                n_jobs=-1,
            )
        return None

    def _prepare_x(self, frame, feature_cols, feature_medians: Optional[Dict[str, float]] = None):
        xdf = frame[feature_cols].copy()
        for c in feature_cols:
            arr = pd.to_numeric(xdf[c], errors="coerce").to_numpy(dtype=float)
            if feature_medians is not None and c in feature_medians:
                med = float(feature_medians[c])
            else:
                finite = arr[np.isfinite(arr)]
                med = float(np.median(finite)) if finite.size > 0 else 0.0
            xdf[c] = pd.Series(arr).fillna(med)
        return xdf.to_numpy(dtype=float)

    def fit(
        self,
        df,
        cell_col: str,
        lon_col: str,
        lat_col: str,
        signal_col: str,
        feature_cols=None,
        progress_every: int = 0,
        log_fn=None,
    ) -> None:
        self.models = {}
        if not self.available:
            return
        self.feature_cols = list(feature_cols or [lon_col, lat_col])

        groups = list(df.groupby(cell_col))
        total = len(groups)
        for idx, (cell_id, group) in enumerate(groups, start=1):
            cols = [c for c in self.feature_cols if c in group.columns]
            if len(cols) != len(self.feature_cols):
                self.models[int(cell_id)] = CellModel(model=None, constant_value=float("-inf"), feature_medians={})
                continue

            # Keep rows with missing features (they are imputed in _prepare_x),
            # drop only rows with missing target signal.
            g = group[cols + [signal_col]].dropna(subset=[signal_col])
            y = pd.to_numeric(g[signal_col], errors="coerce").to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            feature_medians = {}
            for c in cols:
                vc = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
                finite = vc[np.isfinite(vc)]
                feature_medians[c] = float(np.median(finite)) if finite.size > 0 else 0.0
            x = self._prepare_x(g, self.feature_cols, feature_medians=feature_medians)

            if len(g) < self.min_samples:
                const = float(np.mean(y)) if len(y) > 0 else float("-inf")
                self.models[int(cell_id)] = CellModel(model=None, constant_value=const, feature_medians=feature_medians)
                continue

            try:
                reg = self._build_model()
                if reg is None:
                    const = float(np.mean(y)) if len(y) > 0 else float("-inf")
                    self.models[int(cell_id)] = CellModel(model=None, constant_value=const, feature_medians=feature_medians)
                    continue
                reg.fit(x, y)
                self.models[int(cell_id)] = CellModel(model=reg, constant_value=None, feature_medians=feature_medians)
            except Exception:
                const = float(np.mean(y)) if len(y) > 0 else float("-inf")
                self.models[int(cell_id)] = CellModel(model=None, constant_value=const, feature_medians=feature_medians)

            if progress_every > 0 and log_fn is not None and (idx % progress_every == 0 or idx == total):
                log_fn(f"[ML-REG] fit progress: {idx}/{total} cells")

    def predict_on_frame(self, cell_id: int, frame) -> np.ndarray:
        model = self.models.get(int(cell_id))
        if model is None:
            return np.full(shape=len(frame), fill_value=float("-inf"), dtype=float)
        if model.model is None:
            const_val = float(model.constant_value if model.constant_value is not None else float("-inf"))
            return np.full(shape=len(frame), fill_value=const_val, dtype=float)
        cols = [c for c in self.feature_cols if c in frame.columns]
        if len(cols) != len(self.feature_cols):
            return np.full(shape=len(frame), fill_value=float("-inf"), dtype=float)
        x = self._prepare_x(frame, self.feature_cols, feature_medians=model.feature_medians)
        pred = model.model.predict(x)
        return np.asarray(pred, dtype=float)
