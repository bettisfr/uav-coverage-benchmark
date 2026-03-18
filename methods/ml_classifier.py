from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception:  # pragma: no cover
    RandomForestClassifier = None
try:
    from sklearn.ensemble import ExtraTreesClassifier
except Exception:  # pragma: no cover
    ExtraTreesClassifier = None
try:
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # pragma: no cover
    GradientBoostingClassifier = None
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass
class CellModel:
    model: Optional[object]
    constant_class: Optional[int]


class MLCoverageModel:
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
        class_weight: Optional[str] = None,
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
        self.class_weight = class_weight
        self.use_gpu = use_gpu
        self.models: Dict[int, CellModel] = {}
        self.feature_cols = ["lon", "lat"]
        self.available = self._is_backend_available()

    def _is_backend_available(self) -> bool:
        if self.model == "rf":
            return RandomForestClassifier is not None
        if self.model == "et":
            return ExtraTreesClassifier is not None
        if self.model == "gb":
            return GradientBoostingClassifier is not None
        if self.model == "xgb":
            return XGBClassifier is not None
        return False

    def _build_model(self):
        if self.model == "rf":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight=self.class_weight,
            )
        if self.model == "et":
            return ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight=self.class_weight,
            )
        if self.model == "gb":
            return GradientBoostingClassifier(
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
            return XGBClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=(6 if self.max_depth is None else self.max_depth),
                min_child_weight=1,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method=tree_method,
                predictor=predictor,
                random_state=self.random_state,
                n_jobs=-1,
            )
        return None

    def _prepare_x(self, frame, feature_cols):
        xdf = frame[feature_cols].copy()
        for c in feature_cols:
            arr = np.asarray(xdf[c], dtype=float)
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
        label_col: str,
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
                self.models[int(cell_id)] = CellModel(model=None, constant_class=0)
                continue
            g = group[cols + [label_col]].dropna(subset=[label_col])
            y = g[label_col].astype(int).to_numpy()
            x = self._prepare_x(g, self.feature_cols)

            if len(g) < self.min_samples:
                self.models[int(cell_id)] = CellModel(model=None, constant_class=0)
                continue

            uniq = np.unique(y)
            if len(uniq) == 1:
                self.models[int(cell_id)] = CellModel(model=None, constant_class=int(uniq[0]))
                continue

            try:
                clf = self._build_model()
                if clf is None:
                    self.models[int(cell_id)] = CellModel(model=None, constant_class=0)
                    continue
                clf.fit(x, y)
                self.models[int(cell_id)] = CellModel(model=clf, constant_class=None)
            except Exception:
                majority = int(np.round(y.mean()))
                self.models[int(cell_id)] = CellModel(model=None, constant_class=majority)

            if progress_every > 0 and log_fn is not None and (idx % progress_every == 0 or idx == total):
                log_fn(f"[ML] fit progress: {idx}/{total} cells")

    def predict_inside(self, cell_id: int, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        x = np.column_stack([lons, lats])
        return self.predict_on_features(cell_id, x)

    def predict_on_frame(self, cell_id: int, frame) -> np.ndarray:
        model = self.models.get(int(cell_id))
        if model is None:
            return np.zeros(shape=len(frame), dtype=bool)
        if model.model is None:
            const_val = int(model.constant_class or 0)
            return np.full(shape=len(frame), fill_value=bool(const_val), dtype=bool)
        cols = [c for c in self.feature_cols if c in frame.columns]
        if len(cols) != len(self.feature_cols):
            return np.zeros(shape=len(frame), dtype=bool)
        x = self._prepare_x(frame, self.feature_cols)
        pred = model.model.predict(x)
        return np.asarray(pred, dtype=bool)

    def predict_on_features(self, cell_id: int, x: np.ndarray) -> np.ndarray:
        model = self.models.get(int(cell_id))
        if model is None:
            return np.zeros(shape=len(x), dtype=bool)

        if model.model is None:
            const_val = int(model.constant_class or 0)
            return np.full(shape=len(x), fill_value=bool(const_val), dtype=bool)
        pred = model.model.predict(x)
        return np.asarray(pred, dtype=bool)
