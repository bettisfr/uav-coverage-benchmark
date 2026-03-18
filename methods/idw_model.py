from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class CellModel:
    lons: Optional[np.ndarray]
    lats: Optional[np.ndarray]
    signals: Optional[np.ndarray]


class IDWCoverageModel:
    def __init__(
        self,
        min_samples: int = 6,
        k_neighbors: int = 12,
        power: float = 2.0,
        epsilon: float = 1e-12,
    ) -> None:
        self.min_samples = min_samples
        self.k_neighbors = k_neighbors
        self.power = power
        self.epsilon = epsilon
        self.models: Dict[int, CellModel] = {}
        self.available = True

    def fit(self, df, cell_col: str, lon_col: str, lat_col: str, signal_col: str) -> None:
        self.models = {}
        for cell_id, group in df.groupby(cell_col):
            g = group[[lon_col, lat_col, signal_col]].dropna()
            if len(g) < self.min_samples:
                self.models[int(cell_id)] = CellModel(lons=None, lats=None, signals=None)
                continue

            self.models[int(cell_id)] = CellModel(
                lons=g[lon_col].to_numpy(dtype=float),
                lats=g[lat_col].to_numpy(dtype=float),
                signals=g[signal_col].to_numpy(dtype=float),
            )

    def predict_signal(self, cell_id: int, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        model = self.models.get(int(cell_id))
        if model is None or model.lons is None or model.lats is None or model.signals is None:
            return np.full(shape=len(lons), fill_value=np.nan, dtype=float)

        train_lon = model.lons
        train_lat = model.lats
        train_sig = model.signals
        n_train = len(train_sig)
        if n_train == 0:
            return np.full(shape=len(lons), fill_value=np.nan, dtype=float)

        k = int(max(1, min(self.k_neighbors, n_train)))
        pred = np.empty(shape=len(lons), dtype=float)

        for i, (lon_q, lat_q) in enumerate(zip(lons, lats)):
            dx = train_lon - float(lon_q)
            dy = train_lat - float(lat_q)
            dist = np.hypot(dx, dy)

            zero_idx = np.where(dist <= self.epsilon)[0]
            if len(zero_idx) > 0:
                pred[i] = float(np.mean(train_sig[zero_idx]))
                continue

            if k < n_train:
                nn_idx = np.argpartition(dist, k - 1)[:k]
            else:
                nn_idx = np.arange(n_train)

            d = dist[nn_idx]
            s = train_sig[nn_idx]
            w = 1.0 / np.power(d + self.epsilon, self.power)
            w_sum = float(np.sum(w))
            if w_sum <= 0:
                pred[i] = np.nan
            else:
                pred[i] = float(np.dot(w, s) / w_sum)

        return pred
