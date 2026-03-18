from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None
    ConstantKernel = None
    RBF = None
    WhiteKernel = None


@dataclass
class CellModel:
    gpr: Optional[object]


class GPRCoverageModel:
    def __init__(
        self,
        min_samples: int = 8,
        length_scale: float = 0.001,
        noise_level: float = 1.0,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        max_train: int = 300,
        random_state: int = 42,
    ) -> None:
        self.min_samples = min_samples
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.max_train = max_train
        self.random_state = random_state
        self.models: Dict[int, CellModel] = {}
        self.available = GaussianProcessRegressor is not None

    def _build_model(self):
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=self.length_scale, length_scale_bounds=(1e-5, 1e1))
            + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=(1e-6, 1e2))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
            n_restarts_optimizer=0,
        )

    def fit(self, df, cell_col: str, lon_col: str, lat_col: str, signal_col: str) -> None:
        self.models = {}
        if not self.available:
            return

        rng = np.random.default_rng(self.random_state)
        for cell_id, group in df.groupby(cell_col):
            g = group[[lon_col, lat_col, signal_col]].dropna()
            if len(g) < self.min_samples:
                self.models[int(cell_id)] = CellModel(gpr=None)
                continue

            x = g[[lon_col, lat_col]].to_numpy(dtype=float)
            y = g[signal_col].to_numpy(dtype=float)

            if self.max_train and len(g) > int(self.max_train):
                idx = rng.choice(len(g), size=int(self.max_train), replace=False)
                x = x[idx]
                y = y[idx]

            try:
                model = self._build_model()
                model.fit(x, y)
                self.models[int(cell_id)] = CellModel(gpr=model)
            except Exception:
                self.models[int(cell_id)] = CellModel(gpr=None)

    def predict_signal(self, cell_id: int, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        model = self.models.get(int(cell_id))
        if model is None or model.gpr is None:
            return np.full(shape=len(lons), fill_value=np.nan, dtype=float)
        x = np.column_stack([lons, lats]).astype(float)
        try:
            return np.asarray(model.gpr.predict(x), dtype=float)
        except Exception:
            return np.full(shape=len(lons), fill_value=np.nan, dtype=float)
