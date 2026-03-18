from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    from pykrige.ok import OrdinaryKriging
except Exception:  # pragma: no cover
    OrdinaryKriging = None


@dataclass
class CellModel:
    kriging: Optional[object]


class KrigingCoverageModel:
    def __init__(
        self,
        min_samples: int = 6,
        variogram_model: str = "linear",
        nlags: int = 6,
        jitter_epsilon: float = 1e-2,
    ) -> None:
        self.min_samples = min_samples
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.jitter_epsilon = jitter_epsilon
        self.models: Dict[int, CellModel] = {}
        self.available = OrdinaryKriging is not None

    def fit(self, df, cell_col: str, lon_col: str, lat_col: str, signal_col: str) -> None:
        self.models = {}
        if not self.available:
            return

        for cell_id, group in df.groupby(cell_col):
            g = group[[lon_col, lat_col, signal_col]].dropna()
            if len(g) < self.min_samples:
                self.models[int(cell_id)] = CellModel(kriging=None)
                continue

            lons = g[lon_col].to_numpy(dtype=float)
            lats = g[lat_col].to_numpy(dtype=float)
            signals = g[signal_col].to_numpy(dtype=float)

            if np.allclose(signals, signals[0]):
                signals = signals.copy()
                signals[0] += self.jitter_epsilon

            try:
                ok = OrdinaryKriging(
                    lons,
                    lats,
                    signals,
                    variogram_model=self.variogram_model,
                    nlags=self.nlags,
                    verbose=False,
                    enable_plotting=False,
                )
                self.models[int(cell_id)] = CellModel(kriging=ok)
            except Exception:
                self.models[int(cell_id)] = CellModel(kriging=None)

    def predict_signal(self, cell_id: int, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        model = self.models.get(int(cell_id))
        if model is None or model.kriging is None:
            return np.full(shape=len(lons), fill_value=np.nan, dtype=float)

        try:
            z, _ = model.kriging.execute("points", lons, lats)
            return np.asarray(z, dtype=float)
        except Exception:
            return np.full(shape=len(lons), fill_value=np.nan, dtype=float)
