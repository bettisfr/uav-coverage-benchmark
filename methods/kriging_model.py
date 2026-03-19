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
    center_lon: float
    center_lat: float
    mean_signal: float


class KrigingCoverageModel:
    def __init__(
        self,
        min_samples: int = 6,
        variogram_model: str = "spherical",
        nlags: int = 10,
        jitter_epsilon: float = 1e-2,
        min_signal_dbm: float = -150.0,
        max_signal_dbm: float = -30.0,
    ) -> None:
        self.min_samples = min_samples
        self.variogram_model = variogram_model
        self.nlags = nlags
        self.jitter_epsilon = jitter_epsilon
        self.min_signal_dbm = min_signal_dbm
        self.max_signal_dbm = max_signal_dbm
        self.models: Dict[int, CellModel] = {}
        self.available = OrdinaryKriging is not None

    @staticmethod
    def _lonlat_to_local_m(lons: np.ndarray, lats: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
        # Local equirectangular projection (meters) centered at (lon0, lat0).
        r = 6371000.0
        lon = np.radians(lons.astype(float))
        lat = np.radians(lats.astype(float))
        lon0r = np.radians(float(lon0))
        lat0r = np.radians(float(lat0))
        x = (lon - lon0r) * np.cos(lat0r) * r
        y = (lat - lat0r) * r
        return x, y

    def fit(self, df, cell_col: str, lon_col: str, lat_col: str, signal_col: str) -> None:
        self.models = {}
        if not self.available:
            return

        for cell_id, group in df.groupby(cell_col):
            g = group[[lon_col, lat_col, signal_col]].dropna()
            if g.empty:
                self.models[int(cell_id)] = CellModel(
                    kriging=None,
                    center_lon=float("nan"),
                    center_lat=float("nan"),
                    mean_signal=float("nan"),
                )
                continue

            mean_signal = float(np.mean(g[signal_col].to_numpy(dtype=float)))
            center_lon = float(np.mean(g[lon_col].to_numpy(dtype=float)))
            center_lat = float(np.mean(g[lat_col].to_numpy(dtype=float)))

            if len(g) < self.min_samples:
                self.models[int(cell_id)] = CellModel(
                    kriging=None,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    mean_signal=mean_signal,
                )
                continue

            lons = g[lon_col].to_numpy(dtype=float)
            lats = g[lat_col].to_numpy(dtype=float)
            signals = g[signal_col].to_numpy(dtype=float)
            x_m, y_m = self._lonlat_to_local_m(lons, lats, center_lon, center_lat)

            if np.allclose(signals, signals[0]):
                signals = signals.copy()
                signals[0] += self.jitter_epsilon

            try:
                ok = OrdinaryKriging(
                    x_m,
                    y_m,
                    signals,
                    variogram_model=self.variogram_model,
                    nlags=self.nlags,
                    verbose=False,
                    enable_plotting=False,
                )
                self.models[int(cell_id)] = CellModel(
                    kriging=ok,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    mean_signal=mean_signal,
                )
            except Exception:
                self.models[int(cell_id)] = CellModel(
                    kriging=None,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    mean_signal=mean_signal,
                )

    def predict_signal(self, cell_id: int, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        model = self.models.get(int(cell_id))
        if model is None:
            return np.full(shape=len(lons), fill_value=np.nan, dtype=float)
        if model.kriging is None:
            return np.full(shape=len(lons), fill_value=float(model.mean_signal), dtype=float)

        try:
            x_m, y_m = self._lonlat_to_local_m(
                np.asarray(lons, dtype=float),
                np.asarray(lats, dtype=float),
                model.center_lon,
                model.center_lat,
            )
            z, _ = model.kriging.execute("points", x_m, y_m)
            pred = np.asarray(z, dtype=float)
            return np.clip(pred, self.min_signal_dbm, self.max_signal_dbm)
        except Exception:
            return np.full(shape=len(lons), fill_value=float(model.mean_signal), dtype=float)
