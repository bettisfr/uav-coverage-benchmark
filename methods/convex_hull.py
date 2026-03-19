from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon


@dataclass
class CellModel:
    polygon: Optional[Polygon]
    beta0: float
    beta1: float
    signal_mean: float
    feature_mode: str
    ref_lon: float
    ref_lat: float


class ConvexHullCoverageModel:
    def __init__(self, min_samples: int = 3) -> None:
        self.min_samples = min_samples
        self.models: Dict[int, CellModel] = {}

    @staticmethod
    def _haversine_m(lon1: np.ndarray, lat1: np.ndarray, lon2: float, lat2: float) -> np.ndarray:
        r = 6371000.0
        lon1r = np.radians(lon1)
        lat1r = np.radians(lat1)
        lon2r = np.radians(lon2)
        lat2r = np.radians(lat2)
        dlon = lon2r - lon1r
        dlat = lat2r - lat1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2 or np.nanstd(x) <= 1e-12:
            return float(np.nanmean(y)), 0.0
        a = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(a, y, rcond=None)
        return float(beta[0]), float(beta[1])

    def fit(
        self,
        df,
        cell_col: str,
        lon_col: str,
        lat_col: str,
        signal_col: str,
        dist_col: str = "dist_bs_m",
        bs_lon_col: str = "bs_lon",
        bs_lat_col: str = "bs_lat",
    ) -> None:
        self.models = {}
        for cell_id, group in df.groupby(cell_col):
            points = group[[lon_col, lat_col]].dropna().to_numpy()
            poly = self._build_polygon(points)
            gd = group[[lon_col, lat_col, signal_col]].dropna()
            if gd.empty:
                self.models[int(cell_id)] = CellModel(
                    polygon=poly,
                    beta0=float("-inf"),
                    beta1=0.0,
                    signal_mean=float("-inf"),
                    feature_mode="none",
                    ref_lon=float("nan"),
                    ref_lat=float("nan"),
                )
                continue

            sig = gd[signal_col].to_numpy(dtype=float)
            signal_mean = float(np.mean(sig))

            feature_mode = "constant"
            ref_lon = float(np.mean(gd[lon_col].to_numpy(dtype=float)))
            ref_lat = float(np.mean(gd[lat_col].to_numpy(dtype=float)))
            feat = None

            if bs_lon_col in group.columns and bs_lat_col in group.columns:
                bsl = group[[bs_lon_col, bs_lat_col]].dropna()
                if not bsl.empty:
                    ref_lon = float(bsl.iloc[0][bs_lon_col])
                    ref_lat = float(bsl.iloc[0][bs_lat_col])
                    if dist_col in group.columns:
                        gdist = group[[lon_col, lat_col, signal_col, dist_col]].dropna()
                        if len(gdist) >= 2 and np.nanstd(gdist[dist_col].to_numpy(dtype=float)) > 1e-12:
                            gd = gdist[[lon_col, lat_col, signal_col]]
                            sig = gd[signal_col].to_numpy(dtype=float)
                            feat = gdist[dist_col].to_numpy(dtype=float)
                            feature_mode = "dist_col"
                    if feat is None:
                        lonv = gd[lon_col].to_numpy(dtype=float)
                        latv = gd[lat_col].to_numpy(dtype=float)
                        d = self._haversine_m(lonv, latv, ref_lon, ref_lat)
                        if len(d) >= 2 and np.nanstd(d) > 1e-12:
                            feat = d
                            feature_mode = "bs_distance"

            if feat is None:
                lonv = gd[lon_col].to_numpy(dtype=float)
                latv = gd[lat_col].to_numpy(dtype=float)
                d = self._haversine_m(lonv, latv, ref_lon, ref_lat)
                if len(d) >= 2 and np.nanstd(d) > 1e-12:
                    feat = d
                    feature_mode = "centroid_distance"

            if feat is None:
                beta0, beta1 = signal_mean, 0.0
            else:
                beta0, beta1 = self._fit_linear(feat, sig)

            self.models[int(cell_id)] = CellModel(
                polygon=poly,
                beta0=beta0,
                beta1=beta1,
                signal_mean=signal_mean,
                feature_mode=feature_mode,
                ref_lon=ref_lon,
                ref_lat=ref_lat,
            )

    def predict_inside(self, cell_id: int, lon: float, lat: float) -> bool:
        model = self.models.get(int(cell_id))
        if model is None or model.polygon is None:
            return False
        return bool(model.polygon.covers(Point(lon, lat)))

    def predict_signal(self, cell_id: int, lon: float, lat: float) -> float:
        model = self.models.get(int(cell_id))
        if model is None or model.polygon is None:
            return float("-inf")
        if not bool(model.polygon.covers(Point(lon, lat))):
            return float("-inf")
        if model.feature_mode in {"bs_distance", "centroid_distance"}:
            d = float(self._haversine_m(np.array([lon], dtype=float), np.array([lat], dtype=float), model.ref_lon, model.ref_lat)[0])
            return float(model.beta0 + model.beta1 * d)
        if model.feature_mode == "dist_col":
            d = float(self._haversine_m(np.array([lon], dtype=float), np.array([lat], dtype=float), model.ref_lon, model.ref_lat)[0])
            return float(model.beta0 + model.beta1 * d)
        return float(model.signal_mean)

    def _build_polygon(self, points: np.ndarray) -> Optional[Polygon]:
        if len(points) < self.min_samples:
            return None

        uniq = np.unique(points, axis=0)
        if len(uniq) < self.min_samples:
            return None

        try:
            hull = ConvexHull(uniq)
            hull_points = uniq[hull.vertices]
            poly = Polygon(hull_points)
            if not poly.is_valid or poly.area == 0:
                return None
            return poly
        except Exception:
            mp = MultiPoint([(x, y) for x, y in uniq])
            poly = mp.convex_hull
            if poly.geom_type != "Polygon" or poly.area == 0:
                return None
            return poly
