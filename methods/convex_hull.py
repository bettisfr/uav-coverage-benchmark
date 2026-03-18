from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon


@dataclass
class CellModel:
    polygon: Optional[Polygon]


class ConvexHullCoverageModel:
    def __init__(self, min_samples: int = 3) -> None:
        self.min_samples = min_samples
        self.models: Dict[int, CellModel] = {}

    def fit(self, df, cell_col: str, lon_col: str, lat_col: str) -> None:
        self.models = {}
        for cell_id, group in df.groupby(cell_col):
            points = group[[lon_col, lat_col]].dropna().to_numpy()
            poly = self._build_polygon(points)
            self.models[int(cell_id)] = CellModel(polygon=poly)

    def predict_inside(self, cell_id: int, lon: float, lat: float) -> bool:
        model = self.models.get(int(cell_id))
        if model is None or model.polygon is None:
            return False
        return bool(model.polygon.covers(Point(lon, lat)))

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
