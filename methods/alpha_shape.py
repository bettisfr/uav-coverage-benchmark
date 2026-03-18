from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString, MultiPoint, Point, Polygon
from shapely.ops import polygonize, unary_union


@dataclass
class CellModel:
    polygon: Optional[Polygon]


class AlphaShapeCoverageModel:
    def __init__(self, min_samples: int = 4, alpha: float = 150.0) -> None:
        self.min_samples = min_samples
        self.alpha = alpha
        self.models: Dict[int, CellModel] = {}

    def fit(self, df, cell_col: str, lon_col: str, lat_col: str) -> None:
        self.models = {}
        for cell_id, group in df.groupby(cell_col):
            pts = group[[lon_col, lat_col]].dropna().to_numpy()
            poly = self._build_alpha_shape(pts)
            self.models[int(cell_id)] = CellModel(polygon=poly)

    def predict_inside(self, cell_id: int, lon: float, lat: float) -> bool:
        model = self.models.get(int(cell_id))
        if model is None or model.polygon is None:
            return False
        return bool(model.polygon.covers(Point(lon, lat)))

    def _build_alpha_shape(self, points: np.ndarray) -> Optional[Polygon]:
        if len(points) < self.min_samples:
            return None

        uniq = np.unique(points, axis=0)
        if len(uniq) < self.min_samples:
            return None

        if len(uniq) < 4:
            hull = MultiPoint([tuple(p) for p in uniq]).convex_hull
            return hull if hull.geom_type == "Polygon" and hull.area > 0 else None

        try:
            tri = Delaunay(uniq)
        except Exception:
            hull = MultiPoint([tuple(p) for p in uniq]).convex_hull
            return hull if hull.geom_type == "Polygon" and hull.area > 0 else None

        edges = set()
        edge_points = []

        for ia, ib, ic in tri.simplices:
            pa = uniq[ia]
            pb = uniq[ib]
            pc = uniq[ic]

            a = np.linalg.norm(pa - pb)
            b = np.linalg.norm(pb - pc)
            c = np.linalg.norm(pc - pa)
            s = (a + b + c) / 2.0
            area_sq = s * (s - a) * (s - b) * (s - c)
            if area_sq <= 0:
                continue
            area = np.sqrt(area_sq)
            circum_r = (a * b * c) / (4.0 * area)

            if circum_r <= self.alpha:
                for i, j in ((ia, ib), (ib, ic), (ic, ia)):
                    if (i, j) in edges or (j, i) in edges:
                        continue
                    edges.add((i, j))
                    edge_points.append((tuple(uniq[i]), tuple(uniq[j])))

        if not edge_points:
            hull = MultiPoint([tuple(p) for p in uniq]).convex_hull
            return hull if hull.geom_type == "Polygon" and hull.area > 0 else None

        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        if not triangles:
            return None

        poly = unary_union(triangles)
        if poly.geom_type == "Polygon":
            return poly if poly.area > 0 else None
        if poly.geom_type == "MultiPolygon":
            largest = max(poly.geoms, key=lambda g: g.area)
            return largest if largest.area > 0 else None
        return None
