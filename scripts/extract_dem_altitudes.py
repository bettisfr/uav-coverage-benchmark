from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_experiments import load_all_data, load_config


@dataclass
class DemRaster:
    path: str
    name: str
    crs: str
    left: float
    right: float
    bottom: float
    top: float
    nodata: float | None


def _load_towers(towers_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(towers_dir, "*.clf")))
    frames = []
    for path in files:
        df = pd.read_csv(path, sep=";", header=None, names=["plmn", "cell_id", "c2", "c3", "lat", "lon", "c6", "name", "c8"])
        df["cell_id"] = pd.to_numeric(df["cell_id"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["cell_id", "lat", "lon"])
        df = df[["cell_id", "lat", "lon", "plmn", "name"]].copy()
        name_u = df["name"].fillna("").astype(str).str.upper()
        df["bs_is_5g"] = name_u.str.contains(r"\b5G\b", regex=True).astype(int)
        df["bs_has_dss"] = name_u.str.contains(r"\bDSS\b", regex=True).astype(int)
        df["bs_band"] = name_u.str.extract(r"\b(B\d{1,2})\b", expand=False)
        df["bs_sector"] = name_u.str.extract(r"\b(S\d{1,2})\b", expand=False)
        df["bs_band_num"] = pd.to_numeric(df["bs_band"].str.replace("B", "", regex=False), errors="coerce")
        df["bs_sector_num"] = pd.to_numeric(df["bs_sector"].str.replace("S", "", regex=False), errors="coerce")
        df["bs_bands_all"] = name_u.apply(
            lambda s: "|".join(dict.fromkeys(re.findall(r"\bB\d{1,2}\b", s))) if re.findall(r"\bB\d{1,2}\b", s) else pd.NA
        )
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=[
                "cell_id",
                "lat",
                "lon",
                "plmn",
                "name",
                "bs_is_5g",
                "bs_has_dss",
                "bs_band",
                "bs_sector",
                "bs_band_num",
                "bs_sector_num",
                "bs_bands_all",
                "source_file",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["cell_id"] = out["cell_id"].astype(int)
    out = out.drop_duplicates(subset=["cell_id"], keep="first").reset_index(drop=True)
    return out


def _discover_rasters(dem_dir: str) -> list[DemRaster]:
    rasters: list[DemRaster] = []
    for path in sorted(glob.glob(os.path.join(dem_dir, "*.tif"))):
        with rasterio.open(path) as ds:
            b = ds.bounds
            rasters.append(
                DemRaster(
                    path=path,
                    name=os.path.basename(path),
                    crs=ds.crs.to_string() if ds.crs else "EPSG:4326",
                    left=float(b.left),
                    right=float(b.right),
                    bottom=float(b.bottom),
                    top=float(b.top),
                    nodata=ds.nodata,
                )
            )
    return rasters


def _sample_dem_for_points(points_df: pd.DataFrame, rasters: Iterable[DemRaster], lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    out = points_df.copy().reset_index(drop=True)
    out["dem_alt_m"] = np.nan
    out["dem_file"] = pd.NA

    unresolved = np.ones(len(out), dtype=bool)
    lats = out[lat_col].to_numpy(dtype=float)
    lons = out[lon_col].to_numpy(dtype=float)

    for r in rasters:
        if not unresolved.any():
            break
        idx_unresolved = np.where(unresolved)[0]
        if len(idx_unresolved) == 0:
            break

        lat_u = lats[idx_unresolved]
        lon_u = lons[idx_unresolved]

        with rasterio.open(r.path) as ds:
            if r.crs != "EPSG:4326":
                x_u, y_u = transform("EPSG:4326", r.crs, lon_u.tolist(), lat_u.tolist())
                x_u = np.asarray(x_u, dtype=float)
                y_u = np.asarray(y_u, dtype=float)
            else:
                x_u = lon_u
                y_u = lat_u

            in_bounds = (x_u >= r.left) & (x_u <= r.right) & (y_u >= r.bottom) & (y_u <= r.top)
            if not in_bounds.any():
                continue

            idx_local = np.where(in_bounds)[0]
            idx_global = idx_unresolved[idx_local]
            coords = list(zip(x_u[idx_local], y_u[idx_local]))
            vals = np.array([v[0] for v in ds.sample(coords)], dtype=float)

            valid = np.isfinite(vals)
            if r.nodata is not None:
                valid &= vals != r.nodata
            valid &= vals > -1000
            if not valid.any():
                continue

            idx_valid_global = idx_global[np.where(valid)[0]]
            out.loc[idx_valid_global, "dem_alt_m"] = vals[np.where(valid)[0]]
            out.loc[idx_valid_global, "dem_file"] = r.name
            unresolved[idx_valid_global] = False

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DEM altitudes for measurements and towers.")
    parser.add_argument("--config", default="configs/default.yaml", help="Benchmark config used for loading measurements.")
    parser.add_argument("--dem-dir", default="../uav-simulator/elevation", help="Directory containing DEM GeoTIFF files.")
    parser.add_argument("--towers-dir", default="data/connectivity/towers", help="Directory containing tower .clf files.")
    parser.add_argument(
        "--bs-scope",
        default="matched",
        choices=["matched", "all"],
        help="matched: only BS observed in measurements; all: all BS in towers files.",
    )
    parser.add_argument("--out-dir", default="data/connectivity/derived", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("[INIT] Loading measurement data...")
    cfg = load_config(args.config)
    meas = load_all_data(cfg)
    print(f"[INIT] Measurements loaded: {len(meas)} rows, cells={meas['cell_id'].nunique()}")

    print("[INIT] Loading towers...")
    towers = _load_towers(args.towers_dir)
    if args.bs_scope == "matched":
        observed_cells = set(meas["cell_id"].astype(int).unique().tolist())
        towers = towers[towers["cell_id"].isin(observed_cells)].copy()
    print(f"[INIT] Towers selected: {len(towers)}")

    print("[INIT] Discovering DEM rasters...")
    rasters = _discover_rasters(args.dem_dir)
    if not rasters:
        raise RuntimeError(f"No .tif files found in {args.dem_dir}")
    print(f"[INIT] DEM rasters found: {len(rasters)}")

    print("[MEAS] Sampling DEM on unique measurement points...")
    meas_pts = meas[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    meas_alt = _sample_dem_for_points(meas_pts, rasters, lat_col="lat", lon_col="lon")
    meas_join = meas.merge(meas_alt, on=["lat", "lon"], how="left")
    meas_out = os.path.join(args.out_dir, "measurement_altitudes_dem.csv")
    meas_join.to_csv(meas_out, index=False)
    cov_meas = float(meas_join["dem_alt_m"].notna().mean() * 100.0)
    print(f"[MEAS] Saved {meas_out} | coverage={cov_meas:.2f}%")

    print("[BS] Sampling DEM on tower points...")
    bs_points = towers.rename(columns={"lat": "bs_lat", "lon": "bs_lon"}).copy()
    bs_alt = _sample_dem_for_points(bs_points, rasters, lat_col="bs_lat", lon_col="bs_lon")
    bs_out = os.path.join(args.out_dir, f"bs_altitudes_dem_{args.bs_scope}.csv")
    bs_alt.to_csv(bs_out, index=False)
    cov_bs = float(bs_alt["dem_alt_m"].notna().mean() * 100.0) if len(bs_alt) else 0.0
    print(f"[BS] Saved {bs_out} | coverage={cov_bs:.2f}%")

    print("[DONE]")


if __name__ == "__main__":
    main()
