from __future__ import annotations

import glob
import os
from pathlib import Path

import folium
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_new_trace(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in ["lat", "lon"] if c in df.columns]
    if len(keep) < 2:
        return pd.DataFrame(columns=["lat", "lon"])
    out = df[["lat", "lon"]].copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    out = out[(out["lat"] >= -90) & (out["lat"] <= 90) & (out["lon"] >= -180) & (out["lon"] <= 180)]
    return out


def _load_old_trace(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["lat", "lon"])
    out = pd.DataFrame({"lat": pd.to_numeric(df.iloc[:, 0], errors="coerce"), "lon": pd.to_numeric(df.iloc[:, 1], errors="coerce")})
    out = out.dropna(subset=["lat", "lon"])
    out = out[(out["lat"] >= -90) & (out["lat"] <= 90) & (out["lon"] >= -180) & (out["lon"] <= 180)]
    return out


def _load_bs_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {}
    if "bs_lat" in df.columns:
        cols["bs_lat"] = "lat"
    elif "lat" in df.columns:
        cols["lat"] = "lat"
    if "bs_lon" in df.columns:
        cols["bs_lon"] = "lon"
    elif "lon" in df.columns:
        cols["lon"] = "lon"
    if "cell_id" not in df.columns or len(cols) < 2:
        return pd.DataFrame(columns=["cell_id", "lat", "lon"])
    out = df[["cell_id", *cols.keys()]].rename(columns=cols).copy()
    out["cell_id"] = pd.to_numeric(out["cell_id"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["cell_id", "lat", "lon"])
    out = out[(out["lat"] >= -90) & (out["lat"] <= 90) & (out["lon"] >= -180) & (out["lon"] <= 180)]
    out["cell_id"] = out["cell_id"].astype(int)
    out = out.drop_duplicates(subset=["cell_id"])
    return out


def main() -> None:
    dataset_dir = ROOT / "data" / "connectivity" / "dataset"
    dataset_old_dir = ROOT / "data" / "connectivity" / "dataset-old"
    drone_dir = ROOT / "data" / "connectivity" / "drone"
    bs_csv = ROOT / "data" / "connectivity" / "derived" / "bs_altitudes_dem_matched.csv"
    out_html = ROOT / "results" / "dataset" / "traces_bs_drone_map.html"

    ground_files = sorted(glob.glob(str(dataset_dir / "*.csv")))
    for mode in ["bike", "car", "train", "walk"]:
        ground_files.extend(sorted(glob.glob(str(dataset_old_dir / mode / "*.csv"))))
    drone_files = sorted(glob.glob(str(drone_dir / "*.csv")))

    ground_traces: list[tuple[str, pd.DataFrame]] = []
    for p in ground_files:
        if "/dataset-old/" in p:
            tr = _load_old_trace(p)
        else:
            tr = _load_new_trace(p)
        if not tr.empty:
            ground_traces.append((os.path.basename(p), tr))

    drone_traces: list[tuple[str, pd.DataFrame]] = []
    for p in drone_files:
        tr = _load_new_trace(p)
        if not tr.empty:
            drone_traces.append((os.path.basename(p), tr))

    bs = _load_bs_points(str(bs_csv))

    all_points = []
    for _, t in ground_traces:
        all_points.append(t[["lat", "lon"]])
    for _, t in drone_traces:
        all_points.append(t[["lat", "lon"]])
    if not bs.empty:
        all_points.append(bs[["lat", "lon"]])
    if not all_points:
        raise RuntimeError("No valid points found to build map.")
    all_pts = pd.concat(all_points, ignore_index=True)
    center_lat = float(all_pts["lat"].mean())
    center_lon = float(all_pts["lon"].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron", control_scale=True)

    fg_ground = folium.FeatureGroup(name=f"Ground traces ({len(ground_traces)})", show=True)
    for name, tr in ground_traces:
        coords = tr[["lat", "lon"]].to_numpy().tolist()
        folium.PolyLine(coords, color="#1f77b4", weight=2, opacity=0.45, tooltip=f"Ground: {name}").add_to(fg_ground)
    fg_ground.add_to(m)

    fg_drone = folium.FeatureGroup(name=f"Drone traces ({len(drone_traces)})", show=True)
    drone_colors = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c", "#e377c2", "#8c564b", "#17becf", "#bcbd22"]
    for i, (name, tr) in enumerate(drone_traces):
        coords = tr[["lat", "lon"]].to_numpy().tolist()
        color = drone_colors[i % len(drone_colors)]
        folium.PolyLine(coords, color=color, weight=4, opacity=0.95, tooltip=f"Drone: {name}").add_to(fg_drone)
    fg_drone.add_to(m)

    fg_bs = folium.FeatureGroup(name=f"BS points ({len(bs)})", show=False)
    for _, row in bs.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=2,
            color="#444444",
            fill=True,
            fill_opacity=0.5,
            weight=1,
            tooltip=f"cell_id={int(row['cell_id'])}",
        ).add_to(fg_bs)
    fg_bs.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"[OK] Saved map: {out_html}")
    print(f"[INFO] Ground traces: {len(ground_traces)} | Drone traces: {len(drone_traces)} | BS: {len(bs)}")


if __name__ == "__main__":
    main()

