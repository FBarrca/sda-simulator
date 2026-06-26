from __future__ import annotations

import argparse
import io
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from examples.logistics.network import (
    CUSTOMER_COORDS,
    CUSTOMERS,
    WAREHOUSE_COORDS,
    WAREHOUSES,
    distance_km,
    nearest_warehouse,
)

MAP_BOUNDS = {
    "lon_min": -9.55,
    "lon_max": 3.35,
    "lat_min": 35.75,
    "lat_max": 44.35,
}
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
WEB_MERCATOR_RADIUS = 6_378_137.0


def save_network_plot(
    output: str | Path = "examples/logistics/logistics_network.png",
    zoom: int = 7,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for logistics network plots. "
            "Run with `uv run --with matplotlib ...`."
        ) from exc

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    basemap, extent = _load_osm_basemap(zoom=zoom)
    colors = {
        "W_MADRID": "#3a86ff",
        "W_BARCELONA": "#2a9d8f",
        "W_VALENCIA": "#e76f51",
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(basemap, extent=extent, interpolation="bilinear", zorder=0)

    for customer in CUSTOMERS:
        customer_x, customer_y = _project_point(*CUSTOMER_COORDS[customer])
        nearest = nearest_warehouse(customer)
        for warehouse in WAREHOUSES:
            warehouse_x, warehouse_y = _project_point(*WAREHOUSE_COORDS[warehouse])
            is_nearest = warehouse == nearest
            ax.plot(
                [warehouse_x, customer_x],
                [warehouse_y, customer_y],
                color=colors[warehouse],
                alpha=0.82 if is_nearest else 0.18,
                linewidth=2.9 if is_nearest else 1.0,
                zorder=3 if is_nearest else 2,
            )

    for warehouse in WAREHOUSES:
        x, y = _project_point(*WAREHOUSE_COORDS[warehouse])
        ax.scatter(
            x,
            y,
            marker="D",
            s=150,
            color=colors[warehouse],
            edgecolor="white",
            linewidth=1.4,
            zorder=5,
        )
        ax.text(
            x + 14_000,
            y + 16_000,
            warehouse.replace("W_", ""),
            fontsize=9,
            fontweight="bold",
            color="#202020",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=6,
        )

    for customer in CUSTOMERS:
        x, y = _project_point(*CUSTOMER_COORDS[customer])
        nearest = nearest_warehouse(customer)
        ax.scatter(
            x,
            y,
            marker="o",
            s=76,
            color=colors[nearest],
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
        )
        label = customer.replace("C_", "").replace("_", " ").title()
        ax.text(
            x + 10_000,
            y - 11_000,
            label,
            fontsize=7.5,
            color="#202020",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
            },
            zorder=6,
        )

    ax.set_title("Spanish logistics dispatch network", fontsize=15, fontweight="bold")
    x_min, y_min = _project_point(MAP_BOUNDS["lon_min"], MAP_BOUNDS["lat_min"])
    x_max, y_max = _project_point(MAP_BOUNDS["lon_max"], MAP_BOUNDS["lat_max"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    _add_lane_table(ax)
    _add_attribution(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _add_lane_table(ax) -> None:
    text = "\n".join(
        [
            "Example lanes (km)",
            f"Madrid -> Sevilla: {distance_km('W_MADRID', 'C_SEVILLA'):.0f}",
            f"Barcelona -> Bilbao: {distance_km('W_BARCELONA', 'C_BILBAO'):.0f}",
            f"Valencia -> Castellon: {distance_km('W_VALENCIA', 'C_CASTELLON'):.0f}",
        ]
    )
    ax.text(
        0.02,
        0.03,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        zorder=7,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0"},
    )


def _add_attribution(ax) -> None:
    ax.text(
        0.99,
        0.015,
        "Basemap © OpenStreetMap contributors",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#202020",
        zorder=7,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.76,
        },
    )


def _load_osm_basemap(zoom: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to stitch map tiles. It is installed with matplotlib."
        ) from exc

    if not 1 <= zoom <= 19:
        raise ValueError("zoom must be between 1 and 19")

    x_min, y_min = _tile_xy(MAP_BOUNDS["lon_min"], MAP_BOUNDS["lat_max"], zoom)
    x_max, y_max = _tile_xy(MAP_BOUNDS["lon_max"], MAP_BOUNDS["lat_min"], zoom)
    tile_count = (x_max - x_min + 1) * (y_max - y_min + 1)
    if tile_count > 48:
        raise ValueError("zoom is too high for this example map extent")

    canvas = Image.new("RGB", ((x_max - x_min + 1) * 256, (y_max - y_min + 1) * 256))
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile = _fetch_osm_tile(zoom=zoom, x=x, y=y, image_cls=Image)
            canvas.paste(tile, ((x - x_min) * 256, (y - y_min) * 256))

    left, top = _tile_corner_to_web_mercator(x_min, y_min, zoom)
    right, bottom = _tile_corner_to_web_mercator(x_max + 1, y_max + 1, zoom)
    return np.asarray(canvas), (left, right, bottom, top)


def _fetch_osm_tile(*, zoom: int, x: int, y: int, image_cls):
    cache_path = _tile_cache_dir() / str(zoom) / str(x) / f"{y}.png"
    if cache_path.exists():
        return image_cls.open(cache_path).convert("RGB")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        OSM_TILE_URL.format(z=zoom, x=x, y=y),
        headers={"User-Agent": "sda-simulator-v2-logistics-example/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not download OpenStreetMap tiles for the logistics network map. "
            "Check network access or rerun after tiles are cached."
        ) from exc

    cache_path.write_bytes(data)
    return image_cls.open(io.BytesIO(data)).convert("RGB")


def _tile_cache_dir() -> Path:
    configured = os.environ.get("SDA_LOGISTICS_TILE_CACHE")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "sda_simulator_v2" / "osm_tiles"


def _project_point(lon: float, lat: float) -> tuple[float, float]:
    lat = min(max(lat, -85.05112878), 85.05112878)
    x = WEB_MERCATOR_RADIUS * math.radians(lon)
    y = WEB_MERCATOR_RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = min(max(lat, -85.05112878), 85.05112878)
    lat_rad = math.radians(lat)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return x, y


def _tile_corner_to_web_mercator(x: int, y: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return _project_point(lon, lat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="examples/logistics/logistics_network.png",
        help="path to the PNG file to create",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=7,
        help="OpenStreetMap tile zoom level for the basemap",
    )
    args = parser.parse_args()

    output_path = save_network_plot(output=args.output, zoom=args.zoom)
    print(f"Saved logistics network plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
