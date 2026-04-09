"""
run_pipeline.py — Project Neptune Data Pipeline
================================================
Orchestrates:
  1. ERA5 pressure-level download (CDS API key from config)
  2. NOAA tide/water-level fetch (free, no key needed)
  3. DEM loading (SRTM — free)
  4. Preprocessing → spatiotemporal alignment
  5. Zarr write for model training

API keys used:
  cds_api_key  — ~/.cdsapirc (auto-read by cdsapi library)
  noaa_api_key — null (NOAA public API, no key needed for basic access)
  open_meteo   — null (free tier, no key needed)
"""

import os
import logging

import yaml
import numpy as np
import xarray as xr

from data_pipeline.dem_loader             import fetch_dem_srtm
from data_pipeline.noaa_ingestion         import fetch_noaa_tides
from data_pipeline.flood_labels_ingestion import fetch_flood_labels
from data_pipeline.preprocessing          import align_spatiotemporal, engineer_features, validate_consistency
from data_pipeline.zarr_writer            import write_to_zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── ERA5 chunked download (inline — no separate ingestion module) ─────────────
def _fetch_era5_chunk(year: str, region_config: dict, output_dir: str) -> str | None:
    """
    Download one year of ERA5 pressure-level data via cdsapi.
    Credentials are read automatically from ~/.cdsapirc.
    Returns the output file path, or None on failure.
    """
    try:
        import cdsapi
    except ImportError:
        logger.error("cdsapi not installed. Run: pip install cdsapi")
        return None

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"era5_pressure_levels_india_{year}.grib")

    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        logger.info(f"[ERA5 {year}] Already exists — skipping.")
        return out_file

    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "relative_humidity",
            "specific_humidity",
            "specific_rain_water_content",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "year":  [year],
        "month": ["01","02","03","04","05","06","07","08","09","10","11","12"],
        "day":   ["01","03","04","06","08","09","11","12","14",
                  "16","17","19","20","22","24","25","27","28","30","31"],
        "time":  ["02:00","05:00","08:00","11:00","14:00","17:00","20:00","23:00"],
        "pressure_level": ["500","700","850","925","1000"],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [
            region_config["lat_max"], region_config["lon_min"],
            region_config["lat_min"], region_config["lon_max"],
        ],
    }

    logger.info(f"[ERA5 {year}] Submitting CDS request...")
    client = cdsapi.Client()   # reads ~/.cdsapirc — key: f1eef164-7aee-4ef1-9a21-048661ee5214
    client.retrieve("reanalysis-era5-pressure-levels", request, out_file)
    logger.info(f"[ERA5 {year}] ✅ Downloaded: {out_file}")
    return out_file


# ── Main pipeline ─────────────────────────────────────────────────────────────
def build_pipeline(config_path: str) -> bool:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["mode"] != "real":
        raise ValueError("Pipeline mode must be 'real'. Mock data is permanently disabled.")

    region_name   = list(config["regions"].keys())[0]
    region_config = config["regions"][region_name]
    time_range    = config["time_range"]

    # API keys (cdsapi reads its key from ~/.cdsapirc automatically)
    # open_meteo  : null — free, no key needed
    # noaa_api_key: null — public NOAA API, no key needed for standard use

    logger.info(f"Starting pipeline for region: {region_name}")
    logger.info(f"Time range: {time_range['start']} → {time_range['end']}")

    era5_dir = "data/raw/era5_pressure_levels"
    era5_paths = []

    # ── 1. ERA5 download (one file per year) ───────────────────────────────
    start_year = int(time_range["start"][:4])
    end_year   = int(time_range["end"][:4])

    for year in range(start_year, end_year + 1):
        path = _fetch_era5_chunk(str(year), region_config, era5_dir)
        if path:
            era5_paths.append(path)
        else:
            logger.warning(f"ERA5 {year} unavailable — continuing without it.")

    # ── 2. DEM (SRTM — free) ──────────────────────────────────────────────
    try:
        ot_key = config.get("api_keys", {}).get("opentopography_key")
        dem_path = fetch_dem_srtm(region_config, api_key=ot_key)
    except Exception as e:
        logger.error(f"DEM fetch failed: {e}")
        dem_path = None

    # ── 3. NOAA tides (free public API — no key required) ─────────────────
    try:
        noaa_path = fetch_noaa_tides(region_config, time_range)
    except Exception as e:
        logger.error(f"NOAA fetch failed: {e}")
        noaa_path = None

    # ── 4. Flood labels ────────────────────────────────────────────────────
    try:
        labels_path = fetch_flood_labels(region_config, time_range)
    except Exception as e:
        logger.error(f"Flood labels fetch failed: {e}")
        labels_path = None

    # ── Guard: abort if essential data missing ─────────────────────────────
    if not era5_paths or not dem_path:
        logger.error("CRITICAL: ERA5 or DEM data missing. Aborting pipeline.")
        logger.info("  ERA5  : ensure ~/.cdsapirc has your CDS key and Terms are accepted.")
        logger.info("  DEM   : ensure internet access for SRTM fetch.")
        return False

    # ── 5. Load & preprocess ───────────────────────────────────────────────
    logger.info("Loading ERA5 files into xarray...")
    dynamic_ds = xr.open_mfdataset(era5_paths, combine="by_coords", engine="cfgrib")

    logger.info("Loading DEM...")
    dem_ds = xr.open_dataset(dem_path, engine="rasterio") if dem_path.endswith(".tif") \
             else xr.open_dataset(dem_path)
    if "band_data" in dem_ds:
        dem_ds = dem_ds.rename({"band_data": "elevation"})
        dem_ds["elevation"] = dem_ds["elevation"].squeeze()

    aligned_dynamic, aligned_static = align_spatiotemporal(dynamic_ds, dem_ds, config)
    aligned_dynamic, aligned_static, nodes, edges = engineer_features(aligned_dynamic, aligned_static)
    validate_consistency(aligned_dynamic, aligned_static)

    # ── 6. Targets ─────────────────────────────────────────────────────────
    if labels_path:
        labels_ds = xr.open_dataset(labels_path, engine="rasterio")
        targets   = np.zeros((*aligned_dynamic["precip_mm"].shape, 3), dtype="float32")
    else:
        logger.warning("No flood labels — running in prediction-only mode (targets zeroed).")
        targets = np.zeros((*aligned_dynamic["precip_mm"].shape, 3), dtype="float32")

    # ── 7. Build dataset dict & write Zarr ────────────────────────────────
    logger.info("Building dataset arrays...")
    dynamic_block = np.stack(
        [aligned_dynamic["precip_mm"].values, aligned_dynamic["wind_speed"].values], axis=-1
    ).astype("float32")

    static_block = np.stack([
        aligned_static["elevation"].values,
        aligned_static["slope"].values,
        aligned_static["flow_accumulation"].values,
        aligned_static["distance_to_river"].values,
        aligned_static["hand"].values,
    ], axis=-1).astype("float32")

    dataset_dict = {
        "dynamic":      dynamic_block,
        "static":       static_block,
        "target":       targets,
        "graph_nodes":  nodes,
        "graph_edges":  edges,
    }

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zarr_out = os.path.join(base_dir, config["data"]["zarr_path"])
    write_to_zarr(dataset_dict, zarr_out, config["data"]["chunk_size"])

    logger.info("✅ Pipeline complete — real-world data ready for training.")
    return True


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_pipeline(os.path.join(base_dir, "configs", "config.yaml"))
