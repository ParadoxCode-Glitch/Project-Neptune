"""
ERA5 Pressure-Level Chunked Downloader — Project Neptune
=========================================================
Fetches ERA5 data in monthly chunks to stay within
the CDS per-request cost limit and avoid queue timeouts.

Each month is saved as a separate .grib file:
    data/raw/era5_pressure_levels/era5_pressure_levels_india_<YEAR>_<MONTH>.grib

Already-downloaded and validated months are automatically skipped.

Run:
    cd "c:\\Users\\Parad\\OneDrive\\Documents\\Project Neptune"
    python flood_prediction/data_pipeline/era5_chunked_download.py
"""

import os
import time
import logging
import cdsapi
import xarray as xr

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/raw/era5_pressure_levels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET = "reanalysis-era5-pressure-levels"

# Years to download — add/remove as needed
YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]

# Days subset — ~21 days/month keeps each request under the cost limit
DAYS = [
    "01", "03", "04", "06", "08", "09",
    "11", "12", "14", "16", "17", "19",
    "20", "22", "24", "25", "27", "28",
    "30", "31"
]

BASE_REQUEST = {
    "product_type": ["reanalysis"],
    "variable": [
        "relative_humidity",
        "specific_humidity",
        "specific_rain_water_content",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    ],
    "day": DAYS,
    "time": [
        "02:00", "05:00", "08:00",
        "11:00", "14:00", "17:00",
        "20:00", "23:00",
    ],
    "pressure_level": [
        "500",   # mid-troposphere synoptic forcing
        "700",   # moisture transport layer
        "850",   # low-level jet / moisture flux
        "925",   # boundary layer moisture
        "1000",  # near-surface
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [20, 79, 10, 86],   # [N, W, S, E] — India East Coast
}

# ── Download logic ────────────────────────────────────────────────────────────

def download_chunk(client: cdsapi.Client, year: str, month: str) -> bool:
    """
    Submit a single-month ERA5 request and validate using xarray.
    Returns True on success, False on failure.
    """
    out_file = os.path.join(OUTPUT_DIR, f"era5_pressure_levels_india_{year}_{month}.grib")

    def is_valid_grib(path: str) -> bool:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        try:
            with xr.open_dataset(path, engine="cfgrib") as ds:
                if len(ds.data_vars) > 0:
                    return True
        except Exception:
            pass
        return False

    if is_valid_grib(out_file):
        logger.info(f"[{year}-{month}] Valid cache found. Skipping {out_file}.")
        return True

    if os.path.exists(out_file):
        os.remove(out_file)

    logger.info(f"[{year}-{month}] Submitting request to CDS for monthly chunk...")
    request = {**BASE_REQUEST, "year": [year], "month": [month]}

    try:
        result = client.retrieve(DATASET, request)
        result.download(out_file)
        
        if is_valid_grib(out_file):
            size_mb = os.path.getsize(out_file) / (1024 ** 2)
            logger.info(f"[{year}-{month}] Download completed. {size_mb:.1f} MB saved to {out_file}.")
            return True
        else:
            logger.error(f"[{year}-{month}] Download completed but file validation failed. Corrupted artifact removed.")
            os.remove(out_file)
            return False

    except Exception as e:
        logger.error(f"[{year}-{month}] Download failed: {e}")
        # Remove partial file if it exists
        if os.path.exists(out_file):
            os.remove(out_file)
        return False


def main():
    logger.info("=" * 60)
    logger.info("ERA5 Chunked Downloader — Project Neptune")
    logger.info("=" * 60)
    logger.info(f"Dataset   : {DATASET}")
    logger.info(f"Variables : {BASE_REQUEST['variable']}")
    logger.info(f"Years     : {YEARS}")
    logger.info(f"Levels    : {BASE_REQUEST['pressure_level']} hPa")
    logger.info(f"Days/month: {len(DAYS)} days (subset to respect cost limits)")
    logger.info(f"Output dir: {os.path.abspath(OUTPUT_DIR)}")
    logger.info("=" * 60 + "\n")

    client = cdsapi.Client()  # reads ~/.cdsapirc

    results = {}
    months = [f"{m:02d}" for m in range(1, 13)]
    
    total_chunks = len(YEARS) * len(months)
    chunk_idx = 1
    
    for year in YEARS:
        for month in months:
            logger.info(f"── Chunk {chunk_idx}/{total_chunks} : {year}-{month} ──")
            ok = download_chunk(client, year, month)
            results[f"{year}-{month}"] = "OK" if ok else "FAILED"
            chunk_idx += 1

            # Small pause between requests to be polite to the CDS queue
            if chunk_idx <= total_chunks:
                logger.info("Waiting 3s before next request...\n")
                time.sleep(3)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    for year, status in results.items():
        logger.info(f"  {year} : {status}")
    logger.info("=" * 60)
    logger.info(f"Files saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
