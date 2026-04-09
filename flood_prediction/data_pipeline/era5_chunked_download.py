"""
ERA5 Pressure-Level Chunked Downloader — Project Neptune
=========================================================
Fetches ERA5 data year-by-year (2020–2026) to stay within
the CDS per-request cost limit.

Each year is saved as a separate .grib file:
    data/raw/era5_pressure_levels/era5_pressure_levels_india_<YEAR>.grib

Already-downloaded years are automatically skipped (safe to re-run).

Run:
    cd "c:\\Users\\Parad\\OneDrive\\Documents\\Project Neptune"
    python flood_prediction/data_pipeline/era5_chunked_download.py
"""

import os
import time
import logging
import cdsapi

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
    "month": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12",
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

def download_year(client: cdsapi.Client, year: str) -> bool:
    """
    Submit a single-year ERA5 request and download the result.
    Returns True on success, False on failure.
    """
    out_file = os.path.join(OUTPUT_DIR, f"era5_pressure_levels_india_{year}.grib")

    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        logger.info(f"[{year}] ⏭  Already downloaded — skipping ({out_file})")
        return True

    logger.info(f"[{year}] 📤 Submitting request to CDS...")
    request = {**BASE_REQUEST, "year": [year]}

    try:
        result = client.retrieve(DATASET, request)
        result.download(out_file)
        size_mb = os.path.getsize(out_file) / (1024 ** 2)
        logger.info(f"[{year}] ✅ Done — {size_mb:.1f} MB saved to {out_file}")
        return True

    except Exception as e:
        logger.error(f"[{year}] ❌ Failed: {e}")
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
    for i, year in enumerate(YEARS, 1):
        logger.info(f"── Chunk {i}/{len(YEARS)} : Year {year} ──")
        ok = download_year(client, year)
        results[year] = "✅ OK" if ok else "❌ FAILED"

        # Small pause between requests to be polite to the CDS queue
        if i < len(YEARS):
            logger.info("Waiting 5s before next request...\n")
            time.sleep(5)

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
