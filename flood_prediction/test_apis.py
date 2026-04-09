"""
test_apis.py — Project Neptune API Diagnostics
================================================
Tests connectivity and basic responses for all active data APIs:
  1. CDS API (ERA5)     — cdsapi + ~/.cdsapirc
  2. NOAA Tides API     — free public REST
  3. SRTM DEM loader    — free (elevation tiles)
  4. Flood Labels       — fetch_flood_labels

Run:
    cd "c:\\Users\\Parad\\OneDrive\\Documents\\Project Neptune\\flood_prediction"
    python test_apis.py
"""

import os
import sys
import time
import logging
import yaml
import requests

logging.basicConfig(level=logging.WARNING)   # suppress library noise during test

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")

# Small test region + time window so tests finish quickly
TEST_REGION = {"lat_min": 13.0, "lat_max": 13.1, "lon_min": 80.2, "lon_max": 80.3}
TEST_TIME   = {"start": "2020-01-01", "end": "2020-01-01"}

PASS  = "✅ PASS"
FAIL  = "❌ FAIL"
WARN  = "⚠️  WARN"

results = {}

# ─────────────────────────────────────────────────────────────────────────────

def header(n, total, title):
    print(f"\n[{n}/{total}] {title}")
    print("─" * 50)

# ── 1. Config load ────────────────────────────────────────────────────────────
def test_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        key = cfg.get("api_keys", {}).get("cds_api_key", "")
        masked = key[:8] + "****" + key[-4:] if key and len(key) > 12 else "MISSING"
        print(f"  mode        : {cfg.get('mode')}")
        print(f"  cds_api_key : {masked}")
        print(f"  open_meteo  : {cfg['api_keys'].get('open_meteo')} (no key needed)")
        print(f"  noaa_api_key: {cfg['api_keys'].get('noaa_api_key')} (no key needed)")
        results["Config"] = PASS
        return cfg
    except Exception as e:
        print(f"  {FAIL}: {e}")
        results["Config"] = FAIL
        return {}

# ── 2. CDS / ERA5 API ────────────────────────────────────────────────────────
def test_cds():
    """
    Verifies the CDS API token is valid by hitting the whoami endpoint.
    Does NOT submit a data request (avoids queueing).
    """
    try:
        import cdsapi  # noqa
        print("  cdsapi library   : installed ✅")
    except ImportError:
        print(f"  {FAIL}: cdsapi not installed. Run: pip install cdsapi")
        results["CDS ERA5"] = FAIL
        return

    # Check ~/.cdsapirc
    rc_path = os.path.join(os.path.expanduser("~"), ".cdsapirc")
    if not os.path.exists(rc_path):
        print(f"  {FAIL}: ~/.cdsapirc not found")
        results["CDS ERA5"] = FAIL
        return
    print(f"  ~/.cdsapirc      : found ✅")

    with open(rc_path) as f:
        rc = f.read()
    url = next((l.split(":", 1)[1].strip() for l in rc.splitlines() if l.startswith("url")), None)
    key = next((l.split(":", 1)[1].strip() for l in rc.splitlines() if l.startswith("key")), None)

    if not url or not key:
        print(f"  {FAIL}: .cdsapirc missing url or key")
        results["CDS ERA5"] = FAIL
        return

    # Validate token against CDS REST API
    try:
        t0 = time.time()
        r = requests.get(
            f"{url.rstrip('/')}/account/check",
            headers={"PRIVATE-TOKEN": key},
            timeout=10
        )
        elapsed = time.time() - t0
        if r.status_code == 200:
            info = r.json()
            print(f"  Token valid      : ✅  (user: {info.get('uid', 'unknown')})")
            print(f"  CDS response     : {elapsed:.2f}s")
            results["CDS ERA5"] = PASS
        elif r.status_code == 401:
            print(f"  {FAIL}: Invalid API token (401 Unauthorized)")
            results["CDS ERA5"] = FAIL
        else:
            # Some CDS endpoints return 404 for /account/check — try datasets endpoint
            r2 = requests.get(
                f"{url.rstrip('/')}/retrieve/v1/processes/reanalysis-era5-pressure-levels",
                headers={"PRIVATE-TOKEN": key},
                timeout=10
            )
            if r2.status_code == 200:
                print(f"  Token valid      : ✅  ({elapsed:.2f}s)")
                print(f"  Dataset access   : ERA5 pressure-levels reachable ✅")
                results["CDS ERA5"] = PASS
            else:
                print(f"  {WARN}: HTTP {r2.status_code} — token may be valid but endpoint changed")
                results["CDS ERA5"] = WARN
    except requests.exceptions.Timeout:
        print(f"  {WARN}: CDS API timeout — server may be slow")
        results["CDS ERA5"] = WARN
    except Exception as e:
        print(f"  {FAIL}: {e}")
        results["CDS ERA5"] = FAIL

# ── 3. NOAA Tides API ─────────────────────────────────────────────────────────
def test_noaa():
    """Fetch 1 day of water-level data from a known NOAA station."""
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "begin_date": "20200101", "end_date": "20200101",
        "station": "8771450",     # Galveston Pier 21, TX
        "product": "water_level",
        "datum": "MLLW", "units": "metric",
        "time_zone": "gmt", "format": "json",
        "application": "project_neptune_test"
    }
    try:
        t0 = time.time()
        r = requests.get(url, params=params, timeout=15)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            n_records = len(data["data"])
            print(f"  Station 8771450  : ✅  {n_records} records returned ({elapsed:.2f}s)")
            print(f"  Sample reading   : {data['data'][0].get('v')} m (MLLW) at {data['data'][0].get('t')}")
            results["NOAA Tides"] = PASS
        elif "error" in data:
            print(f"  {FAIL}: NOAA error — {data['error'].get('message')}")
            results["NOAA Tides"] = FAIL
        else:
            print(f"  {WARN}: Unexpected response structure ({elapsed:.2f}s)")
            results["NOAA Tides"] = WARN
    except Exception as e:
        print(f"  {FAIL}: {e}")
        results["NOAA Tides"] = FAIL

# ── 4. DEM / SRTM reachability ────────────────────────────────────────────────
def test_dem():
    """Ping the SRTM tile server to confirm it's reachable."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        api_key = cfg.get("api_keys", {}).get("opentopography_key", "demoapikeyot2022")
    except Exception:
        api_key = "demoapikeyot2022"

    # OpenTopography SRTM API (free, no key needed for basic SRTM tiles)
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL3",
        "south": 13.0, "north": 13.1,
        "west": 80.2, "east": 80.3,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }
    try:
        t0 = time.time()
        r = requests.get(url, params=params, timeout=20, stream=True)
        elapsed = time.time() - t0
        content_type = r.headers.get("Content-Type", "")
        size_kb = int(r.headers.get("Content-Length", 0)) // 1024

        if r.status_code == 200 and ("tiff" in content_type or "octet" in content_type or size_kb > 0):
            print(f"  SRTM tile server : ✅  reachable ({elapsed:.2f}s)")
            print(f"  Tile size        : {size_kb or '?'} KB (India East Coast test patch)")
            results["SRTM DEM"] = PASS
        else:
            print(f"  {WARN}: HTTP {r.status_code} — {content_type} ({elapsed:.2f}s)")
            print(f"  Response         : {r.text[:200]}")
            results["SRTM DEM"] = WARN
    except Exception as e:
        print(f"  {FAIL}: {e}")
        results["SRTM DEM"] = FAIL

# ── 5. Open-Meteo (free fallback, no key) ─────────────────────────────────────
def test_open_meteo():
    """Quick ping of Open-Meteo historical API — free, no key needed."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 13.08, "longitude": 80.27,
        "start_date": "2020-01-01", "end_date": "2020-01-01",
        "hourly": "precipitation", "timezone": "UTC"
    }
    try:
        t0 = time.time()
        r = requests.get(url, params=params, timeout=10)
        elapsed = time.time() - t0
        r.raise_for_status()
        data = r.json()
        precip = data.get("hourly", {}).get("precipitation", [])
        print(f"  Open-Meteo API   : ✅  reachable ({elapsed:.2f}s)")
        print(f"  Sample data      : {precip[:4]} mm/hr (2020-01-01 Chennai)")
        results["Open-Meteo"] = PASS
    except Exception as e:
        print(f"  {FAIL}: {e}")
        results["Open-Meteo"] = FAIL

# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 50)
    print("  DIAGNOSTICS SUMMARY")
    print("=" * 50)
    for name, status in results.items():
        print(f"  {name:<20} {status}")
    print("=" * 50)
    failed = [k for k, v in results.items() if "FAIL" in v]
    if failed:
        print(f"\n  ⚠️  Action needed for: {', '.join(failed)}")
    else:
        print("\n  🎉 All APIs operational — ready to download data!")

# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Project Neptune — API Diagnostics")
    print("=" * 50)

    header(1, 5, "Config")
    test_config()

    header(2, 5, "Copernicus CDS (ERA5 pressure-levels)")
    test_cds()

    header(3, 5, "NOAA Tides & Currents (free)")
    test_noaa()

    header(4, 5, "SRTM DEM / OpenTopography (free)")
    test_dem()

    header(5, 5, "Open-Meteo Historical Archive (free, no key)")
    test_open_meteo()

    print_summary()
