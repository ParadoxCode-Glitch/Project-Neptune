import os
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_noaa_tides(region_config, time_range, station_id=None, output_dir="data/raw/noaa"):
    """
    Downloads sea level and tides data via NOAA Tides & Currents API.
    A single station provides the proxy for the regional bounding box,
    or multiple if extended.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not station_id:
        # Default placeholder Galveston, TX for US Gulf Coast if no specific one provided
        logger.warning("No NOAA station ID provided. Using default 8771450 (Galveston Pier 21, TX) for demo.")
        station_id = "8771450"
        
    start_dt = time_range['start'].replace("-", "")
    end_dt = time_range['end'].replace("-", "")
    
    file_path = os.path.join(output_dir, f"noaa_{station_id}_{start_dt}_{end_dt}.csv")
    if os.path.exists(file_path):
        logger.info(f"NOAA data found at {file_path}. Skipping.")
        return file_path

    # NOAA Tides & Currents API URL
    url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        'begin_date': start_dt,
        'end_date': end_dt,
        'station': station_id,
        'product': 'water_level',
        'datum': 'MLLW',
        'units': 'metric',
        'time_zone': 'gmt',
        'format': 'csv',
        'application': 'flood_prediction_system'
    }

    logger.info(f"Fetching NOAA tide data for station {station_id}...")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        with open(file_path, "wb") as f:
            f.write(response.content)
            
        logger.info("NOAA Tide data successfully fetched.")
        return file_path
    except Exception as e:
        logger.error(f"NOAA data download failed: {str(e)}")
        raise e
