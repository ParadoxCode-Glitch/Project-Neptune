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

    # NOAA restricts water_level to 31 day windows. We must chunk by 30 days.
    start_time = pd.to_datetime(time_range['start'])
    end_time = pd.to_datetime(time_range['end'])
    date_ranges = pd.date_range(start=start_time, end=end_time, freq='30D').to_list()
    # Ensure end_time is captured if it doesn't align exactly with 30D intervals
    if date_ranges[-1] < end_time:
        date_ranges.append(end_time)

    logger.info(f"Fetching NOAA tide data for station {station_id} in chunks...")
    all_chunks = []
    
    url = f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    try:
        for i in range(len(date_ranges) - 1):
            chunk_start = date_ranges[i].strftime('%Y%m%d')
            chunk_end = date_ranges[i+1].strftime('%Y%m%d')
            
            params = {
                'begin_date': chunk_start,
                'end_date': chunk_end,
                'station': station_id,
                'product': 'water_level',
                'datum': 'MLLW',
                'units': 'metric',
                'time_zone': 'gmt',
                'format': 'csv',
                'application': 'flood_prediction_system'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # The API returns CSV text. If error, CSV contains 'Error:'
            if "Error" in response.text[:20]:
                logger.warning(f"NOAA API returned error for chunk {chunk_start}-{chunk_end}: {response.text}")
                continue
                
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            all_chunks.append(df)
            
        if all_chunks:
            final_df = pd.concat(all_chunks, ignore_index=True)
            # Remove any trailing duplicates caused by boundary overlaps
            final_df.drop_duplicates(subset=['Date Time'], inplace=True)
            final_df.to_csv(file_path, index=False)
            logger.info("NOAA Tide data successfully fetched and chunked.")
        else:
            raise ValueError("No valid data chunks returned by NOAA.")
            
        return file_path
    except Exception as e:
        logger.error(f"NOAA data download failed: {str(e)}")
        raise e
