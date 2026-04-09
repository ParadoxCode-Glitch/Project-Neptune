import os
import logging
from datetime import datetime
import xarray as xr
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_dem_srtm(region_config, output_dir="data/raw/dem", api_key=None):
    """
    Downloads SRTM / Copernicus DEM for the bounding box.
    Uses public OpenTopography API or similar as a robust source.
    """
    if api_key is None:
        api_key = os.environ.get('OPENTOPOGRAPHY_KEY', None)

    os.makedirs(output_dir, exist_ok=True)
    
    # Bounding box
    lat_min = region_config['lat_min']
    lat_max = region_config['lat_max']
    lon_min = region_config['lon_min']
    lon_max = region_config['lon_max']
    
    file_path = os.path.join(output_dir, f"dem_{lat_min}_{lon_min}_{lat_max}_{lon_max}.tif")
    if os.path.exists(file_path):
        logger.info(f"DEM already exists at {file_path}. Skipping download.")
        return file_path

    logger.info(f"Downloading DEM for bounds: {lat_min}, {lon_min}, {lat_max}, {lon_max}")
    
    # Note: OpenTopography SRTM GL3 (90m) API endpoint
    # Requires API key for heavy usage, but works for limited bounding boxes
    url = f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL3&south={lat_min}&north={lat_max}&west={lon_min}&east={lon_max}&outputFormat=GTiff"
    if api_key:
        url += f"&API_Key={api_key}"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"DEM successfully downloaded to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to download DEM: {str(e)}")
        # Fallback: create an empty template or gracefully fail
        raise RuntimeError("DEM download failed. Please check network or bounds.") from e
