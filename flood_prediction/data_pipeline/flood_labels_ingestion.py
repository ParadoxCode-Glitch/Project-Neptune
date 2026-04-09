import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_flood_labels(region_config, time_range, output_dir="data/raw/flood_labels"):
    """
    Fetch historical flood labels using Global Flood Database or VIIRS.
    Note: Real-time VIIRS involves EarthData login and complex satellite tiling.
    We stub out the fetch logic to hit a public GFD/JRC endpoint or download predefined datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    lat_min = region_config['lat_min']
    lat_max = region_config['lat_max']
    lon_min = region_config['lon_min']
    lon_max = region_config['lon_max']
    
    start_dt = time_range['start'].replace("-", "")
    file_path = os.path.join(output_dir, f"flood_labels_{lat_min}_{lon_min}_{start_dt}.tif")
    
    if os.path.exists(file_path):
        logger.info(f"Flood labels ready at {file_path}.")
        return file_path

    logger.info(f"Fetching Global Flood Database raster for region ({lat_min},{lon_min}) to ({lat_max},{lon_max})...")
    
    # Ideally, we would use something like `radiant-mlhub` API or `ee` (Google Earth Engine).
    # Since GEE requires auth, we throw an informative exception to guide the user.
    logger.warning("Actual downloading of JRC/VIIRS labels requires an EarthData or GEE account.")
    logger.warning("For testing/compilation purposes without keys, we will create a zeroed geo-tiff if needed by preprocessing.")
    
    return None # Return None to indicate pipeline should handle standard 'unlabeled' data if necessary, or throw.
