import logging
import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

logger = logging.getLogger(__name__)

def align_spatiotemporal(ds_dynamic: xr.Dataset, dem_ds: xr.Dataset, config: dict):
    """
    Aligns all dynamic grids and DEM to a single spatial coordinate grid
    and standardizes time intervals.
    """
    logger.info("Aligning spatio-temporal grids...")
    res = config['data']['spatial_resolution']
    
    lat_min, lat_max = config['region']['lat_min'], config['region']['lat_max']
    lon_min, lon_max = config['region']['lon_min'], config['region']['lon_max']
    
    # Create target grid
    lats = np.arange(lat_max, lat_min, -res)
    lons = np.arange(lon_min, lon_max, res)
    
    # 1. Spatial Alignment
    aligned_dynamic = ds_dynamic.interp(latitude=lats, longitude=lons, method="linear")
    aligned_dem = dem_ds.interp(latitude=lats, longitude=lons, method="linear")
    
    # 2. Temporal Alignment
    # Assuming config specifies '1H'.
    freq = config['data']['temporal_resolution']
    aligned_dynamic = aligned_dynamic.resample(time=freq).interpolate("linear")
    
    return aligned_dynamic, aligned_dem

def build_river_graph(accumulation: np.ndarray, threshold: float = 0.8):
    """
    Constructs a river network graph based on flow accumulation threshold.
    Returns:
        nodes: [num_nodes, features] -> features could just be (x,y)
        edges: [num_edges, 2] src -> dst
    """
    logger.info(f"Building river graph with accumulation threshold > {threshold}...")
    normalized_acc = (accumulation - accumulation.min()) / (accumulation.max() - accumulation.min() + 1e-8)
    
    # Identify river pixels
    river_mask = normalized_acc > threshold
    river_coords = np.argwhere(river_mask) # (N, 2) arrays of (y, x)
    
    num_nodes = len(river_coords)
    logger.info(f"Found {num_nodes} river nodes.")
    
    if num_nodes == 0:
        raise ValueError("No river nodes found with the current threshold.")
        
    # Build edges (simple nearest neighbor downstream)
    edges = []
    # For a real graph we'd use flow direction (D8 algorithm).
    # Here we approximate by connecting to the nearest neighbor that has higher accumulation (downstream)
    # or just nearby pixels if they are within 1 step (8-connected).
    
    # Create an index mapping for quick lookup
    coord_to_idx = {tuple(c): i for i, c in enumerate(river_coords)}
    
    for i, (y, x) in enumerate(river_coords):
        # 8-connected neighbors
        neighbors = [
            (y-1, x-1), (y-1, x), (y-1, x+1),
            (y, x-1),            (y, x+1),
            (y+1, x-1), (y+1, x), (y+1, x+1)
        ]
        
        # Connect to neighbors that are also river pixels
        local_edges = 0
        for ny, nx in neighbors:
            if (ny, nx) in coord_to_idx:
                # Downstream flow implies moving to higher accumulation
                if accumulation[ny, nx] > accumulation[y, x]:
                    edges.append([i, coord_to_idx[(ny, nx)]])
                    local_edges += 1
                    
        # Optional: if isolated, connect to nearest
        if local_edges == 0 and num_nodes > 1:
            pass # Sink node
            
    return river_coords.astype('float32'), np.array(edges, dtype='int32')

def build_grid_adjacency_graph(height, width):
    """
    Fallback graph containing all grid pixels with 4-connectivity.
    """
    logger.warning("Building fallback grid adjacency graph.")
    y, x = np.mgrid[0:height, 0:width]
    nodes = np.column_stack([y.ravel(), x.ravel()])
    edges = []
    
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if i > 0: edges.append([idx, (i-1)*width + j]) # up
            if i < height-1: edges.append([idx, (i+1)*width + j]) # down
            if j > 0: edges.append([idx, i*width + (j-1)]) # left
            if j < width-1: edges.append([idx, i*width + (j+1)]) # right
            
    return nodes.astype('float32'), np.array(edges, dtype='int32')

def engineer_features(dynamic: xr.Dataset, static: xr.Dataset):
    """
    Produces final feature variables: distance to river, elevation gradients, etc.
    """
    logger.info("Engineering specialized hydrological features...")
    
    # -- Dynamic --
    if 'u10' in dynamic and 'v10' in dynamic:
        dynamic['wind_speed'] = np.sqrt(dynamic['u10']**2 + dynamic['v10']**2)
        dynamic['wind_speed'].attrs['units'] = 'm/s'
        
    if 'tp' in dynamic:
        # tp is usually in meters in ERA5, convert to mm
        dynamic['precip_mm'] = dynamic['tp'] * 1000.0
        # Rolling 24h sum
        dynamic['rolling_rain_24h'] = dynamic['precip_mm'].rolling(time=24, min_periods=1).sum()

    # -- Static --
    elevation = static['elevation'].values
    
    # Calculate slopes (gradients)
    dy, dx = np.gradient(elevation)
    slope = np.sqrt(dy**2 + dx**2)
    static['slope'] = (('latitude', 'longitude'), slope)
    
    # Flow Accumulation (Approximation via Gaussian blurring of uniform rainfall logic)
    # A rough proxy for flow acc: lower areas get more acc from surrounding higher areas.
    # We invert elevation and smooth
    inv_elev = np.max(elevation) - elevation
    accumulation = gaussian_filter(inv_elev, sigma=2.0)
    static['flow_accumulation'] = (('latitude', 'longitude'), accumulation)
    
    # Distance to river
    river_mask = accumulation > np.percentile(accumulation, 90) # Top 10% accumulation are rivers
    # Invert mask for distance transform (0 at river, 1 elsewhere)
    dist_map = distance_transform_edt(~river_mask)
    static['distance_to_river'] = (('latitude', 'longitude'), dist_map)
    
    # HAND (Height Above Nearest Drainage)
    hand = np.zeros_like(elevation)
    for y in range(elevation.shape[0]):
        for x in range(elevation.shape[1]):
            # Find nearest river pixel (approx via dist map logic or directly)
            # A true HAND involves routing. Here we'll do an approximation.
            hand[y,x] = max(0, elevation[y,x] - (elevation[y,x] - np.log1p(dist_map[y,x])))
    static['hand'] = (('latitude', 'longitude'), hand)
    
    # Graph extraction
    try:
        nodes, edges = build_river_graph(accumulation)
    except Exception as e:
        logger.error(f"Failed to build river graph: {e}")
        nodes, edges = build_grid_adjacency_graph(elevation.shape[0], elevation.shape[1])
        
    return dynamic, static, nodes, edges

def validate_consistency(dynamic, static):
    """
    Asserts shapes, nulls, and grid matching before writing.
    """
    logger.info("Validating dataset consistency...")
    
    assert dynamic.latitude.size == static.latitude.size, "Latitude mismatch"
    assert dynamic.longitude.size == static.longitude.size, "Longitude mismatch"
    
    assert not dynamic.isnull().any().to_array().any(), "NaNs detected in dynamic dataset"
    assert not static.isnull().any().to_array().any(), "NaNs detected in static dataset"
    
    logger.info("Validation passed.")
