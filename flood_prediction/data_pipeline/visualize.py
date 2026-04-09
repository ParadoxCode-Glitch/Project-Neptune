import matplotlib.pyplot as plt
import zarr
import os

def render_pipeline_validations(zarr_path, output_dir="visualizations"):
    """
    Renders visual validation plots of the processed real-world dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading datasets from Zarr at {zarr_path} for validation...")
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, mode='r')
    
    dynamic = root['dynamic'][:]
    static = root['static'][:]
    targets = root['target'][:]
    
    # Validate Rainfall Maps
    plt.figure(figsize=(10, 8))
    # Sum over time dimension (0) and take precip channel (0)
    total_rain = dynamic[:, :, :, 0].sum(axis=0)
    plt.imshow(total_rain, cmap='Blues')
    plt.colorbar(label='Total Accumulated Rainfall (mm)')
    plt.title('Accumulated Real Rainfall from ERA5')
    plt.savefig(os.path.join(output_dir, 'rainfall_map.png'))
    plt.close()
    
    # Validate Elevation and River Graph
    plt.figure(figsize=(10, 8))
    elevation = static[:, :, 0]
    plt.imshow(elevation, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    
    if 'graph_nodes' in root:
        nodes = root['graph_nodes'][:]
        # Scatter river nodes over terrain
        plt.scatter(nodes[:, 1], nodes[:, 0], c='blue', s=2, label='River Network Nodes')
        plt.legend()
        
    plt.title('DEM and River Network Graph Extraction')
    plt.savefig(os.path.join(output_dir, 'terrain_river_network.png'))
    plt.close()
    
    # Validate Flood History (Labels)
    plt.figure(figsize=(10, 8))
    # Sum over time, flood probability channel (0)
    flood_freq = targets[:, :, :, 0].mean(axis=0)
    plt.imshow(flood_freq, cmap='Reds')
    plt.colorbar(label='Flood Probability / Frequency')
    plt.title('Historical Flood Maps (VIIRS/GFD)')
    plt.savefig(os.path.join(output_dir, 'flood_history.png'))
    plt.close()
    
    print(f"Validations rendered and saved to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zarr_path = os.path.join(base_dir, "data/flood_dataset.zarr")
    if os.path.exists(zarr_path):
        render_pipeline_validations(zarr_path)
    else:
        print("Zarr dataset not found. Run the pipeline first.")
