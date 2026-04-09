import os
import zarr
import numpy as np

def write_to_zarr(dataset_dict, zarr_path, chunk_config):
    """
    Writes dynamic, static, target data arrays into a chunked zarr store.
    Validates shapes and NaNs directly before commit.
    """
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Chunks
    Tc = chunk_config.get('time', 24)
    Hc = chunk_config.get('lat', 128)
    Wc = chunk_config.get('lon', 128)
    
    for key in ['dynamic', 'static', 'target']:
        if key in dataset_dict:
            data = dataset_dict[key]
            # Safety Check: NaN explosion Validation
            if np.isnan(data).any():
                raise ValueError(f"NaNs detected in {key} block just before Zarr serialization!")
                
            if key == 'dynamic':
                assert len(data.shape) == 4, "Dynamic must be [time, lat, lon, channel]"
                chunks = (Tc, Hc, Wc, data.shape[-1])
            elif key == 'static':
                assert len(data.shape) == 3, "Static must be [lat, lon, channel]"
                chunks = (Hc, Wc, data.shape[-1])
            elif key == 'target':
                assert len(data.shape) == 4, "Target must be [time, lat, lon, channel]"
                chunks = (Tc, Hc, Wc, data.shape[-1])
                
            root.create_dataset(key, data=data, shape=data.shape, chunks=chunks, dtype='float32')
            
    # Graph artifacts
    if 'graph_nodes' in dataset_dict:
        root.create_dataset('graph_nodes', data=dataset_dict['graph_nodes'], dtype='float32')
    if 'graph_edges' in dataset_dict:
        root.create_dataset('graph_edges', data=dataset_dict['graph_edges'], dtype='int32')
        
    print(f"Dataset successfully written to {zarr_path}")
