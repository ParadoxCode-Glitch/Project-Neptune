import zarr
import torch
from torch.utils.data import Dataset
import numpy as np

class FloodZarrDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset reading from Zarr.
    Implements sliding window over time dimension.
    """
    def __init__(self, zarr_path, window_in=24, window_out=6, split_ratio=(0.7, 0.15, 0.15), mode='train'):
        self.zarr_path = zarr_path
        self.window_in = window_in
        self.window_out = window_out
        self.mode = mode
        
        # Open zarr store lazily
        self.store = zarr.open(zarr_path, mode='r')
        self.dynamic = self.store['dynamic']
        self.static = self.store['static']
        self.target = self.store['target']
        
        self.graph_nodes = self.store['graph_nodes'][:] # small enough to load into RAM
        self.graph_edges = self.store['graph_edges'][:]
        
        # Calculate valid timesteps
        self.T_total = self.dynamic.shape[0]
        self.H = self.dynamic.shape[1]
        self.W = self.dynamic.shape[2]
        
        # valid start indices where we can fit window_in + window_out
        self.num_samples = self.T_total - self.window_in - self.window_out + 1
        
        if self.num_samples <= 0:
            raise ValueError("Total timesteps too small for the specified windows.")
            
        # Time split indices
        train_end = int(self.num_samples * split_ratio[0])
        val_end = train_end + int(self.num_samples * split_ratio[1])
        
        if mode == 'train':
            self.start_idx, self.end_idx = 0, train_end
        elif mode == 'val':
            self.start_idx, self.end_idx = train_end, val_end
        else: # test
            self.start_idx, self.end_idx = val_end, self.num_samples
            
        # load static features once
        self.static_data = torch.from_numpy(self.static[:])
        self.graph_nodes_data = torch.from_numpy(self.graph_nodes)
        self.graph_edges_data = torch.from_numpy(self.graph_edges).long() # edge_index needs to be long

    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        # actual index in zarr
        t_start = self.start_idx + idx
        t_mid = t_start + self.window_in
        t_end = t_mid + self.window_out
        
        # Lazy read chunk
        dyn = self.dynamic[t_start:t_mid]
        tgt = self.target[t_mid:t_end]
        
        X_dyn = torch.from_numpy(dyn)
        Y = torch.from_numpy(tgt)
        
        # For simplicity, duplicate static features across time to concat, 
        # or just return separately to let model handle it.
        # We will return separately
        return {
            "dynamic": X_dyn,          # [T_in, H, W, C_dyn]
            "static": self.static_data, # [H, W, C_stat]
            "target": Y,               # [T_out, H, W, 3]
            "graph_nodes": self.graph_nodes_data, # [N, C_g]
            "graph_edges": self.graph_edges_data, # [N_edges, 2]
        }
