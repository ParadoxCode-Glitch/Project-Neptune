import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class TemporalGraphTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads):
        super().__init__()
        # Graph Attention Network
        self.gat = pyg_nn.GATv2Conv(in_channels, embed_dim, heads=num_heads, concat=False)
        
        # Temporal transformer over node features (if there is a time dimension for nodes, 
        # normally static or pre-processed temporal for simple GAT. 
        # Using simple linear projection for mock).
        self.fc = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x_nodes, edge_index):
        # x_nodes: [B, N, C]
        # edge_index: [2, E]
        
        # GAT currently doesn't natively handle batch dimension like [B, N, C] without PyG DataLoader Batching.
        # So we loop over batch or require batch_index.
        # For simplicity in this demo build, we do independent GAT per batch element
        out_batch = []
        for b in range(x_nodes.shape[0]):
            x = self.gat(x_nodes[b], edge_index)
            # Add simple non-linearity as "temporal/feature" transform
            x = torch.relu(self.fc(x))
            out_batch.append(x)
        
        out = torch.stack(out_batch, dim=0) # [B, N, embed_dim]
        return out
