import torch
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # Cross Attention: Grid querying Graph
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, grid_tokens, graph_tokens):
        # grid_tokens: [B, Seq_Grid, C]
        # graph_tokens: [B, Seq_Graph, C]

        # Grid tokens query graph tokens (What upstream river info is relevant to this grid?)
        attn_out, _ = self.cross_attn(query=grid_tokens, key=graph_tokens, value=graph_tokens)
        
        # Add & Norm
        x = self.norm1(grid_tokens + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x
