import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SpatiotemporalTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, depth, grid_dim):
        super().__init__()
        self.patch_size = 4
        self.grid_dim = grid_dim
        
        # 3D Patch Embedding (Conv3d across Time and Space)
        # B, C, T, H, W -> B, embed_dim, T', H', W'
        self.patch_embed = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=(4, self.patch_size, self.patch_size), 
            stride=(4, self.patch_size, self.patch_size),
            padding=(0, 0, 0)
        )
        
        # Flattened grid size
        self.H_prime = grid_dim // self.patch_size
        self.W_prime = grid_dim // self.patch_size
        self.num_patches = self.H_prime * self.W_prime
        
        # Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, self.H_prime, self.W_prime))
        
        # Transformer
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: [B, T, H, W, C] -> rearrange to [B, C, T, H, W] for Conv3D
        x = x.permute(0, 4, 1, 2, 3) 
        x = self.patch_embed(x) # [B, embed_dim, T/4, H/4, W/4]
        
        x = x + self.pos_embed
        
        # Flatten spatial+temporal dims into sequence
        B, C, T_p, H_p, W_p = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1) # [B, T_p*H_p*W_p, C]
        
        # Multi-head attention
        x_att = self.transformer(x)
        
        # Reshape back to grid
        x_grid = x_att.permute(0, 2, 1).view(B, C, T_p, H_p, W_p)
        return x_grid, x_att
