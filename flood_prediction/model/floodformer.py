import torch
import torch.nn as nn
from .vision_transformer import SpatiotemporalTransformer
from .graph_transformer import TemporalGraphTransformer
from .fusion import CrossModalFusion

class DecoderUNet(nn.Module):
    def __init__(self, embed_dim, t_out_frames, h_grid, w_grid):
        super().__init__()
        self.t_out = t_out_frames
        self.h = h_grid
        self.w = w_grid
        
        # Simple UNet-style upsampling to restore full resolution
        # From [B, embed_dim, T_p, H_p, W_p] back to [B, t_out, H, W, 3] target
        
        # Assuming patch size of 4
        self.up_conv = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 64, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Temporary linear map for T_in' -> t_out
        self.t_map = nn.Conv3d(32, 32, kernel_size=(6, 1, 1), stride=(1, 1, 1)) # simplified assumption based on mock settings
        
        # Output heads
        self.head_prob = nn.Conv2d(32, 1, kernel_size=1)
        self.head_depth = nn.Conv2d(32, 1, kernel_size=1)
        self.head_unc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_grid, original_t_in):
        # x_grid: [B, C, T_p, H_p, W_p]
        x_up = self.up_conv(x_grid) # [B, 64, T_p, H, W]
        # x_up: [B, 32, T_p, H, W]
        
        # We need to map temporal dimension
        # A simple global temporal pooling or mapping to desired T_out.
        # For simplicity in this demo, let's just average pool or map
        B, C, T_p, H, W = x_up.shape
        x_out = torch.mean(x_up, dim=2) # [B, 32, H, W]  # ignoring proper sequence generation for now
        
        # Actually we need T_out
        # So we duplicate or project
        x_out = x_out.unsqueeze(1).repeat(1, self.t_out, 1, 1, 1) # [B, t_out, 32, H, W]
        
        # Compute heads
        prob = []
        depth = []
        unc = []
        
        for t in range(self.t_out):
            x_t = x_out[:, t]
            prob.append(torch.sigmoid(self.head_prob(x_t)))
            depth.append(torch.relu(self.head_depth(x_t)))
            unc.append(torch.nn.functional.softplus(self.head_unc(x_t)))
            
        prob = torch.stack(prob, dim=1).squeeze(2) # [B, t_out, H, W]
        depth = torch.stack(depth, dim=1).squeeze(2)
        unc = torch.stack(unc, dim=1).squeeze(2)
        
        return torch.stack([prob, depth, unc], dim=-1) # [B, t_out, H, W, 3]

class FloodFormerGV(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg_m = cfg['model']
        cfg_t = cfg['data']
        
        in_grid_c = cfg_m['channels_dynamic'] + cfg_m['channels_static']
        embed_dim = cfg_m['embed_dim']
        
        self.vision_encoder = SpatiotemporalTransformer(
            in_channels=in_grid_c, 
            embed_dim=embed_dim, 
            num_heads=cfg_m['num_heads'], 
            depth=cfg_m['depth'], 
            grid_dim=cfg_m['grid_dim']
        )
        
        self.graph_encoder = TemporalGraphTransformer(
            in_channels=cfg_m['channels_graph'],
            embed_dim=embed_dim,
            num_heads=cfg_m['num_heads']
        )
        
        self.fusion = CrossModalFusion(embed_dim, cfg_m['num_heads'])
        
        self.decoder = DecoderUNet(
            embed_dim=embed_dim, 
            t_out_frames=cfg_t['window_out'], 
            h_grid=cfg_m['grid_dim'], 
            w_grid=cfg_m['grid_dim']
        )

    def forward(self, x_dynamic, x_static, graph_nodes, graph_edges):
        # x_dynamic: [B, T, H, W, Cd], x_static: [B, H, W, Cs]
        # Concat static features across time
        B, T, H, W, Cd = x_dynamic.shape
        x_static = x_static.unsqueeze(1).repeat(1, T, 1, 1, 1) # [B, T, H, W, Cs]
        x_grid_in = torch.cat([x_dynamic, x_static], dim=-1) # [B, T, H, W, Cd+Cs]
        
        # Vision Branch
        x_grid, x_grid_flat = self.vision_encoder(x_grid_in) # grid_flat: [B, Seq, C]
        
        # Graph Branch
        # graph_edges is shape [B, 2, E]. For GAT, we assume identical structure or loop.
        # we will use the first edge index as a constant topology for simplify.
        edge_index_0 = graph_edges[0].permute(1, 0) # [2, E]
        x_graph = self.graph_encoder(graph_nodes, edge_index_0) # [B, N, C]
        
        # Cross-Modal Fusion
        x_fused_flat = self.fusion(x_grid_flat, x_graph) # [B, Seq, C]
        
        # Reshape fused flat back to grid
        B, C, Tp, Hp, Wp = x_grid.shape
        x_fused_grid = x_fused_flat.permute(0, 2, 1).view(B, C, Tp, Hp, Wp)
        
        # Decoder
        outputs = self.decoder(x_fused_grid, T)
        
        return outputs
