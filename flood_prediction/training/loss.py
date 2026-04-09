import torch
import torch.nn as nn
import torch.nn.functional as F

class FloodLoss(nn.Module):
    def __init__(self, w_task=1.0, w_mass=0.1, w_elev=0.1, w_smooth=0.05):
        super().__init__()
        self.w_task = w_task
        self.w_mass = w_mass
        self.w_elev = w_elev
        self.w_smooth = w_smooth
        
        self.bce = nn.BCELoss()
        self.huber = nn.HuberLoss()

    def task_loss(self, preds, targets):
        # preds/targets shape: [B, T_out, H, W, 3]
        # channels: prob, depth, unc
        p_prob, p_depth, p_unc = preds[..., 0], preds[..., 1], preds[..., 2]
        t_prob, t_depth, t_unc = targets[..., 0], targets[..., 1], targets[..., 2]
        
        l_seg = self.bce(p_prob, t_prob)
        l_reg = self.huber(p_depth, t_depth)
        
        # Gaussian NLL for uncertainty
        var = p_unc + 1e-6
        l_unc = torch.mean(0.5 * torch.log(var) + 0.5 * ((t_depth - p_depth)**2) / var)
        
        return l_seg + l_reg + l_unc

    def physics_loss_mass(self, p_depth):
        # Simplistic approximation: total water volume shouldn't wildly fluctuate frame-to-frame
        # unless driven by input, but without complex hydrology engine in loss, we just penalize
        # massive unexplained changes between t and t+1.
        delta = p_depth[:, 1:] - p_depth[:, :-1]
        return torch.mean(delta**2)

    def physics_loss_elev(self, p_depth, static_features):
        # Water predicted where elevation is extremely high should be penalized.
        # static_features: [B, H, W, Cs]. C=0 is elevation.
        elevation = static_features[..., 0].unsqueeze(1) # [B, 1, H, W]
        # simple heuristic: height above threshold penalization
        violation = torch.relu((elevation * 0.1) - p_depth) 
        # actual physics: water shouldn't pool on slopes.
        slope = static_features[..., 1].unsqueeze(1)
        slope_violation = slope * p_depth
        return torch.mean(slope_violation)

    def physics_loss_smooth(self, p_depth):
        # Spatial smoothness (Laplacian)
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(p_depth.device)
        B, T, H, W = p_depth.shape
        p_flat = p_depth.view(B*T, 1, H, W)
        lap = F.conv2d(p_flat, laplacian_kernel, padding=1)
        return torch.mean(lap**2)

    def forward(self, preds, targets, static_features):
        l_t = self.task_loss(preds, targets)
        l_m = self.physics_loss_mass(preds[..., 1])
        l_e = self.physics_loss_elev(preds[..., 1], static_features)
        l_s = self.physics_loss_smooth(preds[..., 1])
        
        l_total = self.w_task * l_t + self.w_mass * l_m + self.w_elev * l_e + self.w_smooth * l_s
        
        return l_total, {
            'task': l_t.item(),
            'mass': l_m.item(),
            'elev': l_e.item(),
            'smooth': l_s.item()
        }
