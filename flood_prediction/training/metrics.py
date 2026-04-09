import torch
import numpy as np

def compute_metrics(preds, targets):
    p_prob, p_depth = preds[..., 0], preds[..., 1]
    t_prob, t_depth = targets[..., 0], targets[..., 1]
    
    # Segmentation (Threshold at 0.5)
    p_class = (p_prob > 0.5).float()
    t_class = (t_prob > 0.5).float()
    
    intersection = (p_class * t_class).sum()
    union = p_class.sum() + t_class.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Regression
    mse = torch.mean((p_depth - t_depth)**2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(p_depth - t_depth))
    
    return {
        'IoU': iou.item(),
        'RMSE': rmse.item(),
        'MAE': mae.item()
    }
