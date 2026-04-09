import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_pipeline.run_pipeline import build_pipeline
from training.dataset import FloodZarrDataset
from model.floodformer import FloodFormerGV
from training.loss import FloodLoss
from training.train import train_epoch

def run_demo():
    print("=== Coastal Flood Prediction System Demo ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    
    # 1. Run Real-World Data Pipeline
    print("Initiating real-world data pipeline...")
    success = build_pipeline(config_path)
    
    if not success:
        print("Pipeline aborted because real-world records could not be fetched.")
        print("Set up your API keys in config.yaml and ensure network connectivity to proceed to training.")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Dataset Setup
    print("Setting up dataset from actual processed Zarr...")
    zarr_path = os.path.join(base_dir, cfg['data']['zarr_path'])
    train_dataset = FloodZarrDataset(zarr_path, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    
    # 3. Model Setup
    print("Initializing FloodFormer-GV Model...")
    model = FloodFormerGV(cfg).to(device)
    
    # 4. Training Setup
    criterion = FloodLoss(
        w_task=cfg['training']['loss_weights']['task'],
        w_mass=cfg['training']['loss_weights']['mass'],
        w_elev=cfg['training']['loss_weights']['elev'],
        w_smooth=cfg['training']['loss_weights']['smooth']
    )
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg['training']['lr']))
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 5. Dummy Training Step
    print("Running 1 dummy epoch to verify pipeline loop...")
    loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
    
    print(f"Demo complete! Training loss: {loss:.4f}")
    print("End-to-End Real World Pipeline verified.")

if __name__ == "__main__":
    run_demo()
