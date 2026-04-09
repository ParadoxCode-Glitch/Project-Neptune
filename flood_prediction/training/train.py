import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FloodZarrDataset
from .loss import FloodLoss
from .metrics import compute_metrics
from model.floodformer import FloodFormerGV

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    
    # Progress bar
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        dynamic = batch['dynamic'].to(device)
        static = batch['static'].to(device)
        graph_nodes = batch['graph_nodes'].to(device)
        graph_edges = batch['graph_edges'].to(device)
        targets = batch['target'].to(device)
        
        # AMP
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            preds = model(dynamic, static, graph_nodes, graph_edges)
            loss, loss_dict = criterion(preds, targets, static)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), **loss_dict})
        
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_eval():
        for batch in loader:
            dynamic = batch['dynamic'].to(device)
            static = batch['static'].to(device)
            graph_nodes = batch['graph_nodes'].to(device)
            graph_edges = batch['graph_edges'].to(device)
            targets = batch['target'].to(device)
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                preds = model(dynamic, static, graph_nodes, graph_edges)
                loss, _ = criterion(preds, targets, static)
                
            total_loss += loss.item()
            metrics = compute_metrics(preds, targets)
            all_metrics.append(metrics)
            
    # average metrics
    avg_metrics = {k: sum(m[k] for m in all_metrics)/len(all_metrics) for k in all_metrics[0].keys()}
    return total_loss / len(loader), avg_metrics
