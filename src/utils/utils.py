"""
General utility functions
"""
import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict

def get_device(device_id: str = 'cuda:0') -> torch.device:
    if torch.cuda.is_available() and 'cuda' in device_id:
        device = torch.device(device_id)
        print(f"Using GPU device: {device_id} (total {torch.cuda.device_count()} GPUs)")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU device")
    return device

def init_tensorboard(log_dir: str = './logs') -> SummaryWriter:
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_dir = os.path.join(log_dir, time_str)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    print(f"TensorBoard log directory: {tb_dir}")
    return writer

def save_checkpoint(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_test_acc: float,
    save_path: str,
    is_best: bool = False
) -> None:
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'max_test_acc': max_test_acc
    }
    latest_path = os.path.join(save_path, 'latest.pth')
    torch.save(checkpoint, latest_path)
    print(f"Saved latest model to: {latest_path}")
    if is_best:
        best_path = os.path.join(save_path, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to: {best_path} (accuracy: {max_test_acc:.4f})")

def load_checkpoint(
    net: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {checkpoint['epoch']}, acc {checkpoint['max_test_acc']:.4f})")
    return checkpoint

def calculate_metrics(
    out_fr: torch.Tensor,
    label: torch.Tensor,
    loss: torch.Tensor
) -> tuple[float, float]:
    pred = out_fr.argmax(dim=1)
    acc = (pred == label).float().mean().item()
    return acc, loss.item()
