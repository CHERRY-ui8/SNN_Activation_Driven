"""
General utility functions: log initialization, model saving/loading, device selection, metric calculation
All training scripts can reuse the functions in this file to reduce code redundancy
"""
import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict

def get_device(device_id: str = 'cuda:0') -> torch.device:
    """
    automatically select device (prefer GPU, use CPU if no GPU)
    Args:
        device_id: GPU device ID (e.g. 'cuda:0')
    Returns:
        device: the final selected device
    """
    if torch.cuda.is_available() and 'cuda' in device_id:
        device = torch.device(device_id)
        print(f"Using GPU device: {device_id} (total {torch.cuda.device_count()} GPUs)")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU device")
    return device

def init_tensorboard(log_dir: str = './logs') -> SummaryWriter:
    """
    initialize TensorBoard log (create subdirectory with timestamp to avoid overwriting)
    Args:
        log_dir: log root directory
    Returns:
        writer: TensorBoard SummaryWriter object
    """
    # create subdirectory with timestamp (format: YYYYMMDD_HHMMSS)
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_dir = os.path.join(log_dir, time_str)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    print(f"TensorBoard log directory: {tb_dir} (run tensorboard --logdir {tb_dir} to view)")
    return writer

def save_checkpoint(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_test_acc: float,
    save_path: str,
    is_best: bool = False
) -> None:
    """
    save model checkpoint (contains network parameters, optimizer state, current epoch, best accuracy)
    Args:
        net: trained network model
        optimizer: optimizer
        epoch: current training epoch
        max_test_acc: best test accuracy so far
        save_path: save directory
        is_best: whether the current model is the best model (if True, save as best.pth)
    """
    # create save directory
    os.makedirs(save_path, exist_ok=True)
    # checkpoint content
    checkpoint = {
        'net_state_dict': net.state_dict(),  # network parameters
        'optimizer_state_dict': optimizer.state_dict(),  # optimizer state
        'epoch': epoch,  # current training epoch
        'max_test_acc': max_test_acc  # best test accuracy so far
    }
    # save latest model
    latest_path = os.path.join(save_path, 'latest.pth')
    torch.save(checkpoint, latest_path)
    print(f"Saved latest model to: {latest_path}")
    # if the current model is the best model, save as best.pth
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
    """
    load model checkpoint (restore training state)
    Args:
        net: network model to load parameters
        optimizer: optimizer to restore state (can be None, only load network parameters)
        checkpoint_path: checkpoint file path
        device: device to load the checkpoint
    Returns:
        checkpoint: loaded checkpoint dictionary (contains epoch, max_test_acc, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    # load checkpoint (map to specified device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # load network parameters
    net.load_state_dict(checkpoint['net_state_dict'])
    # load optimizer state (if optimizer is not None)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint: {checkpoint_path} (restored to epoch {checkpoint['epoch']}, best accuracy {checkpoint['max_test_acc']:.4f})")
    return checkpoint

def calculate_metrics(
    out_fr: torch.Tensor,
    label: torch.Tensor,
    loss: torch.Tensor
) -> tuple[float, float]:
    """
    calculate current batch accuracy and average loss
    Args:
        out_fr: average firing rate of network output (shape=[N,10])
        label: true labels (shape=[N])
        loss: current batch loss (scalar)
    Returns:
        acc: accuracy (0~1)
        avg_loss: average loss (loss/batch size)
    """
    batch_size = label.numel()  # batch size
    # calculate accuracy: predicted class = class with highest firing rate
    pred = out_fr.argmax(dim=1)  # predicted class, shape=[N]
    acc = (pred == label).float().sum().item() / batch_size  # accuracy
    # calculate average loss
    avg_loss = loss.item() / batch_size  # loss divided by batch size
    return acc, avg_loss
