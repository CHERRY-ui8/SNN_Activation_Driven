"""
通用工具函数：日志初始化、模型保存/加载、设备选择、指标计算
所有训练脚本可复用此文件的函数，减少代码冗余
"""
import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict

def get_device(device_id: str = 'cuda:0') -> torch.device:
    """
    自动选择设备（优先GPU，无GPU则用CPU）
    Args:
        device_id: GPU设备ID（如'cuda:0'）
    Returns:
        device: 最终使用的设备
    """
    if torch.cuda.is_available() and 'cuda' in device_id:
        device = torch.device(device_id)
        print(f"使用GPU设备：{device_id}（共{torch.cuda.device_count()}个GPU）")
    else:
        device = torch.device('cpu')
        print("GPU不可用，使用CPU设备")
    return device

def init_tensorboard(log_dir: str = './logs') -> SummaryWriter:
    """
    初始化TensorBoard日志（按时间戳创建子目录，避免覆盖）
    Args:
        log_dir: 日志根目录
    Returns:
        writer: TensorBoard SummaryWriter对象
    """
    # 按当前时间创建子目录（格式：YYYYMMDD_HHMMSS）
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_dir = os.path.join(log_dir, time_str)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    print(f"TensorBoard日志目录：{tb_dir}（运行 tensorboard --logdir {tb_dir} 查看）")
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
    保存模型 checkpoint（含网络参数、优化器状态、当前epoch、最优准确率）
    Args:
        net: 训练的网络模型
        optimizer: 优化器
        epoch: 当前训练轮次
        max_test_acc: 目前为止的最大测试准确率
        save_path: 保存目录
        is_best: 是否为当前最优模型（若为True，额外保存为best.pth）
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    # checkpoint内容
    checkpoint = {
        'net_state_dict': net.state_dict(),  # 网络参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        'epoch': epoch,  # 当前epoch
        'max_test_acc': max_test_acc  # 最大测试准确率
    }
    # 保存最新模型
    latest_path = os.path.join(save_path, 'latest.pth')
    torch.save(checkpoint, latest_path)
    print(f"已保存最新模型到：{latest_path}")
    # 若为最优模型，额外保存
    if is_best:
        best_path = os.path.join(save_path, 'best.pth')
        torch.save(checkpoint, best_path)
        print(f"已保存最优模型到：{best_path}（准确率：{max_test_acc:.4f}）")

def load_checkpoint(
    net: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Dict:
    """
    加载模型 checkpoint（恢复训练状态）
    Args:
        net: 待加载参数的网络
        optimizer: 待恢复状态的优化器（可为None，仅加载网络参数）
        checkpoint_path: checkpoint文件路径
        device: 加载到的设备
    Returns:
        checkpoint: 加载的checkpoint字典（含epoch、max_test_acc等）
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在：{checkpoint_path}")
    # 加载checkpoint（映射到指定设备）
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 加载网络参数
    net.load_state_dict(checkpoint['net_state_dict'])
    # 加载优化器状态（若传入optimizer）
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"已加载checkpoint：{checkpoint_path}（恢复到epoch {checkpoint['epoch']}，最优准确率 {checkpoint['max_test_acc']:.4f}）")
    return checkpoint

def calculate_metrics(
    out_fr: torch.Tensor,
    label: torch.Tensor,
    loss: torch.Tensor
) -> tuple[float, float]:
    """
    计算当前批量的准确率和平均损失
    Args:
        out_fr: 网络输出的平均发放率（shape=[N,10]）
        label: 真实标签（shape=[N]）
        loss: 当前批量的损失（标量）
    Returns:
        acc: 准确率（0~1）
        avg_loss: 平均损失（损失/批量大小）
    """
    batch_size = label.numel()  # 批量大小
    # 计算准确率：预测类别=发放率最大的类别
    pred = out_fr.argmax(dim=1)  # 预测类别，shape=[N]
    acc = (pred == label).float().sum().item() / batch_size  # 准确率
    # 计算平均损失
    avg_loss = loss.item() / batch_size  # 损失除以批量大小
    return acc, avg_loss
