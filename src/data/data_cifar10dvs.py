"""
CIFAR10DVS数据集加载模块
支持按事件数或按时间切分事件流数据
"""
import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets import cifar10_dvs


def load_cifar10dvs(frame_num=16, batch_size=32, split_by='time', data_dir='./data/CIFAR10DVS'):
    """
    加载CIFAR10DVS数据集
    
    Args:
        frame_num: 切分的帧数（时间步长T）
        batch_size: 批量大小
        split_by: 切分方式，'number'（按事件数）或'time'（按时间）
        data_dir: 数据集保存目录
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        T: 实际使用的帧数（等于frame_num）
    """
    # 根据split_by选择切分方式
    if split_by == 'number':
        # 按事件数切分
        split_method = 'number'
    elif split_by == 'time':
        # 按时间切分
        split_method = 'time'
    else:
        raise ValueError(f"不支持的切分方式: {split_by}，请使用'number'或'time'")
    
    # 加载训练集
    train_dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        train=True,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_method,
        download=True
    )
    
    # 加载测试集
    test_dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        train=False,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_method,
        download=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, frame_num
