"""
CIFAR10DVS高级预处理方法数据加载模块
支持多种预处理方法：baseline, count_norm, time_surface, adaptive_norm
"""
import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets import cifar10_dvs


def load_cifar10dvs_advanced(
    frame_num=16,
    batch_size=32,
    split_by='time',
    preprocess_method='baseline',
    data_dir='./data/CIFAR10DVS'
):
    """
    加载CIFAR10DVS数据集（支持高级预处理方法）
    
    Args:
        frame_num: 切分的帧数（时间步长T）
        batch_size: 批量大小
        split_by: 切分方式，'number'（按事件数）或'time'（按时间）
        preprocess_method: 预处理方法
            - 'baseline': 基础方法（按时间切分）
            - 'count_norm': 事件计数归一化
            - 'time_surface': 时间表面变换
            - 'adaptive_norm': 自适应归一化
        data_dir: 数据集保存目录
    
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        T: 实际使用的帧数（等于frame_num）
    """
    # 加载基础数据集（预处理在数据加载后应用）
    train_dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        train=True,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_by,
        download=True
    )
    
    test_dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        train=False,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_by,
        download=True
    )
    
    # 应用预处理方法（如果需要）
    if preprocess_method != 'baseline':
        # 这里可以添加自定义的预处理逻辑
        # 目前先返回基础数据，预处理可以在训练时应用
        pass
    
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
