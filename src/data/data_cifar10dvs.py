"""
CIFAR10DVS数据集加载模块
支持按事件数或按时间切分事件流数据
"""
import torch
from torch.utils.data import DataLoader, random_split
from spikingjelly.datasets import cifar10_dvs


def load_cifar10dvs(frame_num=16, batch_size=32, split_by='time', data_dir='./datasets/CIFAR10DVS', train_ratio=0.8, random_seed=42):
    """
    加载CIFAR10DVS数据集
    
    Args:
        frame_num: 切分的帧数（时间步长T）
        batch_size: 批量大小
        split_by: 切分方式，'number'（按事件数）或'time'（按时间）
        data_dir: 数据集保存目录
        train_ratio: 训练集比例（默认0.8，即80%训练，20%测试）
        random_seed: 随机种子，用于确保数据拆分可重复
    
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
    
    # 加载完整数据集
    dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_method
    )
    
    # 获取数据集总大小
    dataset_size = len(dataset)
    print(f"数据集总大小: {dataset_size}")
    
    # 设置随机种子以确保可重复性
    generator = torch.Generator().manual_seed(random_seed)
    
    # 计算训练集和测试集的大小
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    
    # 手动打乱并拆分数据集
    # 使用 random_split 会自动打乱数据
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )
    
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时每个epoch都会打乱
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不需要打乱
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, frame_num
