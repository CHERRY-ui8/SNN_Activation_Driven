import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from spikingjelly.datasets import cifar10_dvs

class AdvancedPreprocessDataset(Dataset):
    """
    数据集包装器：在获取数据时实时应用高级预处理
    """
    def __init__(self, base_dataset, method='baseline'):
        self.base_dataset = base_dataset
        self.method = method

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        frame, label = self.base_dataset[index]
        # frame 原始维度: (T, C, H, W)，类型通常为 float32 或 int
        frame = torch.from_numpy(frame).float()

        if self.method == 'count_norm':
            # 方法3：事件计数归一化
            # 对每个时间步 T 和通道 C 分别寻找最大值并归一化
            # view(T, C, -1) 将 H,W 展平方便求 max
            T, C, H, W = frame.shape
            flat_frame = frame.view(T, C, -1)
            max_vals, _ = flat_frame.max(dim=2, keepdim=True)
            print(max_vals)
            # 防止除以0
            frame = frame / (max_vals.view(T, C, 1, 1) + 1e-5)

        elif self.method == 'time_surface':
            # 方法4：对数缩放计数 (Logarithmic Scaled Count)
            # 公式：xts = log(1 + x)，然后缩放到 [0, 1]
            frame = torch.log1p(frame) 
            # 全局或按帧归一化
            frame = frame / (frame.max() + 1e-5)

        elif self.method == 'adaptive_norm':
            # 方法5：自适应归一化 (基于密度的增益控制)
            # 计算每帧的密度（非零像素占比或平均计数）
            T, C, H, W = frame.shape
            # density 维度为 (T, C, 1, 1)
            density = frame.view(T, C, -1).mean(dim=2).view(T, C, 1, 1)
            norm_factor = 1.0 / (1.0 + density * 0.1)
            frame = frame * norm_factor
            # 最后做一次简单的全局裁剪或缩放确保数值稳定
            frame = torch.clamp(frame, 0, 1)

        return frame, label

def load_cifar10dvs_advanced(
    frame_num=16,
    batch_size=32,
    split_by='time',
    preprocess_method='baseline',
    data_dir='./data/CIFAR10DVS',
    train_ratio=0.8,
    random_seed=42
):
    """
    加载CIFAR10DVS数据集（支持高级预处理方法）
    """
    # 1. 加载原始的 SpikingJelly 数据集
    # 注意：这里 data_type='frame' 会预先完成从事件到帧的积分
    base_dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_by
    )
    
    dataset_size = len(base_dataset)
    print(f"数据集总大小: {dataset_size} | 预处理模式: {preprocess_method}")
    
    # 2. 划分训练集和测试集
    generator = torch.Generator().manual_seed(random_seed)
    train_base, test_base = random_split(
        base_dataset,
        [int(train_ratio * dataset_size), dataset_size - int(train_ratio * dataset_size)],
        generator=generator
    )
    
    # 3. 应用高级预处理装饰器
    train_dataset = AdvancedPreprocessDataset(train_base, method=preprocess_method)
    test_dataset = AdvancedPreprocessDataset(test_base, method=preprocess_method)
    
    # 4. 创建数据加载器
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