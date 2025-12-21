"""
CIFAR10数据集加载模块
返回静态图像，模型内部会处理时间维度扩展
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_cifar10(batch_size=128, data_dir='./data/CIFAR10'):
    """
    加载CIFAR10数据集
    
    Args:
        batch_size: 批量大小
        data_dir: 数据集保存目录
    
    Returns:
        train_loader: 训练数据加载器，返回 (img, label)
            img: [N, 3, 32, 32]，值在[0,1]范围
            label: [N]，类别标签
        test_loader: 测试数据加载器，格式同上
    """
    # 数据预处理：归一化到[0,1]范围
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 注意：不进行标准化，保持[0,1]范围
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform_train,
        download=True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform_test,
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
    
    return train_loader, test_loader
