"""
CIFAR10 Dataset Loading Module
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_cifar10(batch_size=128, data_dir='./data/CIFAR10'):
    """
    load CIFAR10 dataset
    
    Args:
        batch_size: batch size
        data_dir: dataset save directory
    
    Returns:
        train_loader: training data loader, return (img, label)
            img: [N, 3, 32, 32], value in [0,1] range
            label: [N], class labels
        test_loader: test data loader, format same as above
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # load dataset
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
    
    # create data loader
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
