"""
CIFAR10 Dataset Loading Module
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_cifar10(batch_size=128, data_dir='./data/CIFAR10', use_strong_aug=False, use_autoaug=False):
    """
    load CIFAR10 dataset
    
    Args:
        batch_size: batch size
        data_dir: dataset save directory
        use_strong_aug: whether to use strong data augmentation (ColorJitter + RandomErasing)
        use_autoaug: whether to use AutoAugment (overrides use_strong_aug if True)
    
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
    
    if use_autoaug:
        # AutoAugment: strongest augmentation policy
        try:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ])
        except AttributeError:
            # Fallback if AutoAugment not available (older torchvision versions)
            print("Warning: AutoAugment not available, using strong augmentation instead")
            use_autoaug = False
            use_strong_aug = True
            # Continue to strong augmentation logic below
    
    if not use_autoaug:
        # Only set transform if AutoAugment was not used (or failed)
        if use_strong_aug:
            # Strong data augmentation for better performance
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # Color jitter for robustness to color variations
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                normalize,
                # Random erasing (Cutout-like) for regularization
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            ])
        else:
            # Standard augmentation (original)
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
