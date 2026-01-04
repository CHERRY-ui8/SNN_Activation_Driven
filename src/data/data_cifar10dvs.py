"""
CIFAR10DVS Dataset Loading Module
"""
import torch
from torch.utils.data import DataLoader, random_split
from spikingjelly.datasets import cifar10_dvs


def load_cifar10dvs(frame_num=16, batch_size=32, split_by='time', data_dir='./datasets/CIFAR10DVS', train_ratio=0.8, random_seed=42):
    if split_by == 'number':
        split_method = 'number'
    elif split_by == 'time':
        split_method = 'time'
    else:
        raise ValueError(f"Unsupported split mode: {split_by}, use 'number' or 'time'")
    
    dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_method
    )
    
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    
    generator = torch.Generator().manual_seed(random_seed)
    
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
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
