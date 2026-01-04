import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from spikingjelly.datasets import cifar10_dvs

class AdvancedPreprocessDataset(Dataset):
    def __init__(self, base_dataset, method='baseline'):
        self.base_dataset = base_dataset
        self.method = method

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        frame, label = self.base_dataset[index]
        frame = torch.from_numpy(frame).float()

        if self.method == 'count_norm':
            T, C, H, W = frame.shape
            flat_frame = frame.view(T, C, -1)
            max_vals, _ = flat_frame.max(dim=2, keepdim=True)
            frame = frame / (max_vals.view(T, C, 1, 1) + 1e-5)

        elif self.method == 'time_surface':
            frame = torch.log1p(frame) 
            frame = frame / (frame.max() + 1e-5)

        elif self.method == 'adaptive_norm':
            T, C, H, W = frame.shape
            density = frame.view(T, C, -1).mean(dim=2).view(T, C, 1, 1)
            norm_factor = 1.0 / (1.0 + density * 0.1)
            frame = frame * norm_factor
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
    base_dataset = cifar10_dvs.CIFAR10DVS(
        root=data_dir,
        data_type='frame',
        frames_number=frame_num,
        split_by=split_by
    )
    
    dataset_size = len(base_dataset)
    print(f"Dataset size: {dataset_size} | Preprocess: {preprocess_method}")
    
    generator = torch.Generator().manual_seed(random_seed)
    train_base, test_base = random_split(
        base_dataset,
        [int(train_ratio * dataset_size), dataset_size - int(train_ratio * dataset_size)],
        generator=generator
    )
    
    train_dataset = AdvancedPreprocessDataset(train_base, method=preprocess_method)
    test_dataset = AdvancedPreprocessDataset(test_base, method=preprocess_method)
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