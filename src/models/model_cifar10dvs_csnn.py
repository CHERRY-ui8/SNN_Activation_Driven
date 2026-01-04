"""
CIFAR10DVS CSNN Model Definition
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class CIFAR10DVSCSNN(nn.Module):
    def __init__(self, T: int, channels: int, surrogate_func):
        super().__init__()
        self.T = T
        self.surrogate_func = surrogate_func

        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=self.surrogate_func),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, 2 * channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(2 * channels),
            neuron.IFNode(surrogate_function=self.surrogate_func),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Linear(2 * channels * 8 * 8, 64, bias=False),
            neuron.IFNode(surrogate_function=self.surrogate_func),

            layer.Linear(64, 10, bias=False),
            neuron.IFNode(surrogate_function=self.surrogate_func)
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        T = self.T
        
        if x.shape[3] == 128 and x.shape[4] == 128:
            x = torch.nn.functional.interpolate(
                x.view(N * T, 2, 128, 128), 
                size=(32, 32), 
                mode='bilinear', 
                align_corners=False
            ).view(N, T, 2, 32, 32)
        
        x_seq = x.permute(1, 0, 2, 3, 4)
        x_seq_out = self.conv_fc(x_seq)
        fr = x_seq_out.mean(dim=0)
        return fr

    def reset(self):
        functional.reset_net(self.conv_fc)
