"""
CIFAR10 CSNN Model Definition
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional, surrogate

class CIFAR10VGG16(nn.Module):
    def __init__(self, T: int, surrogate_func):
        super().__init__()
        self.T = T
        self.surrogate_func = surrogate_func

        # VGG16-like structure for 32x32x3 input
        def conv_block(in_channels, out_channels, num_convs):
            layers = []
            for _ in range(num_convs):
                layers.append(layer.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
                layers.append(layer.BatchNorm2d(out_channels))
                layers.append(neuron.LIFNode(surrogate_function=self.surrogate_func))
                in_channels = out_channels
            layers.append(layer.AvgPool2d(2, 2))
            return nn.Sequential(*layers)
        self.features = nn.Sequential(
            conv_block(3, 64, 2),
            conv_block(64, 128, 2),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            conv_block(512, 512, 3)
        )
        
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(512, 512),
            neuron.LIFNode(surrogate_function=self.surrogate_func),
            layer.Linear(512, 512),
            neuron.LIFNode(surrogate_function=self.surrogate_func),
            layer.Linear(512, 10)
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # expand time dim: [N,3,32,32] -> [T, N, 3, 32, 32]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq_out = self.features(x_seq)
        x_seq_out = self.classifier(x_seq_out)
        # average over time dimension
        fr = x_seq_out.mean(dim=0)
        return fr

class SpikingBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, surrogate_func=surrogate.ATan()):
        super().__init__()
        self.conv1 = layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(out_channels)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate_func)
        
        self.conv2 = layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(out_channels)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate_func)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.sn2(out)
        return out

class CIFAR10ResNet18(nn.Module):
    def __init__(self, T: int, surrogate_func=surrogate.ATan(), num_classes=10):
        super().__init__()
        self.T = T
        self.in_channels = 64

        self.prep = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate_func)
        )

        self.layer1 = self._make_layer(64, 2, 1, surrogate_func)
        self.layer2 = self._make_layer(128, 2, 2, surrogate_func)
        self.layer3 = self._make_layer(256, 2, 2, surrogate_func)
        self.layer4 = self._make_layer(512, 2, 2, surrogate_func)

        self.classifier = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Flatten(),
            layer.Linear(512, num_classes)
        )

        functional.set_step_mode(self, step_mode='m')

    def _make_layer(self, out_channels, num_blocks, stride, surrogate_func):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(SpikingBasicBlock(self.in_channels, out_channels, s, surrogate_func))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        x_seq = self.prep(x_seq)
        x_seq = self.layer1(x_seq)
        x_seq = self.layer2(x_seq)
        x_seq = self.layer3(x_seq)
        x_seq = self.layer4(x_seq)
        x_seq = self.classifier(x_seq)
        
        return x_seq.mean(dim=0)