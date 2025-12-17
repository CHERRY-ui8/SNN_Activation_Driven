"""
CIFAR10DVS神经形态数据集的激活驱动卷积脉冲神经网络（CSNN）定义
适配2通道事件数据（DVS的极性通道：on/off），支持帧输入
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class CIFAR10DVSCSNN(nn.Module):
    def __init__(self, T: int, channels: int, surrogate_func):
        """
        Args:
            T: 帧数量（DVS事件切分的帧数，即时间步长）
            channels: 第一层卷积的输出通道数
            surrogate_func: 替代梯度函数
        """
        super().__init__()
        self.T = T  # 帧数量（与DVS数据的时间维度匹配）
        self.surrogate_func = surrogate_func

        # 网络结构（适配CIFAR10DVS的32×32×2输入，2为极性通道）
        # 参考CIFAR10模型，使用完整的Sequential，让spikingjelly自动处理
        self.conv_fc = nn.Sequential(
            # 卷积层1：2→channels，3×3卷积， padding=1，输出32×32
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=self.surrogate_func),
            layer.MaxPool2d(2, 2),  # 输出16×16

            # 卷积层2：channels→2*channels，3×3卷积， padding=1，输出16×16
            layer.Conv2d(channels, 2 * channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(2 * channels),
            neuron.IFNode(surrogate_function=self.surrogate_func),
            layer.MaxPool2d(2, 2),  # 输出8×8

            # 全连接层1：2*channels×8×8 → 64
            layer.Flatten(),  # 展平：2*channels×8×8
            layer.Linear(2 * channels * 8 * 8, 64, bias=False),
            neuron.IFNode(surrogate_function=self.surrogate_func),

            # 全连接层2：64→10（CIFAR10DVS共10类）
            layer.Linear(64, 10, bias=False),
            neuron.IFNode(surrogate_function=self.surrogate_func)
        )

        # 设置多步模式
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：DVS帧数据→调整时间维度→脉冲序列输出→计算平均发放率
        Args:
            x: DVS帧数据，shape=[N, T, 2, 32, 32]（N=批量，T=帧数）
        Returns:
            fr: 输出层平均脉冲发放率，shape=[N, 10]
        """
        N = x.shape[0]  # 批量大小
        T = self.T
        
        # 如果输入仍然是128x128（数据加载可能还未更新），先下采样到32x32
        if x.shape[3] == 128 and x.shape[4] == 128:
            # 下采样：[N, T, 2, 128, 128] -> [N, T, 2, 32, 32]
            x = torch.nn.functional.interpolate(
                x.view(N * T, 2, 128, 128), 
                size=(32, 32), 
                mode='bilinear', 
                align_corners=False
            ).view(N, T, 2, 32, 32)
        
        # 调整时间维度顺序：[N, T, 2, 32, 32] → [T, N, 2, 32, 32]（适配多步模式）
        x_seq = x.permute(1, 0, 2, 3, 4)
        # 脉冲序列前向传播（与CIFAR10模型保持一致）
        x_seq_out = self.conv_fc(x_seq)
        # 计算平均发放率
        fr = x_seq_out.mean(dim=0)
        return fr

    def reset(self):
        """重置神经元状态"""
        # 只重置子模块，避免递归调用
        functional.reset_net(self.conv_fc)
