"""
CIFAR10数据集的激活驱动卷积脉冲神经网络（CSNN）定义
适配3通道RGB图像，核心逻辑：以脉冲平均发放率为优化目标
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class CIFAR10CSNN(nn.Module):
    def __init__(self, T: int, channels: int, surrogate_func):
        """
        Args:
            T: 模拟时间步长（脉冲发放的时间序列长度）
            channels: 第一层卷积的输出通道数
            surrogate_func: 替代梯度函数（如ATan、SuperSpike等）
        """
        super().__init__()
        self.T = T  # 时间步长，需与数据编码的时间维度匹配
        self.surrogate_func = surrogate_func  # 替代梯度函数

        # 卷积+全连接网络结构（适配CIFAR10的32×32×3输入）
        self.conv_fc = nn.Sequential(
            # 卷积层1：3→channels，3×3卷积， padding=1，输出32×32
            layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),  # 批归一化，加速收敛
            neuron.IFNode(surrogate_function=self.surrogate_func),  # 脉冲神经元
            layer.MaxPool2d(2, 2),  # 下采样，输出16×16

            # 卷积层2：channels→2*channels，3×3卷积， padding=1，输出16×16
            layer.Conv2d(channels, 2 * channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(2 * channels),
            neuron.IFNode(surrogate_function=self.surrogate_func),
            layer.MaxPool2d(2, 2),  # 下采样，输出8×8

            # 卷积层3：2*channels→4*channels，3×3卷积， padding=1，输出8×8
            layer.Conv2d(2 * channels, 4 * channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(4 * channels),
            neuron.IFNode(surrogate_function=self.surrogate_func),
            layer.MaxPool2d(2, 2),  # 下采样，输出4×4

            # 全连接层1：4*channels×4×4 → 128
            layer.Flatten(),  # 展平：4*channels×4×4 = 64*channels
            layer.Linear(4 * channels * 4 * 4, 128, bias=False),
            neuron.IFNode(surrogate_function=self.surrogate_func),

            # 全连接层2：128→10（CIFAR10共10类）
            layer.Linear(128, 10, bias=False),
            neuron.IFNode(surrogate_function=self.surrogate_func)
        )

        # 设置多步模式（m-step），适配时间序列输入
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输入静态图像→扩展时间维度→脉冲序列输出→计算平均发放率
        Args:
            x: 静态图像张量，shape=[N, 3, 32, 32]（N=批量大小）
        Returns:
            fr: 输出层平均脉冲发放率，shape=[N, 10]
        """
        # 扩展时间维度：[N,3,32,32] → [T, N, 3, 32, 32]（多步输入）
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # 脉冲序列前向传播
        x_seq_out = self.conv_fc(x_seq)
        # 计算时间维度的平均发放率（激活驱动核心：以发放率为优化目标）
        fr = x_seq_out.mean(dim=0)  # 对时间步T求平均，shape=[N,10]
        return fr

    def reset(self):
        """重置神经元状态（多步训练后需重置，避免状态累积）"""
        # 只重置子模块，避免递归调用
        functional.reset_net(self.conv_fc)
