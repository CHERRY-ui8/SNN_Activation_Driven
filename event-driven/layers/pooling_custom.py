"""
改进的池化层实现（Adjusted Average Pooling）
基于论文中的改进，确保梯度不变性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


def read_kernel_config(kernel_size):
    """读取核大小配置"""
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    elif isinstance(kernel_size, str):
        # 格式: "(2, 2)"
        assert kernel_size[0] == '(' and kernel_size[-1] == ')'
        data = kernel_size[1:-1]
        x, y = map(int, data.split(','))
        return (x, y)
    else:
        return kernel_size


class AdjustedAvgPoolFunc(torch.autograd.Function):
    """
    改进的平均池化函数
    前向传播使用标准平均池化
    反向传播使用改进的梯度分配策略，确保梯度不变性
    """
    
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, kernel):
        """
        前向传播：使用标准平均池化
        
        参数:
            inputs: (T*batch_size, C, H, W) - 输入特征图
            kernel: (kernel_h, kernel_w) - 池化核大小
        
        返回:
            outputs: (T*batch_size, C, H_out, W_out) - 池化后的特征图
        """
        outputs = F.avg_pool2d(inputs, kernel)
        # 保存用于反向传播的信息
        ctx.save_for_backward(outputs, torch.tensor(inputs.shape, dtype=torch.long), torch.tensor(kernel))
        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        """
        反向传播：使用改进的梯度分配策略
        
        核心思想：调整梯度分配，使得梯度总和保持不变（梯度不变性）
        """
        outputs, input_shape, kernel = ctx.saved_tensors
        kernel = kernel.tolist()
        input_shape = input_shape.tolist()
        
        # 改进的梯度分配策略
        # 计算调整因子：1 / outputs，但限制在合理范围内
        adjustment = 1.0 / (outputs + 1e-8)  # 避免除零
        
        # 限制调整因子：如果太大（超过kernel面积+1），则设为0
        kernel_area = kernel[0] * kernel[1]
        adjustment[adjustment > kernel_area + 1] = 0
        
        # 归一化调整因子
        adjustment = adjustment / kernel_area
        
        # 将梯度乘以调整因子，然后上采样回原始尺寸
        grad_adjusted = grad_delta * adjustment
        grad_input = F.interpolate(grad_adjusted, size=input_shape[2:], mode='nearest')
        
        return grad_input, None


class PoolLayer(nn.Module):
    """池化层模块"""
    
    def __init__(self, network_config, config, name):
        """
        初始化池化层
        
        参数:
            network_config: 网络配置
            config: 层配置 {'kernel_size': int or tuple}
            name: 层名称
        """
        super(PoolLayer, self).__init__()
        self.name = name
        self.type = 'pooling'
        self.network_config = network_config
        
        kernel_size = config['kernel_size']
        self.kernel = read_kernel_config(kernel_size)
        
        print(f"Pooling layer: {name}")
        print(f"  Kernel size: {self.kernel}")
        print("-----------------------------------------")
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (T, batch_size, C, H, W) - 输入特征图序列
        
        返回:
            y: (T, batch_size, C, H_out, W_out) - 池化后的特征图序列
        """
        pool_type = self.network_config.get('pooling_type', 'adjusted_avg')
        
        T, n_batch, C, H, W = x.shape
        # 重塑为 (T*batch_size, C, H, W) 以便使用标准池化函数
        x = x.reshape(T * n_batch, C, H, W)
        
        if pool_type == 'avg':
            # 标准平均池化
            x = F.avg_pool2d(x, self.kernel)
        elif pool_type == 'max':
            # 最大池化
            x = F.max_pool2d(x, self.kernel)
        elif pool_type == 'adjusted_avg':
            # 改进的平均池化
            x = AdjustedAvgPoolFunc.apply(x, self.kernel)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")
        
        # 重塑回 (T, batch_size, C, H_out, W_out)
        x = x.reshape(T, n_batch, *x.shape[1:])
        return x
