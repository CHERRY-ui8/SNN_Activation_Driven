"""
损失函数实现
基于原始实现的loss_count，确保逻辑正确
"""
import torch
from torch.cuda.amp import custom_fwd, custom_bwd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_custom import glv


class LossCount(torch.autograd.Function):
    """
    Spike count损失函数
    与原始实现保持一致
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        """
        output: (T, batch_size, n_classes) - 脉冲序列
        target: (batch_size, n_classes) - 目标脉冲数
        返回: (T, batch_size, n_classes) - delta，用于计算损失
        """
        desired_count = glv.network_config['desired_count']
        undesired_count = glv.network_config['undesired_count']
        T = output.shape[0]
        out_count = torch.sum(output, dim=0)  # (batch_size, n_classes)
        
        # 计算差值
        delta = (out_count - target) / T
        
        # 关键mask逻辑：只在需要更多脉冲但实际更少，或需要更少脉冲但实际更多时计算损失
        # (target == desired_count) & (delta > 0): 目标类别期望更多脉冲但实际更少
        # (target == undesired_count) & (delta < 0): 非目标类别期望更少脉冲但实际更多
        delta[(target == desired_count) & (delta > 0) | (target == undesired_count) & (delta < 0)] = 0
        
        # 扩展到时间维度
        delta = delta.unsqueeze(0).repeat(T, 1, 1)
        return delta
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        """
        反向传播：根据loss_reverse选项决定符号
        """
        sign = -1 if glv.network_config['loss_reverse'] else 1
        return sign * grad, None


class SpikeLoss(torch.nn.Module):
    """Spike损失函数模块"""
    
    def __init__(self):
        super(SpikeLoss, self).__init__()
    
    def spike_count(self, output, target):
        """
        Spike count损失
        output: (T, batch_size, n_classes)
        target: (batch_size, n_classes)
        """
        delta = LossCount.apply(output, target)
        return 0.5 * torch.sum(delta ** 2)
