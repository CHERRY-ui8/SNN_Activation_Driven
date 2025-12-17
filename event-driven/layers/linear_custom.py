"""
线性层实现（包含神经元和批归一化）
基于对事件驱动反向传播的理解重新实现
"""
import torch
import torch.nn as nn
import sys
import os

# 导入工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_custom import glv
from layers.neuron_custom import neuron_forward, neuron_backward
from torch.cuda.amp import custom_fwd, custom_bwd


def bn_forward(inputs, weight, norm_weight, norm_bias):
    """
    批归一化前向传播
    对权重进行归一化处理
    """
    C = weight.shape[0]  # 输出通道数
    
    # 计算权重的均值和方差
    mean = torch.mean(weight.reshape(C, -1), dim=1)
    var = torch.std(weight.reshape(C, -1), dim=1) ** 2
    
    # 调整形状以匹配权重维度
    if len(weight.shape) == 4:  # 卷积层
        shape = (-1, 1, 1, 1)
    else:  # 线性层
        shape = (-1, 1)
    
    mean = mean.reshape(*shape)
    var = var.reshape(*shape)
    norm_weight = norm_weight.reshape(*shape)
    norm_bias = norm_bias.reshape(*shape)
    
    # 归一化：weight_ = (weight - mean) / sqrt(var + eps) * norm_weight + norm_bias
    weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias
    
    return inputs, mean, var, weight_


def bn_backward(grad_weight, weight, norm_weight, norm_bias, mean, var):
    """
    批归一化反向传播
    """
    C = weight.shape[0]
    std_inv = 1.0 / torch.sqrt(var + 1e-5)
    
    # 调整形状
    if len(weight.shape) == 4:
        shape = (-1, 1, 1, 1)
    else:
        shape = (-1, 1)
    
    # 计算归一化后的权重
    weight_ = (weight - mean) * std_inv * norm_weight.reshape(*shape) + norm_bias.reshape(*shape)
    
    # 计算批归一化参数的梯度
    grad_bn_b = torch.sum(grad_weight.reshape(C, -1), dim=1).reshape(norm_bias.shape)
    grad_bn_w = torch.sum((grad_weight * weight_).reshape(C, -1), dim=1).reshape(norm_weight.shape)
    
    # 计算权重的梯度
    grad_weight = grad_weight * norm_weight.reshape(*shape)
    m = weight.numel() // C
    
    grad_var = grad_weight * (weight - mean) / m * (-0.5) * std_inv ** 3
    grad_mean = -grad_weight * std_inv
    grad_weight = grad_weight * std_inv + grad_var * 2 * (weight - mean) / m + grad_mean / m
    
    return grad_weight, grad_bn_w, grad_bn_b


class LinearFunc(torch.autograd.Function):
    """线性层的自定义autograd函数"""
    
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, norm_weight, norm_bias, neuron_config, labels):
        """
        前向传播
        
        参数:
            inputs: (T, batch_size, n_inputs) - 输入脉冲序列
            weight: (n_outputs, n_inputs) - 权重矩阵
            norm_weight: (n_outputs, 1) - 批归一化权重
            norm_bias: (n_outputs, 1) - 批归一化偏置
            neuron_config: (theta_m, theta_s, theta_grad, threshold) - 神经元配置
            labels: (batch_size,) - 标签（用于输出层的监督信号）
        
        返回:
            outputs: (T, batch_size, n_outputs) - 输出脉冲序列
        """
        # 批归一化
        inputs, mean, var, weight_ = bn_forward(inputs, weight, norm_weight, norm_bias)
        
        # 线性变换：in_I = inputs @ weight_.T
        in_I = torch.matmul(inputs, weight_.t())  # (T, batch_size, n_outputs)
        
        # 神经元前向传播
        delta_u, delta_u_t, outputs = neuron_forward(in_I, neuron_config)
        
        # 如果是输出层且有标签，注入监督信号
        if labels is not None:
            T, n_batch, N = in_I.shape
            glv.outputs_raw = outputs.clone()
            
            # 在膜电位增加时注入脉冲（监督信号）
            i2 = torch.arange(n_batch, device=outputs.device)
            # 找到膜电位增加的时刻
            is_inc = (delta_u[:, i2, labels] > 0.05).float()
            # 找到第一个增加的时刻
            _, i1 = torch.max(is_inc * torch.arange(1, T + 1, device=is_inc.device).unsqueeze(-1), dim=0)
            # 在该时刻注入脉冲
            outputs[i1, i2, labels] = (delta_u[i1, i2, labels] != 0).to(outputs.dtype)
        
        # 保存用于反向传播的中间变量
        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs, weight, norm_weight, norm_bias, mean, var)
        ctx.is_out_layer = labels is not None
        
        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        """
        反向传播
        
        参数:
            grad_delta: (T, batch_size, n_outputs) - 来自损失函数的梯度
        
        返回:
            grad_input: (T, batch_size, n_inputs) - 输入梯度
            grad_weight: (n_outputs, n_inputs) - 权重梯度
            grad_bn_w: (n_outputs, 1) - 批归一化权重梯度
            grad_bn_b: (n_outputs, 1) - 批归一化偏置梯度
        """
        # 获取保存的中间变量
        (delta_u, delta_u_t, inputs, outputs, weight, norm_weight, norm_bias, mean, var) = ctx.saved_tensors
        
        # 事件驱动：只在脉冲时刻有梯度
        grad_delta = grad_delta * outputs
        
        # 神经元反向传播
        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)
        
        # 计算归一化后的权重（用于输入梯度计算）
        weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias
        
        # 计算输入梯度：grad_input = grad_in_ @ weight_ * inputs
        # 这里inputs用于事件驱动的梯度传播
        grad_input = torch.matmul(grad_in_, weight_) * inputs
        
        # 计算权重梯度：grad_weight = sum_t(grad_w_[t].T @ inputs[t])
        grad_weight = torch.sum(torch.matmul(grad_w_.transpose(1, 2), inputs), dim=0)
        
        # 批归一化反向传播
        grad_weight, grad_bn_w, grad_bn_b = bn_backward(grad_weight, weight, norm_weight, norm_bias, mean, var)
        
        # 返回梯度（输入梯度乘以0.85，这是原实现中的做法）
        return grad_input * 0.85, grad_weight, grad_bn_w, grad_bn_b, None, None


class LinearLayer(nn.Module):
    """线性层模块"""
    
    def __init__(self, network_config, config, name):
        """
        初始化线性层
        
        参数:
            network_config: 网络配置
            config: 层配置 {'n_inputs': int, 'n_outputs': int, 'threshold': float}
            name: 层名称
        """
        super(LinearLayer, self).__init__()
        
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.threshold = config.get('threshold', 1.0)
        self.name = name
        self.type = 'linear'
        
        # 创建权重矩阵（不使用bias）
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        # 批归一化参数
        self.norm_weight = nn.Parameter(torch.ones(out_features, 1))
        self.norm_bias = nn.Parameter(torch.zeros(out_features, 1))
        
        print(f"Linear layer: {name}")
        print(f"  Input features: {in_features}")
        print(f"  Output features: {out_features}")
        print(f"  Weight shape: {list(self.weight.shape)}")
        print("-----------------------------------------")
    
    def forward(self, x, labels=None):
        """
        前向传播
        
        参数:
            x: (T, batch_size, n_inputs) 或 (T, batch_size, C, H, W)
            labels: (batch_size,) - 标签（仅输出层需要）
        """
        # 如果是5维（图像），展平为3维（需要在初始化检查之前处理）
        if len(x.shape) == 5:
            T, n_batch, C, H, W = x.shape
            x = x.view(T, n_batch, C * H * W)
        
        # 检查是否需要初始化（在初始化阶段，需要完整的forward来计算脉冲率）
        if glv.init_flag:
            # 初始化阶段也需要完整的forward，但不需要labels
            config_n = glv.network_config
            theta_m = 1.0 / config_n['tau_m']
            theta_s = 1.0 / config_n['tau_s']
            theta_grad = 1.0 / config_n['tau_grad'] if config_n['gradient_type'] == 'exponential' else -123456789
            neuron_config = (theta_m, theta_s, theta_grad, self.threshold)
            return LinearFunc.apply(x, self.weight, self.norm_weight, self.norm_bias, neuron_config, None)
        
        # 获取神经元配置
        config_n = glv.network_config
        theta_m = 1.0 / config_n['tau_m']
        theta_s = 1.0 / config_n['tau_s']
        theta_grad = 1.0 / config_n['tau_grad'] if config_n['gradient_type'] == 'exponential' else -123456789
        
        neuron_config = (theta_m, theta_s, theta_grad, self.threshold)
        
        # 调用自定义函数
        return LinearFunc.apply(x, self.weight, self.norm_weight, self.norm_bias, neuron_config, labels)
    
    def weight_clipper(self):
        """权重裁剪"""
        self.weight.data = self.weight.data.clamp(-4, 4)
