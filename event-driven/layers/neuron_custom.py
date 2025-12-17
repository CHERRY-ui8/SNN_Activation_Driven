"""
神经元前向和反向传播实现
基于对事件驱动反向传播算法的理解重新实现
核心思想：只在神经元发放脉冲时才计算和传播梯度
"""
import torch
import sys
import os

# 导入工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_custom import glv


@torch.jit.script
def neuron_forward_custom(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp):
    """
    神经元前向传播
    
    参数:
        in_I: (T, batch_size, n_neurons) - 输入电流序列
        theta_m: 膜时间常数倒数
        theta_s: 突触时间常数倒数
        theta_grad: 梯度时间常数倒数（用于反向传播）
        threshold: 脉冲发放阈值
        is_forward_leaky: 是否使用leaky模式
        is_grad_exp: 是否使用指数梯度
    
    返回:
        delta_u: (T, batch_size, n_neurons) - 膜电位增量
        delta_u_t: (T, batch_size, n_neurons) - 用于反向传播的时间增量
        outputs: (T, batch_size, n_neurons) - 脉冲输出序列
    """
    # 初始化状态变量
    u_last = torch.zeros_like(in_I[0])  # 上一时刻的膜电位
    syn_m = torch.zeros_like(in_I[0])   # 膜时间常数相关的累积
    syn_s = torch.zeros_like(in_I[0])   # 突触时间常数相关的累积
    syn_grad = torch.zeros_like(in_I[0])  # 梯度时间常数相关的累积
    
    T = in_I.shape[0]
    delta_u = torch.zeros_like(in_I)
    delta_u_t = torch.zeros_like(in_I)
    outputs = torch.zeros_like(in_I)
    
    # 前向传播：遍历每个时间步
    for t in range(T):
        # 更新累积变量（leaky积分器）
        syn_m = (syn_m + in_I[t]) * (1 - theta_m)
        syn_s = (syn_s + in_I[t]) * (1 - theta_s)
        syn_grad = (syn_grad + in_I[t]) * (1 - theta_grad)
        
        # 计算膜电位
        if not is_forward_leaky:
            # Non-leaky模式：直接使用syn_grad
            delta_u_t[t] = syn_grad
            u = u_last + delta_u_t[t]
            delta_u[t] = delta_u_t[t]
        else:
            # Leaky模式：使用syn_m和syn_s计算膜电位
            # u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
            u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
            delta_u[t] = u - u_last
            # delta_u_t用于反向传播，可以是syn_grad或delta_u
            delta_u_t[t] = syn_grad if is_grad_exp else delta_u[t]
        
        # 检查是否发放脉冲
        out = (u >= threshold).to(u.dtype)
        
        # 如果发放脉冲，重置状态
        u_last = u * (1 - out)
        syn_m = syn_m * (1 - out)
        syn_s = syn_s * (1 - out)
        syn_grad = syn_grad * (1 - out)
        
        outputs[t] = out
    
    return delta_u, delta_u_t, outputs


@torch.jit.script
def neuron_backward_custom(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv):
    """
    神经元反向传播（事件驱动）
    
    核心思想：只在脉冲时刻计算和传播梯度
    
    参数:
        grad_delta: (T, batch_size, n_neurons) - 来自上一层的梯度
        outputs: (T, batch_size, n_neurons) - 前向传播的脉冲输出
        delta_u: (T, batch_size, n_neurons) - 膜电位增量
        delta_u_t: (T, batch_size, n_neurons) - 时间增量
        syn_a: (T+1,) - 突触响应函数
        partial_a: (T+1,) - 偏导数函数
        max_dudt_inv: 最大dudt的倒数（用于梯度裁剪）
    
    返回:
        grad_in_: (T, batch_size, n_neurons) - 输入梯度
        grad_w_: (T, batch_size, n_neurons) - 权重梯度
    """
    T = grad_delta.shape[0]
    
    # 初始化梯度
    grad_in_ = torch.zeros_like(outputs)
    grad_w_ = torch.zeros_like(outputs)
    
    # 用于累积的变量
    partial_u_grad_w = torch.zeros_like(outputs[0])  # 权重相关的偏导数累积
    partial_u_grad_t = torch.zeros_like(outputs[0])  # 时间相关的偏导数累积
    delta_t = torch.zeros(outputs.shape[1:], device=outputs.device, dtype=torch.long)  # 距离上次脉冲的时间
    spiked = torch.zeros_like(outputs[0])  # 是否已发放过脉冲
    
    # 反向传播：从最后一个时间步向前遍历
    for t in range(T - 1, -1, -1):
        out = outputs[t]  # 当前时刻的脉冲
        
        # 更新spiked标志：一旦发放过脉冲，就一直为True
        spiked = spiked + (1 - spiked) * out
        
        # 计算偏导数
        # partial_u = -1 / delta_u[t]，限制在[-4, 0]范围内
        partial_u = torch.clamp(-1.0 / delta_u[t], -4.0, 0.0)
        # partial_u_t = -1 / delta_u_t[t]，限制在[-max_dudt_inv, 0]范围内
        partial_u_t = torch.clamp(-1.0 / delta_u_t[t], -max_dudt_inv, 0.0)
        
        # 更新累积的偏导数
        # 如果当前时刻发放脉冲，则更新；否则保持之前的值
        partial_u_grad_w = partial_u_grad_w * (1 - out) + grad_delta[t] * partial_u * out
        partial_u_grad_t = partial_u_grad_t * (1 - out) + grad_delta[t] * partial_u_t * out
        
        # 更新delta_t：如果发放脉冲则重置为0，否则加1
        delta_t = (delta_t + 1) * (1 - out).long()
        
        # 计算梯度（只在已发放过脉冲的神经元上计算）
        # grad_in_[t] = partial_u_grad_t * partial_a[delta_t] * spiked
        grad_in_[t] = partial_u_grad_t * partial_a[delta_t] * spiked.to(partial_a.dtype)
        # grad_w_[t] = partial_u_grad_w * syn_a[delta_t] * spiked
        grad_w_[t] = partial_u_grad_w * syn_a[delta_t] * spiked.to(syn_a.dtype)
    
    return grad_in_, grad_w_


def neuron_forward(in_I, neuron_config):
    """
    神经元前向传播的包装函数
    neuron_config: (theta_m, theta_s, theta_grad, threshold)
    """
    theta_m, theta_s, theta_grad, threshold = neuron_config
    theta_m = torch.tensor(theta_m, device=in_I.device, dtype=in_I.dtype)
    theta_s = torch.tensor(theta_s, device=in_I.device, dtype=in_I.dtype)
    theta_grad = torch.tensor(theta_grad, device=in_I.device, dtype=in_I.dtype)
    threshold = torch.tensor(threshold, device=in_I.device, dtype=in_I.dtype)
    
    assert theta_m != theta_s, "theta_m and theta_s must be different"
    
    is_grad_exp = torch.tensor(glv.network_config['gradient_type'] == 'exponential', device=in_I.device)
    is_forward_leaky = torch.tensor(glv.network_config['forward_type'] == 'leaky', device=in_I.device)
    
    return neuron_forward_custom(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp)


def neuron_backward(grad_delta, outputs, delta_u, delta_u_t):
    """
    神经元反向传播的包装函数
    """
    syn_a = glv.syn_a.to(outputs.device)
    partial_a = -glv.delta_syn_a.to(outputs.device)  # partial_a = -delta_syn_a
    max_dudt_inv = torch.tensor(glv.network_config['max_dudt_inv'], device=outputs.device, dtype=outputs.dtype)
    
    return neuron_backward_custom(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv)
