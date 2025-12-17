"""
工具函数模块：全局变量管理、数据预处理、损失函数等
基于对现有代码的理解重新实现
"""
import torch
import sys
import os

# 添加项目根目录到路径，以便导入数据集加载函数
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class GlobalVars:
    """全局变量管理类，类似原项目的global_v.py"""
    def __init__(self):
        self.T = None
        self.syn_a = None
        self.delta_syn_a = None
        self.rank = 0
        self.network_config = None
        self.layers_config = None
        self.init_flag = False
        self.outputs_raw = None
        self.time_use = 0
        
    def init(self, config_n, config_l=None):
        """初始化全局变量，计算syn_a和delta_syn_a"""
        self.network_config = config_n
        self.layers_config = config_l
        
        # 设置默认值
        if 'loss_reverse' not in self.network_config:
            self.network_config['loss_reverse'] = True
        if 'encoding' not in self.network_config:
            self.network_config['encoding'] = 'None'
        if 'amp' not in self.network_config:
            self.network_config['amp'] = False
        if 'backend' not in self.network_config:
            self.network_config['backend'] = 'python'
        if 'norm_grad' not in self.network_config:
            self.network_config['norm_grad'] = 1
        if 'max_dudt_inv' not in self.network_config:
            self.network_config['max_dudt_inv'] = 123456789
        if 'avg_spike_init' not in self.network_config:
            self.network_config['avg_spike_init'] = 1
        if 'weight_decay' not in self.network_config:
            self.network_config['weight_decay'] = 0
        if 't_train' not in self.network_config:
            self.network_config['t_train'] = self.network_config['n_steps']
        if 'forward_type' not in self.network_config:
            self.network_config['forward_type'] = 'leaky'
            
        # 验证配置
        assert self.network_config['forward_type'] in ['leaky', 'nonleaky']
        assert self.network_config['gradient_type'] in ['original', 'exponential']
        assert not (self.network_config['forward_type'] == 'nonleaky' and 
                   self.network_config['gradient_type'] == 'original')
        
        # 计算syn_a和delta_syn_a
        T = self.network_config['n_steps']
        tau_s = self.network_config['tau_s']
        tau_m = self.network_config['tau_m']
        grad_type = self.network_config['gradient_type']
        
        self.T = T
        self.syn_a = torch.zeros(T + 1, device=torch.device(self.rank))
        self.delta_syn_a = torch.zeros(T + 1, device=torch.device(self.rank))
        
        theta_m = 1.0 / tau_m
        theta_s = 1.0 / tau_s
        
        if grad_type == 'exponential':
            assert 'tau_grad' in self.network_config
            tau_grad = self.network_config['tau_grad']
            theta_grad = 1.0 / tau_grad
        else:
            theta_grad = None
            
        # 计算syn_a和delta_syn_a
        for t in range(T):
            t1 = t + 1
            # syn_a[t] = ((1-theta_m)^(t+1) - (1-theta_s)^(t+1)) * theta_s / (theta_s - theta_m)
            self.syn_a[t] = ((1 - theta_m) ** t1 - (1 - theta_s) ** t1) * theta_s / (theta_s - theta_m)
            
            if grad_type == 'exponential':
                # delta_syn_a[t] = (1-theta_grad)^(t+1)
                self.delta_syn_a[t] = (1 - theta_grad) ** t1
            else:
                # original梯度类型
                def f(t_val):
                    return ((1 - theta_m) ** t_val - (1 - theta_s) ** t_val) * theta_s / (theta_s - theta_m)
                self.delta_syn_a[t] = f(t1) - f(t1 - 1)


# 全局变量实例
glv = GlobalVars()


def TTFS(data, T):
    """
    Time-to-First-Spike编码
    将输入数据编码为脉冲序列，脉冲时间与输入强度成反比
    data: 输入数据，可以是任意形状
    T: 时间步数
    返回: (T, *data.shape) 的脉冲序列
    """
    device = data.device
    # 归一化到[0, 1]（假设输入已经归一化，这里只是确保范围）
    data = data.clamp(0, 1)
    # 计算首次脉冲时间：值越大，脉冲时间越早
    # 使用线性映射：t = T * (1 - data)，但需要处理边界
    spike_times = (T * (1 - data)).long().clamp(0, T - 1)
    
    # 创建脉冲序列
    spikes = torch.zeros(T, *data.shape, device=device, dtype=data.dtype)
    for t in range(T):
        spikes[t] = (spike_times == t).float()
    
    return spikes


def initialize_layer(layer, spikes):
    """
    初始化层权重，确保神经元有合理的脉冲发放率
    参考原始实现的initialize函数
    """
    avg_spike_init = glv.network_config['avg_spike_init']
    from math import sqrt
    T = spikes.shape[0]
    t_start = T * 2 // 3
    
    low, high = 0.05, 500
    while high / low >= 1.01:
        mid = sqrt(high * low)
        layer.norm_weight.data *= mid
        outputs = layer.forward(spikes, None)
        layer.norm_weight.data /= mid
        n_neuron = outputs[0].numel()
        avg_spike = torch.sum(outputs[t_start:]) / n_neuron
        if avg_spike > avg_spike_init / T * (T - t_start) * 1.2:
            high = mid
        else:
            low = mid
    layer.norm_weight.data *= mid
    return layer.forward(spikes, None)


def preprocess_inputs(inputs, network_config, T):
    """
    数据预处理：将输入转换为脉冲序列格式
    inputs: (batch_size, C, H, W) 或 (batch_size, features)
    返回: (T, batch_size, ...) 格式的脉冲序列
    """
    device = inputs.device
    inputs = inputs.to(device)
    
    if network_config.get('encoding') == 'TTFS':
        # TTFS编码
        inputs = torch.stack([TTFS(data, T) for data in inputs], dim=1)
    else:
        # 直接编码：将像素值作为输入电流，重复T次
        if len(inputs.shape) == 4:  # 图像 (batch_size, C, H, W)
            inputs = inputs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        elif len(inputs.shape) == 2:  # 特征向量 (batch_size, features)
            inputs = inputs.unsqueeze(0).repeat(T, 1, 1)
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")
    
    return inputs
