"""
梯度流可视化脚本
用于验证事件驱动反向传播的正确性
绘制各层梯度随时间的变化，展示脉冲事件与梯度计算的对应关系
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_custom import glv, preprocess_inputs
from layers.linear_custom import LinearLayer
from train_mnist_custom import SimpleMLP, get_loss, readout
import yaml


def register_gradient_hooks(network):
    """注册梯度钩子，用于收集梯度信息"""
    gradients = {}
    
    def make_hook(name):
        def hook(grad):
            gradients[name] = grad.detach().cpu().numpy()
            return grad
        return hook
    
    # 为每一层的权重注册钩子
    for name, module in network.named_modules():
        if isinstance(module, LinearLayer):
            if module.weight.requires_grad:
                module.weight.register_hook(make_hook(f'{name}.weight'))
    
    return gradients


def visualize_gradient_flow(network, inputs, labels, network_config, device, save_dir):
    """可视化梯度流"""
    network.train()
    T = network_config['n_steps']
    
    # 注册梯度钩子
    gradients = register_gradient_hooks(network)
    
    # 前向传播
    outputs = network(inputs, labels)
    
    # 计算损失
    loss = get_loss(outputs, labels, network_config)
    
    # 反向传播
    network.zero_grad()
    loss.backward()
    
    # 收集梯度信息
    grad_info = {}
    for name, grad in gradients.items():
        if grad is not None:
            # 计算梯度的统计信息
            grad_info[name] = {
                'mean': np.mean(np.abs(grad)),
                'std': np.std(grad),
                'max': np.max(np.abs(grad)),
                'min': np.min(np.abs(grad)),
                'shape': grad.shape
            }
    
    # 绘制梯度统计信息
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 各层梯度均值随时间的变化
    ax = axes[0, 0]
    layer_names = list(grad_info.keys())
    grad_means = [grad_info[name]['mean'] for name in layer_names]
    ax.bar(range(len(layer_names)), grad_means)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Gradient Magnitude')
    ax.set_title('Gradient Magnitude by Layer')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels([name.split('.')[0] for name in layer_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 2. 梯度分布
    ax = axes[0, 1]
    all_grads = []
    for name, grad in gradients.items():
        if grad is not None:
            all_grads.extend(grad.flatten())
    ax.hist(all_grads, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Gradient Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Gradient Distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. 脉冲输出可视化
    ax = axes[1, 0]
    spike_counts = readout(outputs, T)
    predicted = torch.argmax(spike_counts, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    
    # 绘制每个样本的输出脉冲数
    output_spikes = torch.sum(outputs, dim=0).cpu().numpy()  # (batch_size, n_classes)
    im = ax.imshow(output_spikes[:20].T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Class')
    ax.set_title(f'Output Spikes (First 20 samples, Acc: {100*correct/total:.1f}%)')
    plt.colorbar(im, ax=ax)
    
    # 4. 梯度随时间的变化（如果有时间维度）
    ax = axes[1, 1]
    # 计算每个时间步的梯度总和
    time_grads = []
    for t in range(T):
        grad_sum = 0
        for name, grad in gradients.items():
            if grad is not None:
                grad_sum += np.sum(np.abs(grad))
        time_grads.append(grad_sum)
    
    ax.plot(range(T), time_grads, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Total Gradient Magnitude')
    ax.set_title('Gradient Flow Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradient_flow.png'), dpi=300, bbox_inches='tight')
    print(f"Gradient flow visualization saved to {save_dir}/gradient_flow.png")
    
    # 保存梯度信息到文件
    import json
    grad_summary = {}
    for name, info in grad_info.items():
        grad_summary[name] = {
            'mean': float(info['mean']),
            'std': float(info['std']),
            'max': float(info['max']),
            'min': float(info['min']),
            'shape': list(info['shape'])
        }
    
    with open(os.path.join(save_dir, 'gradient_info.json'), 'w') as f:
        json.dump(grad_summary, f, indent=2)
    
    print(f"Gradient information saved to {save_dir}/gradient_info.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='Path to config file')
    parser.add_argument('-checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('-num_samples', type=int, default=32, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # 读取配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    network_config = config['Network']
    layers_config = config['Layers']
    
    # 初始化全局变量
    glv.rank = 0 if torch.cuda.is_available() else -1
    glv.init(network_config, layers_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    import torchvision
    import torchvision.transforms as transforms
    
    data_path = network_config['data_path']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    testset = torchvision.datasets.MNIST(
        data_path, train=False, transform=transform, download=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.num_samples, shuffle=False
    )
    
    # 创建网络
    network = SimpleMLP(layers_config).to(device)
    
    # 加载检查点（如果有）
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        network.load_state_dict(checkpoint['net'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # 获取一个批次的数据
    inputs, labels = next(iter(testloader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # 数据预处理
    T = network_config['n_steps']
    inputs = preprocess_inputs(inputs, network_config, T)
    
    # 创建保存目录
    save_dir = f"gradient_visualization_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化梯度流
    print("Visualizing gradient flow...")
    visualize_gradient_flow(network, inputs, labels, network_config, device, save_dir)
    
    print("Visualization completed!")


if __name__ == '__main__':
    from datetime import datetime
    main()
