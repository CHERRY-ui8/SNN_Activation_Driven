"""
替代梯度函数可视化脚本
1. surrogate_gradients_comparison.png - 四种替代梯度函数对比
2. sigmoid_beta_comparison.png - Sigmoid不同β值对比
3. esser_beta_comparison.png - Esser不同β值对比
4. superspike_beta_comparison.png - SuperSpike不同β值对比
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.surrogate.surrogate_custom import SigmoidSurrogate, EsserSurrogate, SuperSpikeSurrogate
from spikingjelly.activation_based.surrogate import ATan


def compute_gradient(surrogate_func, x):
    """计算替代梯度"""
    x_grad = x.clone().detach().requires_grad_(True)
    y = surrogate_func(x_grad)
    y.sum().backward()
    grad = x_grad.grad.clone()
    return grad


def plot_surrogate_gradients_comparison():
    """绘制四种替代梯度函数对比图"""
    print("正在生成：surrogate_gradients_comparison.png")
    
    # 生成测试数据：膜电位相对于阈值的值（x = v - v_threshold）
    x = torch.linspace(-2.0, 2.0, 1000)
    
    # 定义替代梯度函数（使用训练时的参数）
    surrogates = {
        'ATan (Default)': ATan(),
        'Sigmoid': SigmoidSurrogate(beta=5.0),
        'Esser': EsserSurrogate(beta=1.0),
        'SuperSpike': SuperSpikeSurrogate(beta=2.0)
    }
    
    plt.figure(figsize=(12, 6))
    for name, surr in surrogates.items():
        x_grad = x.clone().detach().requires_grad_(True)
        y = surr(x_grad)
        y.sum().backward()
        grad = x_grad.grad.clone()
        
        plt.plot(x.numpy(), grad.numpy(), label=name, linewidth=2)
    
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (x=0)')
    plt.xlabel('Membrane Potential Relative to Threshold (x = v - v_threshold)', fontsize=12)
    plt.ylabel('Surrogate Gradient h(x)', fontsize=12)
    plt.title('Comparison of Surrogate Gradient Functions (Current Parameter Settings)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2.0, 2.0)
    
    # 保存图片
    save_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'surrogate_gradients_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存到：{save_path}")


def plot_sigmoid_beta_comparison():
    """绘制Sigmoid不同β值对比图"""
    print("正在生成：sigmoid_beta_comparison.png")
    
    x = torch.linspace(-2.0, 2.0, 1000)
    betas = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    
    plt.figure(figsize=(12, 6))
    for beta in betas:
        surr = SigmoidSurrogate(beta=beta)
        x_grad = x.clone().detach().requires_grad_(True)
        y = surr(x_grad)
        y.sum().backward()
        grad = x_grad.grad.clone()
        
        plt.plot(x.numpy(), grad.numpy(), label=f'β={beta}', linewidth=2)
    
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (x=0)')
    plt.xlabel('Membrane Potential Relative to Threshold (x = v - v_threshold)', fontsize=12)
    plt.ylabel('Surrogate Gradient h(x)', fontsize=12)
    plt.title('Sigmoid Surrogate Gradient Function: Effect of Different β Values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2.0, 2.0)
    
    save_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'sigmoid_beta_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存到：{save_path}")


def plot_esser_beta_comparison():
    """绘制Esser不同β值对比图"""
    print("正在生成：esser_beta_comparison.png")
    
    x = torch.linspace(-2.0, 2.0, 1000)
    betas = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    plt.figure(figsize=(12, 6))
    for beta in betas:
        surr = EsserSurrogate(beta=beta)
        x_grad = x.clone().detach().requires_grad_(True)
        y = surr(x_grad)
        y.sum().backward()
        grad = x_grad.grad.clone()
        
        plt.plot(x.numpy(), grad.numpy(), label=f'β={beta}', linewidth=2)
    
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (x=0)')
    plt.xlabel('Membrane Potential Relative to Threshold (x = v - v_threshold)', fontsize=12)
    plt.ylabel('Surrogate Gradient h(x)', fontsize=12)
    plt.title('Esser Surrogate Gradient Function: Effect of Different β Values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2.0, 2.0)
    
    save_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'esser_beta_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存到：{save_path}")


def plot_superspike_beta_comparison():
    """绘制SuperSpike不同β值对比图"""
    print("正在生成：superspike_beta_comparison.png")
    
    x = torch.linspace(-2.0, 2.0, 1000)
    betas = [0.5, 1.0, 2.0, 4.0, 8.0, 10.0]
    
    plt.figure(figsize=(12, 6))
    for beta in betas:
        surr = SuperSpikeSurrogate(beta=beta)
        x_grad = x.clone().detach().requires_grad_(True)
        y = surr(x_grad)
        y.sum().backward()
        grad = x_grad.grad.clone()
        
        plt.plot(x.numpy(), grad.numpy(), label=f'β={beta}', linewidth=2)
    
    plt.axvline(x=0.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (x=0)')
    plt.xlabel('Membrane Potential Relative to Threshold (x = v - v_threshold)', fontsize=12)
    plt.ylabel('Surrogate Gradient h(x)', fontsize=12)
    plt.title('SuperSpike Surrogate Gradient Function: Effect of Different β Values', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2.0, 2.0)
    
    save_dir = os.path.join(project_root, 'reports', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'superspike_beta_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存到：{save_path}")


def main():
    """生成所有可视化图片"""
    print("=" * 60)
    print("替代梯度函数可视化")
    print("=" * 60)
    print()
    
    # 生成所有图片
    plot_surrogate_gradients_comparison()
    print()
    plot_sigmoid_beta_comparison()
    print()
    plot_esser_beta_comparison()
    print()
    plot_superspike_beta_comparison()
    print()
    
    print("=" * 60)
    print("所有可视化图片已生成完成！")
    print("保存位置：reports/figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()
