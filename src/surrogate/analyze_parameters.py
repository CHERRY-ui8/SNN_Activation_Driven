"""
替代梯度函数参数优化分析
分析不同参数值对梯度分布的影响，提供优化建议
"""
import os
import sys
import torch
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.surrogate.surrogate_custom import SigmoidPrimeSurrogate, EsserSurrogate, SuperSpikeSurrogate

def analyze_surrogate_gradients():
    """分析不同参数下的梯度分布"""
    # 生成测试数据：膜电位相对于阈值的值（x = v - v_threshold）
    x = torch.linspace(-2.0, 2.0, 1000, requires_grad=True)
    
    # 测试不同的参数值（统一使用beta）
    sigmoid_betas = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    esser_betas = [0.5, 1.0, 2.0, 5.0, 10.0]
    superspike_betas = [0.5, 1.0, 2.0, 4.0, 8.0, 10.0]
    
    print("=" * 80)
    print("替代梯度函数参数分析")
    print("=" * 80)
    
    # 1. Sigmoid 分析
    print("\n【Sigmoid 替代梯度函数分析】")
    print("公式: h(x) = s(x)·(1-s(x))，其中 s(x) = sigmoid(β·x)")
    print("论文默认: β=10（但实际使用中需要根据输入范围调整）")
    print("-" * 80)
    print(f"{'Beta':<8} {'Max Grad':<12} {'Grad@0':<12} {'Grad@±1':<12} {'Effective Range':<20}")
    print("-" * 80)
    
    sigmoid_results = []
    for beta in sigmoid_betas:
        surr = SigmoidPrimeSurrogate(beta=beta)
        y = surr(x)
        y.sum().backward()
        grad = x.grad.clone()
        x.grad.zero_()
        
        max_grad = grad.max().item()
        grad_at_0 = grad[500].item()  # x=0的位置
        grad_at_1 = grad[750].item()  # x=1的位置
        grad_at_neg1 = grad[250].item()  # x=-1的位置
        
        # 计算有效范围（梯度 > 0.1 * max_grad 的范围）
        threshold = 0.1 * max_grad
        effective_mask = grad > threshold
        effective_range = (x[effective_mask].max() - x[effective_mask].min()).item()
        
        sigmoid_results.append({
            'beta': beta,
            'max_grad': max_grad,
            'grad_at_0': grad_at_0,
            'grad_at_1': grad_at_1,
            'effective_range': effective_range
        })
        
        print(f"{beta:<8.1f} {max_grad:<12.6f} {grad_at_0:<12.6f} {grad_at_1:<12.6f} {effective_range:<20.4f}")
    
    # 推荐值分析
    best_sigmoid = max(sigmoid_results, key=lambda x: x['effective_range'] * x['max_grad'])
    print(f"\n推荐: beta={best_sigmoid['beta']:.1f} (平衡梯度强度和有效范围)")
    
    # 2. Esser 分析
    print("\n【Esser 替代梯度函数分析】")
    print("公式: h(x) = max(0, 1 - β·|x|)")
    print("-" * 80)
    print(f"{'Beta':<8} {'Max Grad':<12} {'Grad@0':<12} {'Effective Range':<20} {'Non-zero Ratio':<15}")
    print("-" * 80)
    
    esser_results = []
    for beta in esser_betas:
        surr = EsserSurrogate(beta=beta)
        y = surr(x)
        y.sum().backward()
        grad = x.grad.clone()
        x.grad.zero_()
        
        max_grad = grad.max().item()
        grad_at_0 = grad[500].item()  # x=0的位置
        effective_range = 2.0 / beta  # 理论值：|x| < 1/β
        non_zero_ratio = (grad > 0).float().mean().item()
        
        esser_results.append({
            'beta': beta,
            'max_grad': max_grad,
            'grad_at_0': grad_at_0,
            'effective_range': effective_range,
            'non_zero_ratio': non_zero_ratio
        })
        
        print(f"{beta:<8.1f} {max_grad:<12.6f} {grad_at_0:<12.6f} {effective_range:<20.4f} {non_zero_ratio:<15.4f}")
    
    # 推荐值分析（考虑实际输入范围）
    # 如果输入x的典型范围是[-1.5, 1.5]，那么beta应该使得有效范围覆盖这个区间
    best_esser = min(esser_results, key=lambda x: abs(x['effective_range'] - 3.0))
    print(f"\n推荐: beta={best_esser['beta']:.1f} (有效范围约{best_esser['effective_range']:.2f})")
    
    # 3. SuperSpike 分析
    print("\n【SuperSpike 替代梯度函数分析（渐近变体）】")
    print("公式: h(x) = β / (|x|·β + 1)^2")
    print("论文默认: β=10（但实际使用中需要根据输入范围调整）")
    print("-" * 80)
    print(f"{'Beta':<8} {'Max Grad':<12} {'Grad@0':<12} {'Grad@±1':<12} {'Grad@±2':<12}")
    print("-" * 80)
    
    superspike_results = []
    for beta in superspike_betas:
        surr = SuperSpikeSurrogate(beta=beta)
        y = surr(x)
        y.sum().backward()
        grad = x.grad.clone()
        x.grad.zero_()
        
        max_grad = grad.max().item()
        grad_at_0 = grad[500].item()  # x=0的位置
        grad_at_1 = grad[750].item()  # x=1的位置
        grad_at_2 = grad[1000-1].item()  # x=2的位置
        
        superspike_results.append({
            'beta': beta,
            'max_grad': max_grad,
            'grad_at_0': grad_at_0,
            'grad_at_1': grad_at_1,
            'grad_at_2': grad_at_2
        })
        
        print(f"{beta:<8.1f} {max_grad:<12.6f} {grad_at_0:<12.6f} {grad_at_1:<12.6f} {grad_at_2:<12.6f}")
    
    # 推荐值分析（考虑梯度衰减速度）
    # 选择在x=±1处仍有合理梯度的beta值
    best_superspike = max(superspike_results, 
                         key=lambda x: x['grad_at_1'] if x['grad_at_1'] > 0.1 * x['max_grad'] else 0)
    print(f"\n推荐: beta={best_superspike['beta']:.1f} (在x=±1处梯度={best_superspike['grad_at_1']:.6f})")
    
    # 4. 综合建议
    print("\n" + "=" * 80)
    print("【参数优化建议】")
    print("=" * 80)
    print("\n1. Sigmoid:")
    print(f"   - 当前值: beta=5.0")
    print(f"   - 推荐值: beta={best_sigmoid['beta']:.1f} (论文默认β=10，但需要根据输入范围调整)")
    print(f"   - 分析: β越大梯度越集中，但有效范围越窄")
    print(f"   - 建议: 如果输入x的典型范围是[-1.5, 1.5]，使用beta=3.0-6.0；如果已归一化，可使用beta=8.0-10.0")
    
    print("\n2. Esser:")
    print(f"   - 当前值: beta=1.0")
    print(f"   - 推荐值: beta={best_esser['beta']:.1f} (论文默认β=10，但需要根据输入范围调整)")
    print(f"   - 分析: β越大梯度范围越窄，论文中β=10假设输入已归一化到[-0.1, 0.1]")
    print(f"   - 建议: 如果输入x的典型范围是[-1.5, 1.5]，使用beta=0.5-2.0；如果已归一化，可使用beta=5.0-10.0")
    
    print("\n3. SuperSpike:")
    print(f"   - 当前值: beta=2.0")
    print(f"   - 推荐值: beta={best_superspike['beta']:.1f} (论文默认β=10，但需要根据输入范围调整)")
    print(f"   - 分析: β越大梯度衰减越快，但峰值梯度越大")
    print(f"   - 建议: 如果输入x的典型范围是[-1.5, 1.5]，使用beta=1.0-3.0；如果已归一化，可使用beta=5.0-10.0")
    
    print("\n" + "=" * 80)
    print("【实际训练建议】")
    print("=" * 80)
    print("基于当前训练结果（Esser: 69.87%, SuperSpike: 75.45%）:")
    print("\n建议的参数范围（基于实际输入范围[-1.5, 1.5]）：")
    print("  - Sigmoid: beta ∈ [3.0, 6.0]，推荐 4.0-5.0（当前5.0已较好）")
    print("  - Esser: beta ∈ [0.5, 2.0]，推荐 1.0-1.5（当前1.0已较好）")
    print("  - SuperSpike: beta ∈ [1.0, 3.0]，推荐 1.5-2.5（当前2.0已较好）")
    print("\n论文默认值（β=10）适用于输入已归一化到[-0.1, 0.1]的情况")
    print("\n可以尝试的参数组合：")
    print("  - Sigmoid(beta=4.0): 更平滑的梯度，可能提高稳定性")
    print("  - Esser(beta=1.5): 稍微扩大有效范围，可能提高收敛速度")
    print("  - SuperSpike(beta=1.5): 更平滑的梯度衰减，可能提高泛化能力")

if __name__ == '__main__':
    analyze_surrogate_gradients()

