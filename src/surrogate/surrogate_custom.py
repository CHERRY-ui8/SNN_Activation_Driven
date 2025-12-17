"""
论文"The remarkable robustness of surrogate gradient learning..." 4.2.6节要求的3种替代梯度函数
包含：Sigmoid、Esser（分段线性）、SuperSpike（渐近变体）
继承spikingjelly的SurrogateFunctionBase基类，保证接口统一
"""
import torch
import math
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase


class SigmoidSurrogateFunction(torch.autograd.Function):
    """Sigmoid替代梯度函数的autograd Function"""
    @staticmethod
    def forward(ctx, x, beta, threshold):
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.threshold = threshold
        return (x >= 0.0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        beta = ctx.beta
        s = torch.sigmoid(beta * x)
        grad = beta * s * (1 - s)
        return grad_output * grad, None, None


class EsserSurrogateFunction(torch.autograd.Function):
    """Esser替代梯度函数的autograd Function"""
    @staticmethod
    def forward(ctx, x, beta, threshold):
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.threshold = threshold
        result = (x >= 0.0).float()
        
        # 调试信息
        if not hasattr(EsserSurrogateFunction.forward, '_debug_printed'):
            print(f"[DEBUG EsserFunction.forward] x: requires_grad={x.requires_grad}, shape: {x.shape}, mean: {x.mean().item():.4f}")
            print(f"[DEBUG EsserFunction.forward] beta: {beta}, threshold: {threshold}")
            print(f"[DEBUG EsserFunction.forward] result mean: {result.mean().item():.4f}, sum: {result.sum().item()}")
            EsserSurrogateFunction.forward._debug_printed = True
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        beta = ctx.beta
        
        # 调试信息
        if not hasattr(EsserSurrogateFunction.backward, '_debug_printed'):
            print(f"[DEBUG EsserFunction.backward] CALLED! grad_output: {grad_output is not None}, shape: {grad_output.shape if grad_output is not None else None}")
            if grad_output is not None:
                print(f"[DEBUG EsserFunction.backward] grad_output mean: {grad_output.mean().item():.6f}, max: {grad_output.max().item():.6f}, min: {grad_output.min().item():.6f}")
            print(f"[DEBUG EsserFunction.backward] x: requires_grad={x.requires_grad}, shape: {x.shape}, mean: {x.mean().item():.4f}")
            print(f"[DEBUG EsserFunction.backward] beta: {beta}")
            EsserSurrogateFunction.backward._debug_printed = True
        
        grad = torch.clamp(1.0 - beta * torch.abs(x), min=0.0)
        result = grad_output * grad
        
        if not hasattr(EsserSurrogateFunction.backward, '_debug_grad_printed'):
            print(f"[DEBUG EsserFunction.backward] grad mean: {grad.mean().item():.6f}, max: {grad.max().item():.6f}, min: {grad.min().item():.6f}")
            print(f"[DEBUG EsserFunction.backward] grad non-zero count: {(grad > 0).sum().item()}/{grad.numel()}")
            print(f"[DEBUG EsserFunction.backward] grad_output * grad (before sum): mean={result.mean().item():.6f}, sum={result.sum().item():.6f}")
            print(f"[DEBUG EsserFunction.backward] result mean: {result.mean().item():.6f}, max: {result.max().item():.6f}, min: {result.min().item():.6f}")
            EsserSurrogateFunction.backward._debug_grad_printed = True
        
        return result, None, None


class SuperSpikeSurrogateFunction(torch.autograd.Function):
    """SuperSpike替代梯度函数的autograd Function（渐近变体）"""
    @staticmethod
    def forward(ctx, x, beta, threshold):
        ctx.save_for_backward(x)
        ctx.beta = beta
        ctx.threshold = threshold
        return (x >= 0.0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None
        if len(ctx.saved_tensors) == 0:
            return None, None, None
        x, = ctx.saved_tensors
        beta = ctx.beta
        # 确保x和grad_output类型匹配
        if x.dtype != grad_output.dtype:
            x = x.to(grad_output.dtype)
        denominator = (torch.abs(x) * beta + 1.0) ** 2
        grad = beta / denominator
        return grad_output * grad, None, None


class SigmoidSurrogate(SurrogateFunctionBase):
    """Sigmoid替代梯度函数：h(x) = β·s(x)·(1-s(x))，其中s(x)是Sigmoid函数，x=U[n]-1
    
    根据论文公式：h(x) = s(x)(1-s(x))，其中 s(x) = 1/(1+exp(-βx))
    实际实现中乘以β以控制梯度强度：h(x) = β·s(x)·(1-s(x))
    """
    def __init__(self, beta: float = 5.0, spiking: bool = True, threshold: float = 1.0):
        """
        Args:
            beta: 控制Sigmoid函数的陡峭程度（β越大，梯度越集中在阈值附近）
            spiking: 是否使用脉冲模式
            threshold: 脉冲发放阈值（默认1.0，适配IF神经元）
        """
        super().__init__(beta, spiking)
        self.beta = beta
        self.threshold = threshold

    @staticmethod
    def spiking_function(x, beta):
        """前向传播：IF神经元的脉冲发放逻辑"""
        return SigmoidSurrogateFunction.apply(x, beta, 0.0)
    
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, beta: float):
        # 原始函数：Sigmoid的积分形式（x已经是v - v_threshold）
        return torch.sigmoid(beta * x)
    
    @staticmethod
    def backward(grad_output, x, beta):
        """反向传播：Sigmoid替代梯度计算 h(x) = β·s(x)·(1-s(x))，其中s(x) = sigmoid(β·x)"""
        s = torch.sigmoid(beta * x)
        grad = beta * s * (1 - s)
        return grad_output * grad


class EsserSurrogate(SurrogateFunctionBase):
    """Esser替代梯度函数（分段线性）：h(x) = max(0, 1 - β·|x|)，x=U[n]-1"""
    def __init__(self, beta: float = 10.0, spiking: bool = True, threshold: float = 1.0):
        """
        Args:
            beta: 控制梯度的有效范围（β越大，梯度集中在阈值附近的范围越窄）
            spiking: 是否使用脉冲模式
            threshold: 脉冲发放阈值（默认1.0）
        """
        # 将beta作为alpha传入基类（spikingjelly要求alpha参数）
        super().__init__(beta, spiking)
        self.beta = beta
        self.threshold = threshold

    @staticmethod
    def spiking_function(x, beta):
        """前向传播：IF神经元脉冲发放"""
        # 调试信息
        if not hasattr(EsserSurrogate.spiking_function, '_debug_printed'):
            print(f"[DEBUG EsserSurrogate.spiking_function] CALLED! x: requires_grad={x.requires_grad}, shape: {x.shape}, mean: {x.mean().item():.4f}")
            print(f"[DEBUG EsserSurrogate.spiking_function] beta: {beta}")
            EsserSurrogate.spiking_function._debug_printed = True
        
        result = EsserSurrogateFunction.apply(x, beta, 0.0)
        
        if not hasattr(EsserSurrogate.spiking_function, '_debug_result_printed'):
            print(f"[DEBUG EsserSurrogate.spiking_function] result: requires_grad={result.requires_grad}, grad_fn: {type(result.grad_fn).__name__ if result.grad_fn else None}")
            EsserSurrogate.spiking_function._debug_result_printed = True
        
        return result
    
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, beta: float):
        # 原始函数：分段线性函数的积分形式（x已经是v - v_threshold）
        return torch.clamp(x * (1.0 - beta * torch.abs(x) / 2.0) + 0.5, min=0.0, max=1.0)
    
    @staticmethod
    def backward(grad_output, x, beta):
        """反向传播：分段线性替代梯度计算 h(x) = max(0, 1 - β·|x|)"""
        grad = torch.clamp(1.0 - beta * torch.abs(x), min=0.0)
        return grad_output * grad


class SuperSpikeSurrogate(SurrogateFunctionBase):
    """SuperSpike替代梯度函数（渐近变体）：h(x) = β / (|x|·β + 1)^2，x=U[n]-1
    
    根据论文：渐近变体定义为 h(x) = β / (β|x| + 1)^2
    """
    def __init__(self, beta: float = 2.0, spiking: bool = True, threshold: float = 1.0):
        """
        Args:
            beta: 控制梯度的衰减速度（β越大，梯度衰减越快）
            spiking: 是否使用脉冲模式
            threshold: 脉冲发放阈值（默认1.0）
        """
        super().__init__(beta, spiking)
        self.beta = beta
        self.threshold = threshold

    @staticmethod
    def spiking_function(x, beta):
        """前向传播：IF神经元脉冲发放"""
        return SuperSpikeSurrogateFunction.apply(x, beta, 0.0)
    
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, beta: float):
        # 原始函数：SuperSpike的积分形式（x已经是v - v_threshold）
        abs_x = torch.abs(x)
        return torch.sign(x) * (abs_x / (abs_x * beta + 1.0)) + 0.5
    
    @staticmethod
    def backward(grad_output, x, beta):
        """反向传播：渐近变体梯度计算 h(x) = β / (|x|·β + 1)^2"""
        denominator = (torch.abs(x) * beta + 1.0) ** 2
        grad = beta / denominator
        return grad_output * grad


# 测试代码（验证替代梯度函数的计算正确性）
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 生成膜电位数据（围绕阈值1.0）
    x = torch.linspace(0.0, 2.0, 1000, requires_grad=True)  # 膜电位U[n]从0到2
    surrogates = {
        'Sigmoid': SigmoidSurrogate(beta=5.0),
        'Esser': EsserSurrogate(beta=10.0),
        'SuperSpike': SuperSpikeSurrogate(beta=2.0)
    }

    # 绘制替代梯度曲线
    plt.figure(figsize=(10, 6))
    for name, surr in surrogates.items():
        # 通过反向传播计算梯度
        y = surr(x)
        y.sum().backward()
        grad = x.grad.clone()
        x.grad.zero_()
        plt.plot(x.detach().numpy(), grad.numpy(), label=name)
    plt.axvline(x=1.0, color='red', linestyle='--', label='阈值=1.0')
    plt.xlabel('膜电位 U[n]')
    plt.ylabel('替代梯度')
    plt.title('三种替代梯度函数的梯度分布')
    plt.legend()
    plt.grid(True)
    plt.savefig('./surrogate_gradients.png')  # 保存图片（可用于作业报告）
    print("替代梯度曲线已保存为 surrogate_gradients.png")
