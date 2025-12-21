"""
论文"The remarkable robustness of surrogate gradient learning..." 4.2.6节要求的3种替代梯度函数
包含：Sigmoid、Esser（分段线性）、SuperSpike（渐近变体）
继承spikingjelly的SurrogateFunctionBase基类，保证接口统一
"""
import torch
import math
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase, heaviside


def plot_surrogate_function(surrogate_function):
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use(['science', 'no-latex', 'muted', 'grid'])
    fig = plt.figure(dpi=200)
    x = torch.arange(-2.5, 2.5, 0.001)
    plt.plot(x.data, heaviside(x), label='Heaviside', linestyle='-.')

    surrogate_function.set_spiking_mode(False)
    y = surrogate_function(x)
    plt.plot(x.data, y.data, label='Primitive')

    surrogate_function.set_spiking_mode(True)
    x.requires_grad_(True)
    y = surrogate_function(x)
    z = y.sum()
    z.backward()
    plt.plot(x.data, x.grad, label='Gradient')

    plt.xlim(-2, 2)
    plt.legend()
    plt.title(f'{surrogate_function.__class__.__name__} surrogate function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(linestyle='--')
    plt.savefig(
        f"./{surrogate_function.__class__.__name__}.pdf", bbox_inches='tight'
    )
    plt.show()


@torch.jit.script
def superspike_backward(grad_output: torch.Tensor, x: torch.Tensor, beta: float):
    grad = (torch.abs(x) * beta + 1.0) ** -2
    return grad_output * grad, None

class SuperSpikeSurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.beta = beta
        return heaviside(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        beta = ctx.beta
        
        return superspike_backward(grad_output, x, beta)

class SuperSpikeSurrogate(SurrogateFunctionBase):
    """
    SuperSpike surrogate function: g'(x) = 1 / (|x|·β + 1)^2
    Corresponding primitive function: g(x) = (1 + βx / (1 + β|x|)) / β
    """
    def __init__(self, beta: float = 10.0, spiking: bool = True):
        super().__init__(beta, spiking)
        self.beta = beta

    @staticmethod
    def spiking_function(x, beta):
        return SuperSpikeSurrogateFunction.apply(x, beta)
    
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, beta: float):
        return (1 + beta * x / (1 + beta * torch.abs(x))) / beta
    
    @staticmethod
    def backward(grad_output, x, beta):
        return superspike_backward(grad_output, x, beta)[0]


@torch.jit.script
def sigmoid_prime_backward(grad_output: torch.Tensor, x: torch.Tensor, beta: float):
    s = torch.sigmoid(beta * x)
    grad = s * (1 - s)
    return grad_output * grad, None

class SigmoidPrimeSurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.beta = beta
        return heaviside(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        beta = ctx.beta
        
        return sigmoid_prime_backward(grad_output, x, beta)

class SigmoidPrimeSurrogate(SurrogateFunctionBase):
    """
    Sigmoid surrogate function: g'(x) = s(x)·(1-s(x))
    Corresponding primitive function: g(x) = sigmoid(βx) / β
    """
    def __init__(self, beta: float = 10.0, spiking: bool = True):
        super().__init__(beta, spiking)
        self.beta = beta

    @staticmethod
    def spiking_function(x, beta):
        return SigmoidPrimeSurrogateFunction.apply(x, beta)
    
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, beta: float):
        return torch.sigmoid(beta * x) / beta
    
    @staticmethod
    def backward(grad_output, x, beta):
        return sigmoid_prime_backward(grad_output, x, beta)[0]


@torch.jit.script
def esser_backward(grad_output: torch.Tensor, x: torch.Tensor, beta: float):
    grad = torch.clamp(1.0 - beta * torch.abs(x), min=0.0)
    return grad_output * grad, None

class EsserSurrogateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.beta = beta
        return heaviside(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        beta = ctx.beta
        
        return esser_backward(grad_output, x, beta)

class EsserSurrogate(SurrogateFunctionBase):
    """
    Esser surrogate function: g'(x) = max(0, 1 - β·|x|)
    Corresponding primitive function: g(x) = If(x < -1/beta, 0, x < 0, 0.5 * beta^2 * (x + 1/beta)^2, x < 1/beta, 1 - 0.5 * beta^2 * (x - 1/beta)^2, 1)
    """
    def __init__(self, beta: float = 10.0, spiking: bool = True):
        super().__init__(beta, spiking)
        self.beta = beta

    @staticmethod
    def spiking_function(x, beta):
        return EsserSurrogateFunction.apply(x, beta)
    
    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, beta: float):
        return torch.where(x < -1/beta, 0.0, torch.where(x < 0.0, 0.5 * beta**2 * (x + 1/beta)**2, torch.where(x < 1/beta, 1.0 - 0.5 * beta**2 * (x - 1/beta)**2, 1.0)))
    
    @staticmethod
    def backward(grad_output, x, beta):
        return esser_backward(grad_output, x, beta)[0]

