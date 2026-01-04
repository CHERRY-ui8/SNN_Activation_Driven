import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .cuda_kernel.auto_cuda import cfunction
from .triton_kernel.torch2triton import compile_triton_code_str

tab4_str = '\t\t\t\t'  # used for aligning code
curly_bracket_l = '{'
curly_bracket_r = '}'


@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)


def check_manual_grad(primitive_function, spiking_function, *args, **kwargs):
    x = torch.arange(-2, 2, 32 / 8192)
    # x = torch.as_tensor([-1., 0., 1.])
    x.requires_grad_(True)
    primitive_function(x, *args, **kwargs).sum().backward()
    x_grad_auto = x.grad.clone()
    x.grad.zero_()
    spiking_function(x, *args, **kwargs).sum().backward()
    x_grad_manual = x.grad.clone()
    print('auto   grad', x_grad_auto)
    print('manual grad', x_grad_manual)
    abs_error = (x_grad_manual - x_grad_auto).abs()
    idx = abs_error.argmax()
    print('max error', abs_error[idx], 'occurs at')
    print(f'x[{idx}] = {x[idx]}')
    print('auto   grad', x_grad_auto[idx])
    print('manual grad', x_grad_manual[idx])


def check_cuda_grad(neu, surrogate_function, device, *args, **kwargs):
    # check_cuda_grad(neuron.IFNode, surrogate.S2NN, device='cuda:1', alpha=4., beta=1.)
    for dtype in [torch.float, torch.half]:
        print(dtype)
        net = neu(surrogate_function=surrogate_function(*args, **kwargs), step_mode='m')
        net.to(device)
        x = torch.arange(-2, 2, 32 / 8192, device=device, dtype=dtype)
        x.requires_grad_(True)
        net.backend = 'torch'
        net(x.unsqueeze(0)).sum().backward()
        x_grad_py = x.grad.clone()
        x.grad.zero_()
        net.reset()
        net.backend = 'cupy'
        net(x.unsqueeze(0)).sum().backward()

        x_grad_cp = x.grad.clone()
        # print('python grad', x_grad_py)
        # print('cupy   grad', x_grad_cp)
        abs_error = (x_grad_cp - x_grad_py).abs()
        idx = abs_error.argmax()
        print('max error', abs_error[idx], 'occurs at')
        print(f'x[{idx}] = {x[idx]}')
        print('python grad', x_grad_py[idx])
        print('cupy   grad', x_grad_cp[idx])


def plot_surrogate_function(surrogate_function):
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use(['science', 'muted', 'grid'])
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


class SurrogateFunctionBase(nn.Module):
    def __init__(self, alpha, spiking=True):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplementedError

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)

    def cuda_codes(self, y: str, x: str, dtype: str):
        # new version
        raise NotImplementedError

    def triton_codes(self):
        raise NotImplementedError


class MultiArgsSurrogateFunctionBase(nn.Module):
    def __init__(self, spiking: bool, *args, **kwargs):
        super().__init__()
        self.spiking = spiking

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.activation_based.surrogate.{self._get_name()}.cuda_code'

    def cuda_codes(self, y: str, x: str, dtype: str):
        # new version
        raise NotImplementedError

    def triton_codes(self):
        raise NotImplementedError


@torch.jit.script
def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    x_abs = x.abs()
    mask = (x_abs > (1 / alpha))
    grad_x = (grad_output * (- (alpha ** 2) * x_abs + alpha)).masked_fill_(mask, 0)
    return grad_x, None


class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class PiecewiseQuadratic(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_quadratic.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask0 = (x > (1.0 / alpha)).to(x)
        mask1 = (x.abs() <= (1.0 / alpha)).to(x)

        return mask0 + mask1 * (-(alpha ** 2) / 2 * x.square() * x.sign() + alpha * x + 0.5)

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_quadratic_backward(grad_output, x, alpha)[0]


@torch.jit.script
def piecewise_exp_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 * (- alpha * x.abs()).exp_() * grad_output, None


class piecewise_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_exp_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class PiecewiseExp(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_exp.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2. - 1.
        exp_x = (mask_sign * x * -alpha).exp_() / 2.
        return mask_nonnegative - exp_x * mask_sign

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_exp_backward(grad_output, x, alpha)[0]


@torch.jit.script
def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None


sigmoid_backward_triton_template = """
@triton.jit
def sigmoid_surrogate_{alpha_str}(h):
    # triton's exp() supports only fp32 and fp64, so we must convert it to fp32!
    sg = tl.sigmoid(h.to(tl.float32) * {alpha})
    sg = {alpha} * sg * (1. - sg)
    return sg.to(h.dtype)
"""


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (x * alpha).sigmoid()

    @staticmethod
    def backward(grad_output, x, alpha):
        return sigmoid_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha};
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {y} = __hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha);
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.sigmoid_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

    def triton_codes(self):
        alpha_str = str(self.alpha).replace(".", "_")
        code_str = sigmoid_backward_triton_template.format(
            alpha=self.alpha, alpha_str=alpha_str
        ).strip()
        kernel_name = f"sigmoid_surrogate_{alpha_str}"
        return compile_triton_code_str(code_str, kernel_name)


@torch.jit.script
def soft_sign_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (2 * alpha * (1 / alpha + x.abs()).pow_(2)), None


class soft_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return soft_sign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class SoftSign(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0'

    @staticmethod
    def spiking_function(x, alpha):
        return soft_sign.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (F.softsign(x * alpha) + 1.) / 2.

    @staticmethod
    def backward(grad_output, x, alpha):
        return soft_sign_backward(grad_output, x, alpha)[0]


@torch.jit.script
def super_spike_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha * grad_output / torch.pow(torch.abs(x) + 1., 2), None

class super_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return super_spike_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class SuperSpike(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        raise NotImplementedError

    @staticmethod
    def backward(grad_output, x, alpha):
        return super_spike_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError
    def cuda_codes(self, y: str, x: str, dtype: str):
        raise NotImplementedError


@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    a = alpha / 2
    ax = math.pi * a * x
    return a / (1 + ax*ax) * grad_output, None


atan_backward_triton_template = """
@triton.jit
def atan_surrogate_{alpha_str}(h):
    sg = 3.141592653589793 * h * {alpha} / 2.
    sg = {alpha} / 2. / tl.fma(sg, sg, 1.)
    return sg.to(h.dtype)
"""


class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class ATan(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    @staticmethod
    def backward(grad_output, x, alpha):
        return atan_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''
        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_M_PI_2__alpha__x = ((float) 1.57079632679489661923) * {alpha} * {x};
            {tab4_str}const float {y} = {alpha} / 2.0f / (1.0f + {sg_name}_M_PI_2__alpha__x * {sg_name}_M_PI_2__alpha__x);
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha =  __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_M_PI_2__alpha__x = __hmul2(__hmul2(__float2half2_rn((float) 1.57079632679489661923), {sg_name}_alpha), {x});
            {tab4_str}const half2 {y} = __h2div(__h2div({sg_name}_alpha, __float2half2_rn(2.0f)), __hfma2({sg_name}_M_PI_2__alpha__x, {sg_name}_M_PI_2__alpha__x, __float2half2_rn(1.0f)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.atan_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

    def triton_codes(self):
        alpha_str = str(self.alpha).replace(".", "_")
        code_str = atan_backward_triton_template.format(
            alpha=self.alpha,
            alpha_str=alpha_str,
        ).strip()
        kernel_name = f"atan_surrogate_{alpha_str}"
        return compile_triton_code_str(code_str, kernel_name)



@torch.jit.script
def nonzero_sign_log_abs_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (1 / alpha + x.abs()), None


class nonzero_sign_log_abs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return nonzero_sign_log_abs_backward((grad_output, ctx.saved_tensors[0], ctx.alpha))


class NonzeroSignLogAbs(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return nonzero_sign_log_abs.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return nonzero_sign_log_abs_backward(grad_output, x, alpha)[0]

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        # the gradient of ``(heaviside(x) * 2 - 1) * (alpha * x.abs() + 1).log()`` by autograd is wrong at ``x==0``
        mask_p = heaviside(x) * 2. - 1.
        return mask_p * (alpha * mask_p * x + 1).log()


@torch.jit.script
def erf_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output * (- (x * alpha).pow_(2)).exp_() * (alpha / math.sqrt(math.pi)), None


class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return erf_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Erf(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return erf.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.erfc_(-alpha * x) / 2.

    @staticmethod
    def backward(grad_output, x, alpha):
        return erf_backward(grad_output, x, alpha)[0]


@torch.jit.script
def piecewise_leaky_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, w: float, c: float):
    mask_width = (x.abs() < w)
    mask_c = mask_width.logical_not()
    return grad_output * x.masked_fill(mask_width, 1 / (2*w)).masked_fill(mask_c, c), None, None


class piecewise_leaky_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w=1, c=0.01):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.w = w
            ctx.c = c
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_leaky_relu_backward(grad_output, ctx.saved_tensors[0], ctx.w, ctx.c)


class PiecewiseLeakyReLU(MultiArgsSurrogateFunctionBase):
    def __init__(self, w=1., c=0.01, spiking=True):
        super().__init__(spiking)
        assert w > 0.
        self.w = w
        self.c = c
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.w, self.c)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return piecewise_leaky_relu.apply(x, w, c)

    @staticmethod
    def backward(grad_output, x, w, c):
        return piecewise_leaky_relu_backward(grad_output, x, w, c)[0]

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, w: float, c: float):
        mask0 = (x < -w).to(x)
        mask1 = (x > w).to(x)
        mask2 = torch.ones_like(x.data) - mask0 - mask1
        if c == 0:
            return mask2 * (x / (2 * w) + 1 / 2) + mask1
        else:
            cw = c * w
            return mask0 * (c * x + cw) + mask1 * (c * x + (- cw + 1)) \
                   + mask2 * (x / (2 * w) + 1 / 2)

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1. / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_x_abs = fabsf({x});
            float {y};
            if ({sg_name}_x_abs > {w})
            {curly_bracket_l}
                {y} = {c};
            {curly_bracket_r}
            else
            {curly_bracket_l}
                {y} = {w_inv};
            {curly_bracket_r}
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_x_abs = __habs2({x});
            {tab4_str}const half2 {sg_name}_x_abs_ge_w = __hge2({sg_name}_x_abs, __float2half2_rn({w}));
            {tab4_str}half2 {y} = __hadd2(__hmul2(__float2half2_rn({c}),  {sg_name}_x_abs_ge_w), __hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_x_abs_ge_w), __float2half2_rn({w_inv})));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.piecewise_leaky_relu_backward(y=y, x=x, w=self.w, c=self.c, dtype=dtype)


class squarewave_fourier_series(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, n: int, T_period: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.n = n
            ctx.T_period = T_period
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = 0.
        x = ctx.saved_tensors[0]
        w = math.pi * 2. / ctx.T_period
        for i in range(1, ctx.n):
            grad_x += torch.cos_((2 * i - 1.) * w * x)

        grad_x *= 4. / ctx.T_period
        grad_x *= grad_output

        return grad_x, None, None


class SquarewaveFourierSeries(MultiArgsSurrogateFunctionBase):
    def __init__(self, n: int = 2, T_period: float = 8, spiking=True):
        super().__init__(spiking)
        assert isinstance(n, int) and T_period > 0.
        self.n = n
        self.T_period = T_period
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function
        return f(x, self.n, self.T_period)

    @staticmethod
    def spiking_function(x: torch.Tensor, w, c):
        return squarewave_fourier_series.apply(x, w, c)

    @staticmethod
    def primitive_function(x: torch.Tensor, n: int, T_period: float):
        w = math.pi * 2. / T_period
        ret = torch.zeros_like(x.data)
        for i in range(1, n):
            c = (2 * i - 1.)
            ret += torch.sin(c * w * x) / c

        return 0.5 + 2. / math.pi * ret

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        w = str(self.w) + 'f'
        w_inv = str(1. / self.w) + 'f'
        c = str(self.c) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            raise NotImplementedError
        elif dtype == 'fp16':
            raise NotImplementedError
        else:
            raise NotImplementedError

        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    # import torch
    # from spikingjelly.activation_based import surrogate
    # from matplotlib import pyplot as plt
    # plt.style.use(['science', 'muted', 'grid'])
    # fig = plt.figure(dpi=200, figsize=(6, 4))
    # x = torch.arange(-2.5, 2.5, 0.001)
    # plt.plot(x.data, surrogate.heaviside(x), label='Heaviside', linestyle='-.')
    #
    # c_list = []
    # for n in [2, 4, 8]:
    #     surrogate_function = surrogate.SquarewaveFourierSeries(n=n, T_period=8, spiking=False)
    #     y = surrogate_function(x)
    #     plt.plot(x.data, y.data, label=f'Primitive, $n={n}$')
    #     c_list.append(plt.gca().lines[-1].get_color())
    #
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title(f'SquarewaveFourierSeries surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # # plt.grid(linestyle='--')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries1.pdf')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries1.svg')
    # plt.clf()
    # for i, n in enumerate([2, 4, 8]):
    #     surrogate_function = surrogate.SquarewaveFourierSeries(n=n, T_period=8, spiking=True)
    #     x = x.detach()
    #     x.requires_grad_(True)
    #     y = surrogate_function(x)
    #     z = y.sum()
    #     z.backward()
    #     plt.plot(x.data, x.grad, label=f'Gradient, $n={n}$', c=c_list[i])
    #     x.grad.zero_()
    #
    # plt.xlim(-2, 2)
    # plt.legend()
    # plt.title(f'SquarewaveFourierSeries surrogate function')
    # plt.xlabel('Input')
    # plt.ylabel('Output')
    # # plt.grid(linestyle='--')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries2.pdf')
    # plt.savefig('./docs/source/_static/API/activation_based/surrogate/SquarewaveFourierSeries2.svg')


class s2nn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, beta: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.beta = beta
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sgax = torch.sigmoid(ctx.alpha * x)
        grad_x = torch.where(x < 0., ctx.alpha * sgax * (1. - sgax), ctx.beta / (x + 1.))
        return grad_x * grad_output, None, None


class S2NN(MultiArgsSurrogateFunctionBase):
    def __init__(self, alpha=4., beta=1., spiking=True):
        super().__init__(spiking)
        self.alpha = alpha
        self.beta = beta
        self.spiking = spiking

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.alpha, self.beta)

    @staticmethod
    def spiking_function(x: torch.Tensor, alpha, beta):
        return s2nn.apply(x, alpha, beta)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float, beta: float):
        return torch.where(x < 0., torch.sigmoid(x * alpha), beta * torch.log((x + 1.).abs_() + 1e-5) + 0.5)
        # abs and 1e-5 are used to avoid nan

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        beta = str(self.beta) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_sigmoid_ax = 1.0f / (1.0f + expf(- {alpha} * {x}));
            {tab4_str}const float {sg_name}_mask_l = (float)({x} < 0.0f);
            {tab4_str}const float {y} = (1.0f - {sg_name}_sigmoid_ax) * {sg_name}_sigmoid_ax * {alpha} * {sg_name}_mask_l + {beta} / ({x} + 1.0f) * (1.0f - {sg_name}_mask_l);
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_sigmoid_ax = __h2div(__float2half2_rn(1.0f), __hadd2(h2exp(__hneg2(__hmul2({sg_name}_alpha, {x}))), __float2half2_rn(1.0f)));
            {tab4_str}const half2 {sg_name}_mask_l = __hlt2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hadd2(__hmul2(__hmul2(__hmul2(__hsub2(__float2half2_rn(1.0f), {sg_name}_sigmoid_ax), {sg_name}_sigmoid_ax), {sg_name}_alpha), {sg_name}_mask_l), __hmul2(__h2div(__float2half2_rn({beta}), __hadd2({x}, __float2half2_rn(1.0f))), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask_l)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.s2nn_backward(y=y, x=x, alpha=self.alpha, beta=self.beta, dtype=dtype)


class q_pseudo_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        x = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_x = ((1 + 2 / (ctx.alpha - 1) * x.abs()).pow_(-ctx.alpha)) * grad_output
        return grad_x, None


class QPseudoSpike(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return q_pseudo_spike.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2. - 1.

        return mask_nonnegative - mask_sign * (0.5 * ((1. + 2. / (alpha - 1.) * x * mask_sign).pow_(1. - alpha)))

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_base = 1.0f + 2.0f / ({alpha} - 1.0f) * fabsf({x});
            {tab4_str}const float {y} = powf({sg_name}_base, -{alpha});
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_alpha = __float2half2_rn({alpha});
            {tab4_str}const half2 {sg_name}_base = __hadd2(__float2half2_rn(1.0f), __h2div(__hmul2(__float2half2_rn(2.0f), __habs2({x})), __hsub2({sg_name}_alpha, __float2half2_rn(1.0f))));
            {tab4_str}const half2 {y} = h2exp2(__hmul2(h2log2({sg_name}_base), __hneg2({sg_name}_alpha))); // Replace power with combination of log and exp, since CUDA has no power function for FP16.
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.q_pseudo_spike_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def leaky_k_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, leak: float, k: float):
    mask1 = (x >= 0.).to(x)
    grad_x = mask1 * k + (1. - mask1) * leak
    return grad_output * grad_x, None, None


class leaky_k_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, leak, k):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.leak = leak
            ctx.k = k
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return leaky_k_relu_backward(grad_output, ctx.saved_tensors[0], ctx.leak, ctx.k)


class LeakyKReLU(MultiArgsSurrogateFunctionBase):
    def __init__(self, spiking=True, leak: float = 0., k: float = 1.):
        super().__init__(spiking, leak, k)
        self.leak = leak
        self.k = k

    @staticmethod
    def spiking_function(x, leak, k):
        return leaky_k_relu.apply(x, leak, k)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, leak: float, k: float):
        mask1 = (x >= 0.).to(x)
        return (leak * (1. - mask1) + k * mask1) * x

    @staticmethod
    def backward(grad_output, x, leak, k):
        return leaky_k_relu_backward(grad_output, x, leak, k)[0]

    def forward(self, x):
        if self.spiking:
            f = self.spiking_function
        else:
            f = self.primitive_function

        return f(x, self.leak, self.k)

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        leak = str(self.leak) + 'f'
        k = str(self.k) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_mask1 = (float) ({x} >= 0.0f);
            {tab4_str}const float {y} = {leak} * (1.0f - {sg_name}_mask1) + {k} * {sg_name}_mask1;
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_mask1 = __hgeu2({x}, __float2half2_rn(0.0f));
            {tab4_str}const half2 {y} = __hfma2(__float2half2_rn({k}), {sg_name}_mask1, __hmul2(__float2half2_rn({leak}), __hsub2(__float2half2_rn(1.0f), {sg_name}_mask1)));
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.leaky_k_relu_backward(y=y, x=x, leak=self.leak, k=self.k, dtype=dtype)


@torch.jit.script
def fake_numerical_gradient_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    grad_x = torch.clamp_max(((x >= 0.) * 2. - 1.) / x, alpha)
    return grad_output * grad_x, None


class fake_numerical_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return fake_numerical_gradient_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class FakeNumericalGradient(SurrogateFunctionBase):
    def __init__(self, alpha=0.3):
        super().__init__(alpha, spiking=True)

    @staticmethod
    def spiking_function(x, alpha):
        return fake_numerical_gradient.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return fake_numerical_gradient_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}const float {sg_name}_sign = (float) ({x} >= 0.0f) * 2.0f - 1.0f;
            {tab4_str}const float {y} = min({sg_name}_sign / {x}, {alpha});
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half2 {sg_name}_sign = __hfma2(__hgeu2({x}, __float2half2_rn(0.0f)), __float2half2_rn(2.0f), __float2half2_rn(-1.0f));
            #if (__CUDA_ARCH__ < 800)
            {tab4_str}const half2 {sg_name}_grad_x = __h2div({sg_name}_sign, {x});
            {tab4_str}const half2 {sg_name}_grad_max = __float2half2_rn({alpha});
            {tab4_str}const half2 {y} = make_half2({sg_name}_grad_x.x <= {sg_name}_grad_max.x ? {sg_name}_grad_x.x : {sg_name}_grad_max.x, {sg_name}_grad_x.y <= {sg_name}_grad_max.y ? {sg_name}_grad_x.y : {sg_name}_grad_max.y);
            #else
            {tab4_str}const half2 {y} = __hmin2(__h2div({sg_name}_sign, {x}), __float2half2_rn({alpha}));
            #endif
            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code


    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.fake_numerical_gradient_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def log_tailed_relu_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    mask_gt1 = x > 1.
    mask_le0 = x <= 0.
    grad_x = torch.ones_like(grad_output)
    grad_x[mask_gt1] = 1. / x[mask_gt1]
    grad_x[mask_le0] = alpha
    return grad_output * grad_x, None


class log_tailed_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return log_tailed_relu_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class LogTailedReLU(SurrogateFunctionBase):
    def __init__(self, alpha=0., spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return log_tailed_relu.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_ge1 = (x > 1.).to(x)
        y = (1. - mask_ge1) * F.leaky_relu(x, alpha) + mask_ge1 * (torch.log(x) + 1.)
        return y

    @staticmethod
    def backward(grad_output, x, alpha):
        return log_tailed_relu_backward(grad_output, x, alpha)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        alpha = str(self.alpha) + 'f'
        code = f'''
            {tab4_str}{self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
            {tab4_str}float {y} = 0.0f;
            {tab4_str}if({x} <= 0.0f)
            {tab4_str}{curly_bracket_l}{y} = {alpha};{curly_bracket_r}
            {tab4_str}else if({x} <= 1.0f)
            {tab4_str}{curly_bracket_l}{y} = 1.0f;{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{y} = 1.0f / {x};{curly_bracket_r}
            '''
        elif dtype == 'fp16':
            code += f'''
            {tab4_str}const half {sg_name}_alpha = __float2half_rn({alpha});

            {tab4_str}half {sg_name}_{y}_low;
            {tab4_str}const half {sg_name}_{x}_low = __low2half({x});
            {tab4_str}if(__hle({sg_name}_{x}_low, __float2half_rn(0.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = {sg_name}_alpha;{curly_bracket_r}
            {tab4_str}else if(__hle({sg_name}_{x}_low, __float2half_rn(1.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = __float2half_rn(1.0f);{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_low = __hdiv(__float2half_rn(1.0f), {sg_name}_{x}_low);{curly_bracket_r}

            {tab4_str}half {sg_name}_{y}_high;
            {tab4_str}const half {sg_name}_{x}_high = __high2half({x});
            {tab4_str}if(__hle({sg_name}_{x}_high, __float2half_rn(0.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = {sg_name}_alpha;{curly_bracket_r}
            {tab4_str}else if(__hle({sg_name}_{x}_high, __float2half_rn(1.0f)))
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = __float2half_rn(1.0f);{curly_bracket_r}
            {tab4_str}else
            {tab4_str}{curly_bracket_l}{sg_name}_{y}_high = __hdiv(__float2half_rn(1.0f), {sg_name}_{x}_high);{curly_bracket_r}


            {tab4_str}const half2 {y} = __halves2half2({sg_name}_{y}_low, {sg_name}_{y}_high);

            '''
        else:
            raise NotImplementedError
        code += f'''
            {tab4_str}{self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.log_tailed_relu_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)


@torch.jit.script
def deterministic_pass_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output, None


class deterministic_pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return deterministic_pass_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class DeterministicPass(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return deterministic_pass.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return x

    @staticmethod
    def backward(grad_output, x, alpha):
        return deterministic_pass_backward(grad_output, x, alpha)[0]


@torch.jit.script
def rect_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha * (x.abs() < 0.5 / alpha).to(x) * grad_output, None


class rect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return rect_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Rect(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rect.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.clamp(alpha * x + 0.5, min=0.0, max=1.0)

    @staticmethod
    def backward(grad_output, x, alpha):
        return rect_backward(grad_output, x, alpha)[0]


class poisson_pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


_has_cuda_ = [
    ATan,
    Sigmoid,
    PiecewiseLeakyReLU,
    S2NN,
    QPseudoSpike,
    LeakyKReLU,
    FakeNumericalGradient
]