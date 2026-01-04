import torch
import torch.nn as nn
import torch.nn.functional as f
from layers.functions import (
    neuron_forward, neuron_backward,
    bn_forward, bn_backward,
    readConfig, initialize
)
import global_v as glv
import torch.backends.cudnn as cudnn
from torch.cuda.amp import custom_fwd, custom_bwd
from datetime import datetime


class ConvLayer(nn.Conv2d):
    """
    Convolutional layer with neuron dynamics and weight-based batch normalization.

    This layer extends nn.Conv2d by integrating spiking neuron behavior and
    event-driven backpropagation.
    """
    def __init__(self, network_config, config, name, groups=1):
        self.name = name
        self.threshold = config['threshold'] if 'threshold' in config else None
        self.type = config['type']

        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        padding = config['padding'] if 'padding' in config else 0
        stride = config['stride'] if 'stride' in config else 1
        dilation = config['dilation'] if 'dilation' in config else 1

        # Parse convolution hyperparameters
        self.kernel = readConfig(kernel_size, 'kernelSize')
        self.stride = readConfig(stride, 'stride')
        self.padding = readConfig(padding, 'stride')
        self.dilation = readConfig(dilation, 'stride')

        super(ConvLayer, self).__init__(
            in_features, out_features,
            self.kernel, self.stride,
            self.padding, self.dilation,
            groups, bias=False
        )

        # Register learnable parameters
        self.weight = torch.nn.Parameter(self.weight.cuda(), requires_grad=True)
        self.norm_weight = torch.nn.Parameter(torch.ones(out_features, 1, 1, 1, device='cuda'))
        self.norm_bias = torch.nn.Parameter(torch.zeros(out_features, 1, 1, 1, device='cuda'))

        print('conv')
        print(f'Shape of weight is {list(self.weight.shape)}')  # Cout * Cin * Hk * Wk
        print(f'stride = {self.stride}, padding = {self.padding}, dilation = {self.dilation}, groups = {self.groups}')
        print("-----------------------------------------")

    def forward(self, x):
        """
        Forward pass.

        During the initialization phase, this layer runs a special forward
        routine to estimate firing statistics.
        """
        if glv.init_flag:
            glv.init_flag = False
            x = initialize(self, x)
            glv.init_flag = True
            return x

        config_n = glv.network_config
        theta_m = 1 / config_n['tau_m']
        theta_s = 1 / config_n['tau_s']
        theta_grad = (
            1 / config_n['tau_grad']
            if config_n['gradient_type'] == 'exponential'
            else -123456789  # placeholder instead of None
        )

        y = ConvFunc.apply(
            x,
            self.weight,
            self.norm_weight,
            self.norm_bias,
            (self.bias, self.stride, self.padding, self.dilation, self.groups),
            (theta_m, theta_s, theta_grad, self.threshold)
        )
        return y

    def weight_clipper(self):
        """Clamp convolution weights to a fixed range."""
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w


class ConvFunc(torch.autograd.Function):
    """
    Custom autograd function for the convolutional layer.

    Combines convolution, weight normalization, neuron dynamics,
    and event-driven gradient propagation.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, norm_weight, norm_bias, conv_config, neuron_config):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (T, batch_size, C_in, H_in, W_in).
            weight: Convolution weight tensor.
            norm_weight, norm_bias: Batch normalization parameters.
            conv_config: Tuple (bias, stride, padding, dilation, groups).
            neuron_config: Neuron parameter tuple.

        Returns:
            outputs: Tensor of spike outputs with shape
                     (T, batch_size, C_out, H_out, W_out).
        """
        bias, stride, padding, dilation, groups = conv_config
        T, n_batch, C, H, W = inputs.shape

        # Apply batch normalization on weights
        inputs, mean, var, weight_ = bn_forward(inputs, weight, norm_weight, norm_bias)

        # Convolution over merged (T * batch) dimension
        in_I = f.conv2d(
            inputs.reshape(T * n_batch, C, H, W),
            weight_, bias, stride, padding, dilation, groups
        )
        _, C, H, W = in_I.shape
        in_I = in_I.reshape(T, n_batch, C, H, W)

        # Neuron forward dynamics
        delta_u, delta_u_t, outputs = neuron_forward(in_I, neuron_config)

        # Save tensors for backward pass
        ctx.save_for_backward(delta_u, delta_u_t, inputs, outputs,
                              weight, norm_weight, norm_bias, mean, var)
        ctx.conv_config = conv_config

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        """
        Backward pass.

        Gradients are propagated only at spike events and reconstructed
        using transposed convolution and unfold-based weight updates.
        """
        (
            delta_u, delta_u_t, inputs, outputs,
            weight, norm_weight, norm_bias, mean, var
        ) = ctx.saved_tensors
        bias, stride, padding, dilation, groups = ctx.conv_config

        # Event-driven masking
        grad_delta *= outputs

        # Neuron backward dynamics
        grad_in_, grad_w_ = neuron_backward(grad_delta, outputs, delta_u, delta_u_t)

        # Recompute normalized weights
        weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias

        T, n_batch, C, H, W = grad_delta.shape
        inputs = inputs.reshape(T * n_batch, *inputs.shape[2:])
        grad_in_, grad_w_ = map(
            lambda x: x.reshape(T * n_batch, C, H, W),
            [grad_in_, grad_w_]
        )

        # Input gradient via transposed convolution
        grad_input_transposed = f.conv_transpose2d(
            grad_in_.to(weight_), weight_, None,
            stride, padding, groups=groups, dilation=dilation
        )

        # Align spatial dimensions if necessary
        if grad_input_transposed.shape != inputs.shape:
            _, C_in, H_in, W_in = inputs.shape
            _, _, H_grad, W_grad = grad_input_transposed.shape

            if H_grad > H_in:
                grad_input_transposed = grad_input_transposed[:, :, :H_in, :]
            elif H_grad < H_in:
                grad_input_transposed = f.pad(
                    grad_input_transposed, (0, 0, 0, H_in - H_grad)
                )

            if W_grad > W_in:
                grad_input_transposed = grad_input_transposed[:, :, :, :W_in]
            elif W_grad < W_in:
                grad_input_transposed = f.pad(
                    grad_input_transposed, (0, W_in - W_grad, 0, 0)
                )

        # Gate gradient with input spikes
        grad_input = grad_input_transposed * inputs

        # Weight gradient via unfold
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        inputs_unfolded = f.unfold(
            inputs, (kernel_h, kernel_w),
            stride=stride, padding=padding, dilation=dilation
        )

        grad_weight = torch.zeros_like(weight)

        for c_out in range(out_channels):
            grad_w_c = grad_w_[:, c_out, :, :]
            grad_w_c = grad_w_c.contiguous().view(T * n_batch, 1, -1)
            grad_w_expanded = grad_w_c.expand(-1, in_channels * kernel_h * kernel_w, -1)
            grad_weight_c = (inputs_unfolded * grad_w_expanded).sum(dim=(0, 2))
            grad_weight[c_out] = grad_weight_c.view(in_channels, kernel_h, kernel_w)

        # Batch normalization backward
        grad_weight, grad_bn_w, grad_bn_b = bn_backward(
            grad_weight, weight, norm_weight, norm_bias, mean, var
        )

        # Scale input gradient as in the original implementation
        return (
            grad_input.reshape(T, n_batch, *inputs.shape[1:]) * 0.85,
            grad_weight,
            grad_bn_w,
            grad_bn_b,
            None, None, None
        )
