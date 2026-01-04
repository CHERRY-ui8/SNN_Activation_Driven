"""
Linear layer implementation with neuron dynamics and batch normalization.

This version is reimplemented based on an understanding of event-driven
backpropagation, while preserving the original computational behavior.
"""
import torch
import torch.nn as nn
import sys
import os

# Import utility modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_custom import glv
from layers.neuron_custom import neuron_forward, neuron_backward
from torch.cuda.amp import custom_fwd, custom_bwd


def bn_forward(inputs, weight, norm_weight, norm_bias):
    """
    Batch normalization forward pass.

    Normalizes the weight parameters rather than activations.
    """
    C = weight.shape[0]  # number of output channels
    
    # Compute per-channel mean and variance
    mean = torch.mean(weight.reshape(C, -1), dim=1)
    var = torch.std(weight.reshape(C, -1), dim=1) ** 2
    
    # Reshape statistics to match weight dimensions
    if len(weight.shape) == 4:  # convolutional weights
        shape = (-1, 1, 1, 1)
    else:  # linear weights
        shape = (-1, 1)
    
    mean = mean.reshape(*shape)
    var = var.reshape(*shape)
    norm_weight = norm_weight.reshape(*shape)
    norm_bias = norm_bias.reshape(*shape)
    
    # Normalize weights:
    # weight_ = (weight - mean) / sqrt(var + eps) * norm_weight + norm_bias
    weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias
    
    return inputs, mean, var, weight_


def bn_backward(grad_weight, weight, norm_weight, norm_bias, mean, var):
    """
    Batch normalization backward pass.
    """
    C = weight.shape[0]
    std_inv = 1.0 / torch.sqrt(var + 1e-5)
    
    # Match parameter shape
    if len(weight.shape) == 4:
        shape = (-1, 1, 1, 1)
    else:
        shape = (-1, 1)
    
    # Reconstruct normalized weights
    weight_ = (weight - mean) * std_inv * norm_weight.reshape(*shape) + norm_bias.reshape(*shape)
    
    # Gradients for batch normalization parameters
    grad_bn_b = torch.sum(grad_weight.reshape(C, -1), dim=1).reshape(norm_bias.shape)
    grad_bn_w = torch.sum((grad_weight * weight_).reshape(C, -1), dim=1).reshape(norm_weight.shape)
    
    # Gradient w.r.t. original weights
    grad_weight = grad_weight * norm_weight.reshape(*shape)
    m = weight.numel() // C
    
    grad_var = grad_weight * (weight - mean) / m * (-0.5) * std_inv ** 3
    grad_mean = -grad_weight * std_inv
    grad_weight = (
        grad_weight * std_inv
        + grad_var * 2 * (weight - mean) / m
        + grad_mean / m
    )
    
    return grad_weight, grad_bn_w, grad_bn_b


class LinearFunc(torch.autograd.Function):
    """Custom autograd function for the linear layer."""
    
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, weight, norm_weight, norm_bias, neuron_config, labels):
        """
        Forward pass.

        Args:
            inputs: Tensor of shape (T, batch_size, n_inputs),
                    representing input spike trains.
            weight: Tensor of shape (n_outputs, n_inputs).
            norm_weight: Batch norm scale parameter, shape (n_outputs, 1).
            norm_bias: Batch norm bias parameter, shape (n_outputs, 1).
            neuron_config: Tuple (theta_m, theta_s, theta_grad, threshold)
                            defining neuron dynamics.
            labels: Tensor of shape (batch_size,), used only for the output layer.

        Returns:
            outputs: Tensor of shape (T, batch_size, n_outputs),
                     representing output spike trains.
        """
        # Apply batch normalization to weights
        inputs, mean, var, weight_ = bn_forward(inputs, weight, norm_weight, norm_bias)
        
        # Linear transformation
        # in_I = inputs @ weight_.T
        in_I = torch.matmul(inputs, weight_.t())  # (T, batch_size, n_outputs)
        
        # Neuron forward dynamics
        delta_u, delta_u_t, outputs = neuron_forward(in_I, neuron_config)
        
        # Inject supervision signal for the output layer
        if labels is not None:
            T, n_batch, N = in_I.shape
            glv.outputs_raw = outputs.clone()
            
            i2 = torch.arange(n_batch, device=outputs.device)
            # Identify time steps where membrane potential increases
            is_inc = (delta_u[:, i2, labels] > 0.05).float()
            # Select the first increasing time step
            _, i1 = torch.max(
                is_inc
                * torch.arange(1, T + 1, device=is_inc.device).unsqueeze(-1),
                dim=0
            )
            # Inject a spike at that time step
            outputs[i1, i2, labels] = (delta_u[i1, i2, labels] != 0).to(outputs.dtype)
        
        # Save intermediate results for backward pass
        ctx.save_for_backward(
            delta_u, delta_u_t, inputs, outputs,
            weight, norm_weight, norm_bias, mean, var
        )
        ctx.is_out_layer = labels is not None
        
        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        """
        Backward pass.

        Args:
            grad_delta: Tensor of shape (T, batch_size, n_outputs),
                        gradient from the loss function.

        Returns:
            grad_input: Gradient w.r.t. inputs.
            grad_weight: Gradient w.r.t. weights.
            grad_bn_w: Gradient w.r.t. batch norm scale.
            grad_bn_b: Gradient w.r.t. batch norm bias.
        """
        (
            delta_u, delta_u_t, inputs, outputs,
            weight, norm_weight, norm_bias, mean, var
        ) = ctx.saved_tensors
        
        # Event-driven rule: gradients only flow at spike times
        grad_delta = grad_delta * outputs
        
        # Neuron backward dynamics
        grad_in_, grad_w_ = neuron_backward(
            grad_delta, outputs, delta_u, delta_u_t
        )
        
        # Normalized weights (for input gradient computation)
        weight_ = (weight - mean) / torch.sqrt(var + 1e-5) * norm_weight + norm_bias
        
        # Input gradient
        # inputs act as a gate for event-driven propagation
        grad_input = torch.matmul(grad_in_, weight_) * inputs
        
        # Weight gradient: sum over time
        grad_weight = torch.sum(
            torch.matmul(grad_w_.transpose(1, 2), inputs),
            dim=0
        )
        
        # Batch normalization backward
        grad_weight, grad_bn_w, grad_bn_b = bn_backward(
            grad_weight, weight, norm_weight, norm_bias, mean, var
        )
        
        # Scale input gradient (as in the original implementation)
        return grad_input * 0.85, grad_weight, grad_bn_w, grad_bn_b, None, None


class LinearLayer(nn.Module):
    """Linear layer module with neuron dynamics."""
    
    def __init__(self, network_config, config, name):
        """
        Initialize the linear layer.

        Args:
            network_config: Global network configuration.
            config: Layer configuration dictionary, e.g.
                    {'n_inputs': int, 'n_outputs': int, 'threshold': float}.
            name: Layer name.
        """
        super(LinearLayer, self).__init__()
        
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.threshold = config.get('threshold', 1.0)
        self.name = name
        self.type = 'linear'
        
        # Weight matrix (no bias term)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        # Batch normalization parameters
        self.norm_weight = nn.Parameter(torch.ones(out_features, 1))
        self.norm_bias = nn.Parameter(torch.zeros(out_features, 1))
        
        print(f"Linear layer: {name}")
        print(f"  Input features: {in_features}")
        print(f"  Output features: {out_features}")
        print(f"  Weight shape: {list(self.weight.shape)}")
        print("-----------------------------------------")
    
    def forward(self, x, labels=None):
        """
        Forward pass.

        Args:
            x: Tensor of shape (T, batch_size, n_inputs) or
               (T, batch_size, C, H, W).
            labels: Tensor of shape (batch_size,), used only for the output layer.
        """
        # Flatten spatial dimensions if input is 5D
        if len(x.shape) == 5:
            T, n_batch, C, H, W = x.shape
            x = x.view(T, n_batch, C * H * W)
        
        # Initialization phase: full forward pass without labels
        if glv.init_flag:
            config_n = glv.network_config
            theta_m = 1.0 / config_n['tau_m']
            theta_s = 1.0 / config_n['tau_s']
            theta_grad = (
                1.0 / config_n['tau_grad']
                if config_n['gradient_type'] == 'exponential'
                else -123456789
            )
            neuron_config = (theta_m, theta_s, theta_grad, self.threshold)
            return LinearFunc.apply(
                x, self.weight, self.norm_weight, self.norm_bias,
                neuron_config, None
            )
        
        # Neuron configuration
        config_n = glv.network_config
        theta_m = 1.0 / config_n['tau_m']
        theta_s = 1.0 / config_n['tau_s']
        theta_grad = (
            1.0 / config_n['tau_grad']
            if config_n['gradient_type'] == 'exponential'
            else -123456789
        )
        neuron_config = (theta_m, theta_s, theta_grad, self.threshold)
        
        # Call custom autograd function
        return LinearFunc.apply(
            x, self.weight, self.norm_weight, self.norm_bias,
            neuron_config, labels
        )
    
    def weight_clipper(self):
        """Clamp weights to a fixed range."""
        self.weight.data = self.weight.data.clamp(-4, 4)
