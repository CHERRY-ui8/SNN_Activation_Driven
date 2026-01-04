"""
Adjusted Average Pooling layer implementation.

This module follows a modified average pooling strategy inspired by prior work,
with a custom backward pass designed to preserve gradient consistency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


def read_kernel_config(kernel_size):
    """Parse and normalize kernel size configuration."""
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    elif isinstance(kernel_size, str):
        # Expected format: "(2, 2)"
        assert kernel_size[0] == '(' and kernel_size[-1] == ')'
        data = kernel_size[1:-1]
        x, y = map(int, data.split(','))
        return (x, y)
    else:
        return kernel_size


class AdjustedAvgPoolFunc(torch.autograd.Function):
    """
    Custom average pooling function.

    The forward pass uses standard average pooling.
    The backward pass redistributes gradients using an adjusted rule to
    maintain gradient invariance.
    """
    
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, kernel):
        """
        Forward pass using standard average pooling.

        Args:
            inputs: Tensor of shape (T * batch_size, C, H, W)
            kernel: Tuple (kernel_h, kernel_w)

        Returns:
            outputs: Tensor of shape (T * batch_size, C, H_out, W_out)
        """
        outputs = F.avg_pool2d(inputs, kernel)
        # Store tensors required for backward computation
        ctx.save_for_backward(
            outputs,
            torch.tensor(inputs.shape, dtype=torch.long),
            torch.tensor(kernel)
        )
        return outputs
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_delta):
        """
        Backward pass with adjusted gradient redistribution.

        The main idea is to rescale gradients so that their total contribution
        remains stable after pooling (gradient invariance).
        """
        outputs, input_shape, kernel = ctx.saved_tensors
        kernel = kernel.tolist()
        input_shape = input_shape.tolist()
        
        # Compute adjustment factor based on pooled outputs
        adjustment = 1.0 / (outputs + 1e-8)  # Prevent division by zero
        
        # Clamp excessively large adjustment values
        kernel_area = kernel[0] * kernel[1]
        adjustment[adjustment > kernel_area + 1] = 0
        
        # Normalize by kernel area
        adjustment = adjustment / kernel_area
        
        # Apply adjustment and upsample back to input resolution
        grad_adjusted = grad_delta * adjustment
        grad_input = F.interpolate(
            grad_adjusted,
            size=input_shape[2:],
            mode='nearest'
        )
        
        return grad_input, None


class PoolLayer(nn.Module):
    """Pooling layer wrapper supporting multiple pooling strategies."""
    
    def __init__(self, network_config, config, name):
        """
        Initialize the pooling layer.

        Args:
            network_config: Global network configuration dictionary.
            config: Layer-specific configuration, e.g. {'kernel_size': int or tuple}.
            name: Layer name.
        """
        super(PoolLayer, self).__init__()
        self.name = name
        self.type = 'pooling'
        self.network_config = network_config
        
        kernel_size = config['kernel_size']
        self.kernel = read_kernel_config(kernel_size)
        
        print(f"Pooling layer: {name}")
        print(f"  Kernel size: {self.kernel}")
        print("-----------------------------------------")
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (T, batch_size, C, H, W).

        Returns:
            y: Output tensor of shape (T, batch_size, C, H_out, W_out).
        """
        pool_type = self.network_config.get('pooling_type', 'adjusted_avg')
        
        T, n_batch, C, H, W = x.shape
        # Flatten temporal and batch dimensions for pooling
        x = x.reshape(T * n_batch, C, H, W)
        
        if pool_type == 'avg':
            # Standard average pooling
            x = F.avg_pool2d(x, self.kernel)
        elif pool_type == 'max':
            # Max pooling
            x = F.max_pool2d(x, self.kernel)
        elif pool_type == 'adjusted_avg':
            # Average pooling with custom backward behavior
            x = AdjustedAvgPoolFunc.apply(x, self.kernel)
        else:
            raise ValueError(f"Unknown pooling type: {pool_type}")
        
        # Restore original (T, batch) dimensions
        x = x.reshape(T, n_batch, *x.shape[1:])
        return x
