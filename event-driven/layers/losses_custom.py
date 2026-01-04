"""
Loss function implementation.

This module is adapted from the original `loss_count` implementation,
with the same logic preserved to ensure correctness.
"""
import torch
from torch.cuda.amp import custom_fwd, custom_bwd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_custom import glv


class LossCount(torch.autograd.Function):
    """
    Spike count–based loss function.

    This implementation is kept consistent with the original version.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, output, target):
        """
        Args:
            output: Tensor of shape (T, batch_size, n_classes),
                    representing spike trains over time.
            target: Tensor of shape (batch_size, n_classes),
                    representing desired spike counts.

        Returns:
            delta: Tensor of shape (T, batch_size, n_classes),
                   used for loss computation.
        """
        desired_count = glv.network_config['desired_count']
        undesired_count = glv.network_config['undesired_count']
        T = output.shape[0]

        # Total spike count over the temporal dimension
        out_count = torch.sum(output, dim=0)  # (batch_size, n_classes)
        
        # Difference between actual and target spike counts
        delta = (out_count - target) / T
        
        # Masking rule:
        # Only penalize cases where
        # 1) the target class expects more spikes but receives fewer, or
        # 2) non-target classes expect fewer spikes but receive more.
        delta[
            (target == desired_count) & (delta > 0) |
            (target == undesired_count) & (delta < 0)
        ] = 0
        
        # Expand back to the temporal dimension
        delta = delta.unsqueeze(0).repeat(T, 1, 1)
        return delta
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        """
        Backward pass.

        The gradient sign is optionally flipped depending on the
        `loss_reverse` configuration flag.
        """
        sign = -1 if glv.network_config['loss_reverse'] else 1
        return sign * grad, None


class SpikeLoss(torch.nn.Module):
    """Wrapper module for spike-based loss computation."""
    
    def __init__(self):
        super(SpikeLoss, self).__init__()
    
    def spike_count(self, output, target):
        """
        Spike count loss.

        Args:
            output: Tensor of shape (T, batch_size, n_classes).
            target: Tensor of shape (batch_size, n_classes).

        Returns:
            Scalar loss value.
        """
        delta = LossCount.apply(output, target)
        return 0.5 * torch.sum(delta ** 2)
