"""
Neuron forward and backward propagation implementation
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_custom import glv


@torch.jit.script
def neuron_forward_custom(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp):
    u_last = torch.zeros_like(in_I[0])
    syn_m = torch.zeros_like(in_I[0])
    syn_s = torch.zeros_like(in_I[0])
    syn_grad = torch.zeros_like(in_I[0])
    
    T = in_I.shape[0]
    delta_u = torch.zeros_like(in_I)
    delta_u_t = torch.zeros_like(in_I)
    outputs = torch.zeros_like(in_I)
    
    for t in range(T):
        syn_m = (syn_m + in_I[t]) * (1 - theta_m)
        syn_s = (syn_s + in_I[t]) * (1 - theta_s)
        syn_grad = (syn_grad + in_I[t]) * (1 - theta_grad)
        
        if not is_forward_leaky:
            delta_u_t[t] = syn_grad
            u = u_last + delta_u_t[t]
            delta_u[t] = delta_u_t[t]
        else:
            u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)
            delta_u[t] = u - u_last
            delta_u_t[t] = syn_grad if is_grad_exp else delta_u[t]
        
        out = (u >= threshold).to(u.dtype)
        
        u_last = u * (1 - out)
        syn_m = syn_m * (1 - out)
        syn_s = syn_s * (1 - out)
        syn_grad = syn_grad * (1 - out)
        
        outputs[t] = out
    
    return delta_u, delta_u_t, outputs


@torch.jit.script
def neuron_backward_custom(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv):
    T = grad_delta.shape[0]
    
    grad_in_ = torch.zeros_like(outputs)
    grad_w_ = torch.zeros_like(outputs)
    
    partial_u_grad_w = torch.zeros_like(outputs[0])
    partial_u_grad_t = torch.zeros_like(outputs[0])
    delta_t = torch.zeros(outputs.shape[1:], device=outputs.device, dtype=torch.long)
    spiked = torch.zeros_like(outputs[0])
    
    for t in range(T - 1, -1, -1):
        out = outputs[t]
        
        spiked = spiked + (1 - spiked) * out
        
        partial_u = torch.clamp(-1.0 / delta_u[t], -4.0, 0.0)
        partial_u_t = torch.clamp(-1.0 / delta_u_t[t], -max_dudt_inv, 0.0)
        
        partial_u_grad_w = partial_u_grad_w * (1 - out) + grad_delta[t] * partial_u * out
        partial_u_grad_t = partial_u_grad_t * (1 - out) + grad_delta[t] * partial_u_t * out
        
        delta_t = (delta_t + 1) * (1 - out).long()
        
        grad_in_[t] = partial_u_grad_t * partial_a[delta_t] * spiked.to(partial_a.dtype)
        grad_w_[t] = partial_u_grad_w * syn_a[delta_t] * spiked.to(syn_a.dtype)
    
    return grad_in_, grad_w_


def neuron_forward(in_I, neuron_config):
    theta_m, theta_s, theta_grad, threshold = neuron_config
    theta_m = torch.tensor(theta_m, device=in_I.device, dtype=in_I.dtype)
    theta_s = torch.tensor(theta_s, device=in_I.device, dtype=in_I.dtype)
    theta_grad = torch.tensor(theta_grad, device=in_I.device, dtype=in_I.dtype)
    threshold = torch.tensor(threshold, device=in_I.device, dtype=in_I.dtype)
    
    assert theta_m != theta_s, "theta_m and theta_s must be different"
    
    is_grad_exp = torch.tensor(glv.network_config['gradient_type'] == 'exponential', device=in_I.device)
    is_forward_leaky = torch.tensor(glv.network_config['forward_type'] == 'leaky', device=in_I.device)
    
    return neuron_forward_custom(in_I, theta_m, theta_s, theta_grad, threshold, is_forward_leaky, is_grad_exp)


def neuron_backward(grad_delta, outputs, delta_u, delta_u_t):
    syn_a = glv.syn_a.to(outputs.device)
    partial_a = -glv.delta_syn_a.to(outputs.device)  # partial_a = -delta_syn_a
    max_dudt_inv = torch.tensor(glv.network_config['max_dudt_inv'], device=outputs.device, dtype=outputs.dtype)
    
    return neuron_backward_custom(grad_delta, outputs, delta_u, delta_u_t, syn_a, partial_a, max_dudt_inv)
