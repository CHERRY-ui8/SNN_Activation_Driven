# 事件驱动反向传播实现说明

本目录包含基于对论文"Training Spiking Neural Networks with Event-driven Backpropagation"理解而重新实现的代码。

## 文件结构

```
event-driven/
├── layers/
│   ├── __init__.py
│   ├── neuron_custom.py          # 神经元前向/反向传播实现
│   ├── linear_custom.py          # 线性层实现
│   └── pooling_custom.py          # 改进的池化层实现
├── networks/
│   ├── mnist.yaml                # MNIST配置
│   ├── cifar10.yaml              # CIFAR-10配置（需要卷积层实现）
│   └── cifar100.yaml             # CIFAR-100配置（需要卷积层实现）
├── train_mnist.py                # MNIST训练脚本
├── train_cifar.py                # CIFAR训练脚本框架
├── visualize_gradient_flow.py     # 梯度流可视化脚本
├── utils_custom.py                # 工具函数（全局变量、数据预处理、损失函数）
└── README.md                      # 本文件
```

## 核心算法实现

### 1. 事件驱动反向传播原理

事件驱动反向传播的核心思想是：**只在神经元发放脉冲（spike）时才计算和传播梯度**。

#### 前向传播

在 `neuron_custom.py` 中实现的神经元前向传播：

- **膜电位更新**：使用leaky积分器模型
  - `syn_m = (syn_m + in_I[t]) * (1 - theta_m)`
  - `syn_s = (syn_s + in_I[t]) * (1 - theta_s)`
  - `u = (syn_m - syn_s) * theta_s / (theta_s - theta_m)`

- **脉冲发放**：当膜电位超过阈值时发放脉冲
  - `out = (u >= threshold)`
  - 发放脉冲后重置：`u = 0, syn_m = 0, syn_s = 0`

- **关键变量**：
  - `delta_u[t]`：膜电位增量，用于前向传播
  - `delta_u_t[t]`：时间增量，用于反向传播的梯度计算

#### 反向传播

在 `neuron_custom.py` 中实现的神经元反向传播：

- **事件驱动**：从最后一个时间步向前遍历，只在已发放过脉冲的神经元上计算梯度

- **梯度计算**：
  ```python
  partial_u = clamp(-1 / delta_u[t], -4, 0)
  partial_u_t = clamp(-1 / delta_u_t[t], -max_dudt_inv, 0)
  grad_w[t] = partial_u_grad_w * syn_a[delta_t] * spiked
  grad_in[t] = partial_u_grad_t * partial_a[delta_t] * spiked
  ```

- **关键变量**：
  - `syn_a[delta_t]`：突触响应函数，用于权重梯度
  - `partial_a[delta_t]`：偏导数函数，用于输入梯度
  - `delta_t`：距离上次脉冲的时间间隔

### 2. 梯度不变性

论文证明了梯度总和在时间维度上保持不变。这通过以下方式实现：

- `syn_a` 和 `partial_a` 的计算确保梯度在时间维度上的总和不变
- 改进的池化层（adjusted average pooling）确保池化操作不会破坏梯度不变性

### 3. syn_a 和 partial_a 的计算

在 `utils_custom.py` 的 `GlobalVars.init()` 中计算：

- **syn_a[t]**：突触响应函数
  ```
  syn_a[t] = ((1-theta_m)^(t+1) - (1-theta_s)^(t+1)) * theta_s / (theta_s - theta_m)
  ```

- **delta_syn_a[t]**：根据梯度类型计算
  - 指数类型：`delta_syn_a[t] = (1-theta_grad)^(t+1)`
  - 原始类型：使用差分计算

- **partial_a**：`partial_a = -delta_syn_a`

### 4. 线性层实现

在 `linear_custom.py` 中实现：

- **前向传播**：
  1. 批归一化：`weight_ = (weight - mean) / sqrt(var + eps) * norm_weight + norm_bias`
  2. 线性变换：`in_I = inputs @ weight_.T`
  3. 神经元前向传播：调用 `neuron_forward`
  4. 监督信号注入（输出层）：在膜电位增加时注入脉冲

- **反向传播**：
  1. 梯度过滤：`grad_delta *= outputs`（只在脉冲时刻有梯度）
  2. 神经元反向传播：调用 `neuron_backward`
  3. 权重梯度：`grad_weight = sum_t(grad_w[t].T @ inputs[t])`
  4. 输入梯度：`grad_input = grad_in @ weight_ * inputs`
  5. 批归一化反向传播

### 5. 改进的池化层

在 `pooling_custom.py` 中实现 adjusted average pooling：

- **原理**：标准平均池化在反向传播时会导致梯度消失，改进方法调整梯度分配以保持梯度不变性

- **前向**：使用标准平均池化

- **反向**：使用改进的梯度分配策略
  ```python
  adjustment = 1.0 / (outputs + eps)
  adjustment[adjustment > kernel_area + 1] = 0
  adjustment = adjustment / kernel_area
  grad_input = interpolate(grad_delta * adjustment, size=input_shape)
  ```

## 使用方法

### 训练MNIST

```bash
cd event-driven
python train_mnist.py -config networks/mnist.yaml -seed 42
```

### 可视化梯度流

```bash
python visualize_gradient_flow.py -config networks/mnist.yaml -checkpoint path/to/checkpoint.pth
```

### CIFAR-10/100训练

```bash
# CIFAR-10
python train_cifar.py -config networks/cifar10.yaml -seed 42

# CIFAR-100
python train_cifar.py -config networks/cifar100.yaml -seed 42
```

注意：当前CIFAR训练脚本需要实现卷积层才能完整使用。框架已创建，可以在此基础上扩展。

## 与论文的对应关系

1. **事件驱动反向传播**：对应论文第3节的核心算法
   - 前向传播：论文公式(1)-(4)
   - 反向传播：论文公式(5)-(8)

2. **梯度不变性**：对应论文第4节的理论分析
   - syn_a和partial_a的计算确保梯度总和不变

3. **改进的池化层**：对应论文第5节的改进方法
   - adjusted average pooling解决梯度消失问题

## 实验结果

训练MNIST应该能够达到合理的准确率（通常>90%）。具体结果取决于超参数设置和训练轮数。

## 注意事项

1. 确保CUDA可用（如果使用GPU）
2. 数据路径需要正确设置
3. 超参数（tau_m, tau_s, tau_grad等）对训练效果有重要影响
4. CIFAR训练需要实现卷积层才能完整使用

## 扩展方向

1. 实现卷积层（`layers/conv_custom.py`）以支持CIFAR训练
2. 添加更多损失函数（如TET loss）
3. 实现dropout层
4. 优化性能（如使用CUDA后端）

## 参考文献

- Training Spiking Neural Networks with Event-driven Backpropagation (NeurIPS 2022)
