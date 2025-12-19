"""
CIFAR-10/100训练脚本
使用自定义实现的事件驱动反向传播算法和改进的池化层
"""
import os
import sys
import yaml
import shutil
import argparse
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix

# 导入自定义模块
my_impl_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_impl_dir)
from utils_custom import glv, preprocess_inputs, initialize_layer
from layers.linear_custom import LinearLayer
from layers.pooling_custom import PoolLayer
from layers.losses_custom import SpikeLoss


def _load_original_layers(project_root, my_impl_dir):
    """加载原始实现的卷积层和dropout层，避免与自定义layers冲突"""
    import importlib.util
    import types
    
    # 调整路径优先级：项目根目录优先于 my_implementation
    for path in [my_impl_dir, project_root]:
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, project_root)
    
    # 清理已缓存的 layers 模块
    for mod_name in [k for k in sys.modules.keys() if k.startswith('layers')]:
        del sys.modules[mod_name]
    sys.modules['layers'] = types.ModuleType('layers')
    
    # 加载依赖
    import global_v as glv_original
    
    # 辅助函数：加载并注册模块
    def _load_module(name, file_path):
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    
    # 按依赖顺序加载
    layers_dir = os.path.join(project_root, 'layers')
    _load_module('layers.functions', os.path.join(layers_dir, 'functions.py'))
    conv = _load_module('layers.conv', os.path.join(layers_dir, 'conv.py'))
    dropout = _load_module('layers.dropout', os.path.join(layers_dir, 'dropout.py'))
    
    sys.path.insert(1, my_impl_dir)  # 恢复 my_implementation 到路径
    return conv, dropout, glv_original


conv, dropout, glv_original = _load_original_layers(project_root, my_impl_dir)

class CIFARNetwork(nn.Module):
    """支持卷积层的CIFAR网络"""
    def __init__(self, layers_config):
        super(CIFARNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        for key, config in layers_config.items():
            if config['type'] == 'conv':
                layer = conv.ConvLayer(glv.network_config, config, key)
                self.layers.append(layer)
            elif config['type'] == 'linear':
                layer = LinearLayer(glv.network_config, config, key)
                self.layers.append(layer)
            elif config['type'] == 'pooling':
                layer = PoolLayer(glv.network_config, config, key)
                self.layers.append(layer)
            elif config['type'] == 'dropout':
                layer = dropout.DropoutLayer(config, key)
                self.layers.append(layer)
            else:
                raise ValueError(f"Unknown layer type: {config['type']}")
        
        print("Network structure created")
        print("-----------------------------------------")
    
    def forward(self, x, labels=None):
        """前向传播"""
        # x shape: (T, batch_size, C, H, W) for conv layers
        for i, layer in enumerate(self.layers):
            # 检查是否是dropout层
            if hasattr(layer, 'type') and layer.type == 'dropout':
                # 原始DropoutLayer会根据glv.init_flag和training状态自动处理
                x = layer(x)
            elif i == len(self.layers) - 1:
                # 最后一层（输出层）需要labels
                # 如果是线性层，需要展平
                if hasattr(layer, 'type') and layer.type == 'linear':
                    # 展平卷积输出: (T, batch_size, C, H, W) -> (T, batch_size, C*H*W)
                    if len(x.shape) == 5:
                        T, batch_size, C, H, W = x.shape
                        x = x.view(T, batch_size, C * H * W)
                x = layer(x, labels)
            else:
                # 隐藏层：conv, pooling, linear（非最后一层）
                # 如果是线性层且输入是5D，需要先展平
                if hasattr(layer, 'type') and layer.type == 'linear':
                    if len(x.shape) == 5:
                        T, batch_size, C, H, W = x.shape
                        flattened_size = C * H * W
                        x = x.view(T, batch_size, flattened_size)
                # 调用层的forward方法
                # 原始卷积层和池化层不需要labels参数
                x = layer(x)
        return x


def get_loss(err, outputs, labels, network_config):
    """计算损失函数（与原始实现保持一致）"""
    desired_count = network_config['desired_count']
    undesired_count = network_config['undesired_count']
    
    # 创建目标脉冲数
    targets = torch.ones_like(outputs[0]) * undesired_count
    for i in range(len(labels)):
        targets[i, labels[i]] = desired_count
    
    # 使用SpikeLoss计算损失
    loss = err.spike_count(outputs, targets)
    
    return loss


def readout(outputs, T):
    """读取输出：对时间维度加权求和"""
    weights = 1.1 - torch.arange(T, device=outputs.device).reshape(T, 1, 1) / T / 10
    weighted_outputs = outputs * weights
    return torch.sum(weighted_outputs, dim=0)


def train_epoch(network, trainloader, optimizer, epoch, network_config, device, err):
    """训练一个epoch"""
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    T = network_config['n_steps']
    
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 数据预处理
        inputs = preprocess_inputs(inputs, network_config, T)
        
        # 前向传播
        outputs = network(inputs, labels)
        
        # 计算损失
        loss = get_loss(err, outputs, labels, network_config)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        spike_counts = readout(outputs, T)
        predicted = torch.argmax(spike_counts, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            acc = 100.0 * correct / total
            print(f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(trainloader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
    
    acc = 100.0 * correct / total
    avg_loss = train_loss / len(trainloader)
    return acc, avg_loss


def test_epoch(network, testloader, epoch, network_config, device):
    """测试一个epoch"""
    network.eval()
    correct = 0
    total = 0
    T = network_config['n_steps']
    n_class = network_config['n_class']
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 数据预处理
            inputs = preprocess_inputs(inputs, network_config, T)
            
            # 前向传播
            outputs = network(inputs, None)
            
            # 统计
            spike_counts = readout(outputs, T)
            predicted = torch.argmax(spike_counts, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_pred.append(predicted.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    
    acc = 100.0 * correct / total
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    
    # 计算混淆矩阵
    nums = np.bincount(y_true)
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(n_class)) / nums.reshape(-1, 1)
    
    return acc, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='Path to config file')
    parser.add_argument('-seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    # 读取配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    network_config = config['Network']
    layers_config = config['Layers']
    
    # 初始化全局变量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # rank用于指定GPU设备ID，0表示第一个GPU，-1表示CPU（但原始实现可能不支持）
    glv.rank = 0 if torch.cuda.is_available() else -1
    glv.init(network_config, layers_config)
    
    # 初始化原始全局变量
    glv_original.rank = 0 if torch.cuda.is_available() else 0
    glv_original.init(network_config, layers_config)
    print(f'Using device: {device}')
    
    # 加载数据
    data_path = network_config['data_path']
    os.makedirs(data_path, exist_ok=True)
    
    # CIFAR-10数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if network_config['dataset'] == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            data_path, train=True, transform=transform_train, download=True
        )
        testset = torchvision.datasets.CIFAR10(
            data_path, train=False, transform=transform_test, download=True
        )
    elif network_config['dataset'] == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(
            data_path, train=True, transform=transform_train, download=True
        )
        testset = torchvision.datasets.CIFAR100(
            data_path, train=False, transform=transform_test, download=True
        )
    else:
        raise ValueError(f"Unknown dataset: {network_config['dataset']}")
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=network_config['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=network_config['batch_size'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 创建网络
    network = CIFARNetwork(layers_config).to(device)
    
    # 权重初始化（关键步骤！）
    print("Initializing weights...")
    batch_size = network_config['batch_size']
    T = network_config['n_steps']
    # 获取一个批次的数据用于初始化
    init_inputs, _ = next(iter(trainloader))
    init_inputs = init_inputs[:batch_size].to(device)
    init_inputs = preprocess_inputs(init_inputs, network_config, T)
    
    # 初始化每一层
    network.eval()
    glv.init_flag = True
    glv_original.init_flag = True
    with torch.no_grad():
        x = init_inputs
        for i, layer in enumerate(network.layers):
            if hasattr(layer, 'type'):
                if layer.type == 'linear':
                    # 如果是线性层，需要先展平
                    if len(x.shape) == 5:
                        T, batch_size, C, H, W = x.shape
                        flattened_size = C * H * W
                        x = x.view(T, batch_size, flattened_size)
                    x = initialize_layer(layer, x)
                elif layer.type == 'conv':
                    # 原始卷积层的初始化在forward中处理
                    x = layer(x)
                elif layer.type == 'pooling':
                    # 池化层直接前向传播
                    x = layer(x)
                elif layer.type == 'dropout':
                    # Dropout层在初始化时跳过
                    continue
            elif isinstance(layer, (nn.Dropout3d, nn.Dropout)):
                # Dropout层不需要初始化
                continue
    glv.init_flag = False
    glv_original.init_flag = False
    network.train()
    print("Weight initialization completed.")
    
    # 创建损失函数
    err = SpikeLoss().to(device)
    
    # 创建优化器
    optim_type = network_config['optimizer']
    lr = network_config['lr']
    weight_decay = network_config['weight_decay']
    
    # 收集需要优化的参数（原始实现中，卷积层和线性层有不同的参数组）
    norm_param, param = [], []
    for layer in network.modules():
        if hasattr(layer, 'type'):
            if layer.type in ['conv', 'linear']:
                if hasattr(layer, 'norm_weight') and hasattr(layer, 'norm_bias'):
                    norm_param.extend([layer.norm_weight, layer.norm_bias])
                if hasattr(layer, 'weight'):
                    param.append(layer.weight)
    
    norm_grad = network_config.get('norm_grad', 1)
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': param},
            {'params': norm_param, 'lr': lr * norm_grad}
        ], lr=lr, weight_decay=weight_decay)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': param},
            {'params': norm_param, 'lr': lr * norm_grad}
        ], lr=lr, weight_decay=weight_decay)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW([
            {'params': param},
            {'params': norm_param, 'lr': lr * norm_grad}
        ], lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_type}")
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=network_config['epochs'])
    
    # 创建日志目录
    log_dir = f"{network_config['log_path']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(log_dir, 'config.yaml'))
    
    writer = SummaryWriter(log_dir)
    best_acc = 0
    
    # 训练循环
    print("Starting training...")
    for epoch in range(1, network_config['epochs'] + 1):
        train_acc, train_loss = train_epoch(network, trainloader, optimizer, epoch, network_config, device, err)
        test_acc, confusion = test_epoch(network, testloader, epoch, network_config, device)
        
        scheduler.step()
        
        print(f'Epoch [{epoch}/{network_config["epochs"]}]: '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Loss: {train_loss:.4f}')
        
        # 记录到tensorboard
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        
        # 保存模型
        state = {
            'net': network.state_dict(),
            'epoch': epoch,
            'test_acc': test_acc,
        }
        torch.save(state, os.path.join(log_dir, 'last.pth'))
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(state, os.path.join(log_dir, 'best.pth'))
            print(f'New best accuracy: {best_acc:.2f}%')
        
        # 保存混淆矩阵
        np.save(os.path.join(log_dir, f'confusion_epoch_{epoch}.npy'), confusion)
    
    print(f'Training completed! Best test accuracy: {best_acc:.2f}%')
    writer.close()


if __name__ == '__main__':
    main()
