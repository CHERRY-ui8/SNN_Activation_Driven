"""
MNIST训练脚本
使用自定义实现的事件驱动反向传播算法
"""
import os
import sys
import yaml
import shutil
import argparse
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import confusion_matrix

# 导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_custom import glv, preprocess_inputs, initialize_layer
from layers.linear_custom import LinearLayer
from layers.losses_custom import SpikeLoss


class SimpleMLP(nn.Module):
    """简单的MLP网络"""
    
    def __init__(self, layers_config):
        super(SimpleMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        for key, config in layers_config.items():
            if config['type'] == 'linear':
                layer = LinearLayer(glv.network_config, config, key)
                self.layers.append(layer)
        
        print("Network structure created")
        print("-----------------------------------------")
    
    def forward(self, x, labels=None):
        """前向传播"""
        # 如果输入是图像格式 (T, batch_size, 1, 28, 28)，需要展平
        if len(x.shape) == 5:
            T, batch_size, C, H, W = x.shape
            x = x.view(T, batch_size, C * H * W)
        
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                # 最后一层（输出层）需要labels
                x = layer(x, labels)
            else:
                # 隐藏层不需要labels
                x = layer(x, None)
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
    # 早期脉冲权重更大
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


def test_epoch(network, testloader, epoch, network_config, device, err):
    """测试一个epoch"""
    network.eval()
    correct = 0
    total = 0
    test_loss = 0
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
            
            # 计算损失
            loss = get_loss(err, outputs, labels, network_config)
            test_loss += loss.item()
            
            # 统计
            spike_counts = readout(outputs, T)
            predicted = torch.argmax(spike_counts, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_pred.append(predicted.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    
    acc = 100.0 * correct / total
    avg_loss = test_loss / len(testloader)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    
    # 计算混淆矩阵
    nums = np.bincount(y_true)
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(n_class)) / nums.reshape(-1, 1)
    
    return acc, avg_loss, confusion


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
    glv.rank = 0 if torch.cuda.is_available() else -1
    glv.init(network_config, layers_config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    data_path = network_config['data_path']
    os.makedirs(data_path, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = torchvision.datasets.MNIST(
        data_path, train=True, transform=transform, download=True
    )
    testset = torchvision.datasets.MNIST(
        data_path, train=False, transform=transform, download=True
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=network_config['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=network_config['batch_size'], 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 创建网络
    network = SimpleMLP(layers_config).to(device)
    
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
    with torch.no_grad():
        x = init_inputs
        for i, layer in enumerate(network.layers):
            if isinstance(layer, LinearLayer):
                x = initialize_layer(layer, x)
    glv.init_flag = False
    network.train()
    print("Weight initialization completed.")
    
    # 创建损失函数
    err = SpikeLoss().to(device)
    
    # 创建优化器
    optim_type = network_config['optimizer']
    lr = network_config['lr']
    weight_decay = network_config['weight_decay']
    
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
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
    best_epoch = 0
    training_results = []
    
    # 训练循环
    print("Starting training...")
    for epoch in range(1, network_config['epochs'] + 1):
        train_acc, train_loss = train_epoch(network, trainloader, optimizer, epoch, network_config, device, err)
        test_acc, test_loss, confusion = test_epoch(network, testloader, epoch, network_config, device, err)
        
        scheduler.step()
        
        print(f'Epoch [{epoch}/{network_config["epochs"]}]: '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
              f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # 记录到tensorboard
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        
        # 保存结果到列表
        training_results.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'test_loss': float(test_loss),
            'test_acc': float(test_acc)
        })
        
        # 保存模型
        state = {
            'net': network.state_dict(),
            'epoch': epoch,
            'test_acc': test_acc,
        }
        torch.save(state, os.path.join(log_dir, 'last.pth'))
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(state, os.path.join(log_dir, 'best.pth'))
            print(f'New best accuracy: {best_acc:.2f}%')
        
        # 保存混淆矩阵
        np.save(os.path.join(log_dir, f'confusion_epoch_{epoch}.npy'), confusion)
    
    # 保存训练结果到JSON文件
    results_dict = {
        'epochs': training_results,
        'best_epoch': best_epoch,
        'best_test_acc': float(best_acc)
    }
    with open(os.path.join(log_dir, 'training_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f'Training completed! Best test accuracy: {best_acc:.2f}% at epoch {best_epoch}')
    writer.close()


if __name__ == '__main__':
    main()
