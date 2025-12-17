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
from utils_custom import glv, preprocess_inputs
from layers.linear_custom import LinearLayer
from layers.pooling_custom import PoolLayer
from layers.losses_custom import SpikeLoss

# 注意：这里需要实现卷积层才能完整训练CIFAR
# 当前版本仅作为框架，实际使用时需要添加卷积层实现


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
    
    print("Note: This script requires convolutional layer implementation.")
    print("Please implement ConvLayer in layers/conv_custom.py to use this script.")
    print("For now, this is a framework that can be extended.")
    
    # 创建日志目录
    log_dir = f"{network_config['log_path']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(log_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(log_dir, 'config.yaml'))
    
    # 创建损失函数
    err = SpikeLoss().to(device)
    
    # 创建优化器
    optim_type = network_config['optimizer']
    lr = network_config['lr']
    weight_decay = network_config['weight_decay']
    
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD([], lr=lr, weight_decay=weight_decay)  # 空参数列表，因为网络未创建
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam([], lr=lr, weight_decay=weight_decay)
    elif optim_type == 'AdamW':
        optimizer = torch.optim.AdamW([], lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_type}")
    
    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=network_config['epochs'])
    
    writer = SummaryWriter(log_dir)
    best_acc = 0
    best_epoch = 0
    training_results = []
    
    # 注意：以下训练循环需要网络实现才能运行
    # 这里仅作为框架，保持与 train_mnist.py 的一致性
    print(f"Log directory: {log_dir}")
    print("CIFAR training framework created. Implement ConvLayer to complete.")
    print("Training loop structure is ready for loss recording and JSON saving.")
    
    # 示例：如果网络已创建，训练循环应该如下：
    # for epoch in range(1, network_config['epochs'] + 1):
    #     train_acc, train_loss = train_epoch(network, trainloader, optimizer, epoch, network_config, device, err)
    #     test_acc, test_loss, confusion = test_epoch(network, testloader, epoch, network_config, device, err)
    #     
    #     scheduler.step()
    #     
    #     print(f'Epoch [{epoch}/{network_config["epochs"]}]: '
    #           f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, '
    #           f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    #     
    #     # 记录到tensorboard
    #     writer.add_scalar('Accuracy/Train', train_acc, epoch)
    #     writer.add_scalar('Accuracy/Test', test_acc, epoch)
    #     writer.add_scalar('Loss/Train', train_loss, epoch)
    #     writer.add_scalar('Loss/Test', test_loss, epoch)
    #     
    #     # 保存结果到列表
    #     training_results.append({
    #         'epoch': epoch,
    #         'train_loss': float(train_loss),
    #         'train_acc': float(train_acc),
    #         'test_loss': float(test_loss),
    #         'test_acc': float(test_acc)
    #     })
    #     
    #     if test_acc > best_acc:
    #         best_acc = test_acc
    #         best_epoch = epoch
    #     
    #     # 保存混淆矩阵
    #     np.save(os.path.join(log_dir, f'confusion_epoch_{epoch}.npy'), confusion)
    # 
    # # 保存训练结果到JSON文件
    # results_dict = {
    #     'epochs': training_results,
    #     'best_epoch': best_epoch,
    #     'best_test_acc': float(best_acc)
    # }
    # with open(os.path.join(log_dir, 'training_results.json'), 'w') as f:
    #     json.dump(results_dict, f, indent=2)
    # 
    # writer.close()


if __name__ == '__main__':
    main()
