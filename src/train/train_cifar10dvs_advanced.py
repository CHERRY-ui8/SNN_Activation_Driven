"""
CIFAR10DVS高级预处理方法对比训练脚本
对比不同的预处理方法（baseline, count_norm, time_surface, adaptive_norm）
"""
import os
import sys
import argparse
import torch
import pandas as pd
from torch.cuda.amp import GradScaler
import time

# 添加项目根目录到 Python 路径，确保可以导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from src.data.data_cifar10dvs_advanced import load_cifar10dvs_advanced
from src.models.model_cifar10dvs_csnn import CIFAR10DVSCSNN
from src.surrogate.surrogate_custom import SuperSpikeSurrogate
from src.utils.utils import get_device, init_tensorboard

def train_one_epoch(net, train_loader, optimizer, scaler, device, epoch, writer):
    """训练一个epoch"""
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = torch.nn.MSELoss()
    
    for batch_idx, (img, label) in enumerate(train_loader):
        img = img.to(device)  # [N, T, C, H, W]
        label = label.to(device)
        label_onehot = torch.zeros(label.size(0), 10).to(device)
        label_onehot.scatter_(1, label.view(-1, 1), 1.0)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            fr = net(img)  # [N, 10]
            loss = criterion(fr, label_onehot)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 重置神经元状态（多步训练后需重置，避免状态累积）
        net.reset()
        
        total_loss += loss.item()
        pred = fr.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct / total
    
    writer.add_scalar('Train/Avg_Loss', avg_loss, epoch)
    writer.add_scalar('Train/Avg_Acc', avg_acc, epoch)
    
    return avg_loss, avg_acc

def test_one_epoch(net, test_loader, device, epoch, writer):
    """测试一个epoch"""
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            label_onehot = torch.zeros(label.size(0), 10).to(device)
            label_onehot.scatter_(1, label.view(-1, 1), 1.0)
            
            fr = net(img)
            loss = criterion(fr, label_onehot)
            
            # 重置神经元状态
            net.reset()
            
            total_loss += loss.item()
            pred = fr.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    
    avg_loss = total_loss / len(test_loader)
    avg_acc = correct / total
    
    writer.add_scalar('Test/Avg_Loss', avg_loss, epoch)
    writer.add_scalar('Test/Avg_Acc', avg_acc, epoch)
    
    return avg_loss, avg_acc

def train_preprocess_method(
    preprocess_method: str,
    args: argparse.Namespace,
    device: torch.device
) -> dict:
    """
    训练单个预处理方法
    """
    print(f"\n=== 开始训练预处理方法：{preprocess_method}（帧数T={args.frame_num}）===")
    
    # 1. 加载数据
    train_loader, test_loader, T = load_cifar10dvs_advanced(
        frame_num=args.frame_num,
        batch_size=args.batch_size,
        split_by='time',  # 使用按时间切分（之前实验显示效果更好）
        preprocess_method=preprocess_method,
        data_dir=args.data_dir
    )
    
    # 2. 初始化模型
    surrogate_func = SuperSpikeSurrogate(beta=2.0)
    net = CIFAR10DVSCSNN(
        T=T,
        channels=args.channels,
        surrogate_func=surrogate_func
    ).to(device)
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型配置：T={T}, channels={args.channels}, surrogate=SuperSpike")
    print(f"模型参数：总参数={total_params:,}")
    
    # 3. 初始化优化器
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. 混合精度缩放器
    scaler = GradScaler()
    
    # 5. 初始化日志
    tb_dir = os.path.join(args.log_dir, f'preprocess_{preprocess_method}')
    writer = init_tensorboard(log_dir=tb_dir)
    
    # 6. 训练循环
    max_test_acc = 0.0
    total_train_time = 0.0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            net, train_loader, optimizer, scaler, device, epoch, writer
        )
        
        # 测试
        test_loss, test_acc = test_one_epoch(
            net, test_loader, device, epoch, writer
        )
        
        # 更新学习率
        lr_scheduler.step()
        
        # 记录最佳准确率
        if test_acc > max_test_acc:
            max_test_acc = test_acc
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.4f}, "
                  f"Max Test Acc: {max_test_acc:.4f}")
    
    writer.close()
    
    # 返回结果
    result = {
        'Preprocess_Method': preprocess_method,
        'Frame_Number': args.frame_num,
        'Max_Test_Accuracy': max_test_acc,
        'Total_Train_Time': total_train_time,
        'Final_Train_Loss': train_loss,
        'Final_Test_Loss': test_loss
    }
    
    return result

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CIFAR10DVS高级预处理方法对比训练')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备ID')
    parser.add_argument('--frame_num', type=int, default=16, help='帧数（时间步长T）')
    parser.add_argument('--channels', type=int, default=32, help='第一层卷积输出通道数')
    parser.add_argument('--epochs', type=int, default=32, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--data_dir', type=str, default='./data/CIFAR10DVS', help='数据集目录')
    parser.add_argument('--log_dir', type=str, default='./logs/cifar10dvs_advanced', help='日志目录')
    parser.add_argument('--save_result', type=str, default='./results/cifar10dvs_advanced_preprocess_compare.csv',
                       help='预处理对比结果CSV')
    parser.add_argument('--methods', type=str, default='baseline,count_norm,time_surface,adaptive_norm',
                       help='要对比的预处理方法（逗号分隔）')
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device(args.device)
    
    # 解析预处理方法列表
    preprocess_methods = [m.strip() for m in args.methods.split(',')]
    
    print("=" * 80)
    print("CIFAR10DVS高级预处理方法对比训练")
    print("=" * 80)
    print(f"预处理方法：{preprocess_methods}")
    print(f"训练配置：T={args.frame_num}, epochs={args.epochs}, batch_size={args.batch_size}")
    
    # 逐个训练不同的预处理方法
    all_results = []
    for method in preprocess_methods:
        result = train_preprocess_method(method, args, device)
        all_results.append(result)
        print(f"\n{method} 训练完成：最大测试准确率 = {result['Max_Test_Accuracy']:.4f}")
    
    # 保存结果为CSV
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.save_result, index=False, encoding='utf-8')
    print(f"\n=== 所有预处理方法对比结果已保存到：{args.save_result} ===")
    print("\n对比结果汇总：")
    print(result_df[['Preprocess_Method', 'Max_Test_Accuracy', 'Total_Train_Time']].to_string(index=False))
    
    # 输出最优预处理方法
    best_result = max(all_results, key=lambda x: x['Max_Test_Accuracy'])
    print(f"\n=== 最优预处理方法 ===")
    print(f"方法：{best_result['Preprocess_Method']}")
    print(f"最大测试准确率：{best_result['Max_Test_Accuracy']:.4f}")
    print(f"总训练时间：{best_result['Total_Train_Time']:.2f}s")

if __name__ == '__main__':
    main()

