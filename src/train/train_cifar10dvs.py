"""
CIFAR10DVS神经形态数据集的SNN训练（任务3核心）
使用任务2选出的最优替代梯度，对比两种数据预处理方式（按事件数/按时间切分）
"""
import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import GradScaler

# 添加项目根目录到 Python 路径，确保可以导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from src.models.model_cifar10dvs_csnn import CIFAR10DVSCSNN
from src.data.data_cifar10dvs import load_cifar10dvs
from src.surrogate.surrogate_custom import SuperSpikeSurrogate  # 假设SuperSpike是最优（可替换）
from src.utils.utils import get_device, init_tensorboard, save_checkpoint, calculate_metrics

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR10DVS SNN训练与预处理对比')
    # 设备与训练参数
    parser.add_argument('--device', default='cuda:0', type=str, help='设备ID')
    parser.add_argument('--frame_num', default=16, type=int, help='DVS切分的帧数（时间步长T）')
    parser.add_argument('--epochs', default=32, type=int, help='训练轮次')
    parser.add_argument('--batch_size', default=32, type=int, help='批量大小（DVS数据较大，建议 smaller）')
    parser.add_argument('--lr', default=0.05, type=float, help='初始学习率')
    parser.add_argument('--channels', default=16, type=int, help='第一层卷积通道数')
    # 数据与预处理
    parser.add_argument('--data_dir', default='./data/CIFAR10DVS', type=str, help='CIFAR10DVS目录')
    parser.add_argument('--split_modes', default='number,time', type=str, help='预处理方式（逗号分隔：number/time）')
    # 日志与结果
    parser.add_argument('--log_dir', default='./logs/cifar10dvs_train', type=str, help='日志目录')
    parser.add_argument('--save_result', default='./results/cifar10dvs_preprocess_compare.csv', type=str, help='预处理对比结果CSV')
    return parser.parse_args()

def train_dvs_split_mode(
    split_mode: str,
    args: argparse.Namespace,
    device: torch.device
) -> dict:
    """
    单个预处理方式（split_mode）的CIFAR10DVS训练
    Args:
        split_mode: 预处理方式（'number'：按事件数切分，'time'：按时间切分）
        args: 命令行参数
        device: 设备
    Returns:
        result: 训练结果字典
    """
    print(f"\n=== 开始训练预处理方式：{split_mode}（帧数T={args.frame_num}）===")
    # 1. 加载DVS数据（按当前预处理方式）
    train_loader, test_loader, T = load_cifar10dvs(
        frame_num=args.frame_num,
        batch_size=args.batch_size,
        split_by=split_mode,
        data_dir=args.data_dir
    )

    # 2. 初始化模型（使用最优替代梯度，此处假设SuperSpike）
    surrogate_func = SuperSpikeSurrogate(beta=2.0)
    net = CIFAR10DVSCSNN(
        T=T,
        channels=args.channels,
        surrogate_func=surrogate_func
    ).to(device)
    # 打印模型基本信息（避免使用 repr 导致递归错误）
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型配置：T={T}, channels={args.channels}, surrogate=SuperSpike")
    print(f"模型参数：总参数={total_params:,}, 可训练参数={trainable_params:,}")

    # 3. 初始化优化器和调度器
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
    tb_dir = os.path.join(args.log_dir, f'split_{split_mode}')
    writer = init_tensorboard(log_dir=tb_dir)

    # 6. 训练循环变量
    max_test_acc = 0.0
    total_train_time = 0.0

    # 7. 训练循环
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        # 训练一个epoch
        net.train()
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_samples = 0.0

        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            batch_size = img.shape[0]
            total_samples += batch_size

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out_fr = net(img)  # img.shape=[N,T,2,32,32]
                label_onehot = F.one_hot(label, 10).float()
                loss = F.mse_loss(out_fr, label_onehot)

            scaler.scale(loss).backward()
            
            # 梯度裁剪：防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
            total_train_acc += batch_acc * batch_size
            total_train_loss += batch_avg_loss * batch_size
            net.reset()

        # 计算训练指标
        avg_train_loss = total_train_loss / total_samples
        avg_train_acc = total_train_acc / total_samples
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time

        # 测试一个epoch
        net.eval()
        total_test_loss = 0.0
        total_test_acc = 0.0
        total_test_samples = 0.0

        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                batch_size = img.shape[0]
                total_test_samples += batch_size

                out_fr = net(img)
                label_onehot = F.one_hot(label, 10).float()
                loss = F.mse_loss(out_fr, label_onehot)

                batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
                total_test_acc += batch_acc * batch_size
                total_test_loss += batch_avg_loss * batch_size
                net.reset()

        avg_test_loss = total_test_loss / total_test_samples
        avg_test_acc = total_test_acc / total_test_samples

        # 更新学习率
        lr_scheduler.step()
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        # 记录日志
        writer.add_scalar('Train/Avg_Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Avg_Acc', avg_train_acc, epoch)
        writer.add_scalar('Test/Avg_Loss', avg_test_loss, epoch)
        writer.add_scalar('Test/Avg_Acc', avg_test_acc, epoch)

        # 更新最大准确率并保存模型
        if avg_test_acc > max_test_acc:
            max_test_acc = avg_test_acc
            save_checkpoint(
                net=net,
                optimizer=optimizer,
                epoch=epoch,
                max_test_acc=max_test_acc,
                save_path=os.path.join(tb_dir, 'checkpoints'),
                is_best=True
            )

        # 打印epoch总结
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] | Split: {split_mode} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

    # 训练结束，关闭日志
    writer.close()

    # 整理结果
    result = {
        'Split_Mode': split_mode,
        'Frame_Number': args.frame_num,
        'Max_Test_Accuracy': max_test_acc,
        'Total_Train_Time': total_train_time,
        'Final_Train_Loss': avg_train_loss,
        'Final_Test_Loss': avg_test_loss
    }
    print(f"\n=== {split_mode} 训练结果 ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    return result

def main(args: argparse.Namespace):
    # 1. 初始化设备
    device = get_device(args.device)

    # 2. 解析预处理方式列表（如'number,time'→['number', 'time']）
    split_modes = args.split_modes.split(',')
    split_modes = [mode.strip() for mode in split_modes if mode.strip() in ['number', 'time']]
    if not split_modes:
        raise ValueError("请指定有效的预处理方式：number 或 time")
    print(f"\n=== 待对比的预处理方式：{split_modes} ===")

    # 3. 逐个训练不同预处理方式并记录结果
    all_results = []
    for mode in split_modes:
        result = train_dvs_split_mode(
            split_mode=mode,
            args=args,
            device=device
        )
        all_results.append(result)

    # 4. 保存预处理对比结果为CSV
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.save_result, index=False, encoding='utf-8')
    print(f"\n=== 预处理对比结果已保存到：{args.save_result} ===")
    print("\n预处理对比汇总：")
    print(result_df[['Split_Mode', 'Frame_Number', 'Max_Test_Accuracy', 'Total_Train_Time']].to_string(index=False))

    # 5. 输出最优预处理方式
    best_result = max(all_results, key=lambda x: x['Max_Test_Accuracy'])
    print(f"\n=== 最优预处理方式 ===")
    print(f"方式：{best_result['Split_Mode']}（按{'事件数' if best_result['Split_Mode']=='number' else '时间'}切分）")
    print(f"最大测试准确率：{best_result['Max_Test_Accuracy']:.4f}")
    print(f"总训练时间：{best_result['Total_Train_Time']:.2f}s")

if __name__ == '__main__':
    args = parse_args()
    main(args)
