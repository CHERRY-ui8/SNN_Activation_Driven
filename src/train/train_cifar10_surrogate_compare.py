"""
CIFAR10数据集上的替代梯度函数性能对比（任务2核心）
对比4种替代梯度：ATan（默认）、Sigmoid、Esser、SuperSpike
输出准确率、收敛速度等对比结果
"""
import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import pandas as pd  # 用于保存对比结果表格
from torch.cuda.amp import GradScaler
from spikingjelly.activation_based.surrogate import ATan

# 添加项目根目录到 Python 路径，确保可以导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from src.models.model_cifar10_csnn import CIFAR10CSNN
from src.data.data_cifar10 import load_cifar10
from src.surrogate.surrogate_custom import SigmoidSurrogate, EsserSurrogate, SuperSpikeSurrogate
from src.utils.utils import get_device, init_tensorboard, save_checkpoint, calculate_metrics, load_checkpoint

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR10替代梯度函数性能对比')
    # 设备与训练参数
    parser.add_argument('--device', default='cuda:0', type=str, help='设备ID')
    parser.add_argument('--T', default=4, type=int, help='时间步长')
    parser.add_argument('--epochs', default=64, type=int, help='训练轮次')
    parser.add_argument('--batch_size', default=128, type=int, help='批量大小')
    parser.add_argument('--lr', default=0.1, type=float, help='初始学习率')
    parser.add_argument('--channels', default=32, type=int, help='第一层卷积通道数')
    # 数据与日志
    parser.add_argument('--data_dir', default='./data/CIFAR10', type=str, help='CIFAR10目录')
    parser.add_argument('--log_dir', default='./logs/cifar10_surrogate_compare', type=str, help='日志目录')
    parser.add_argument('--save_result', default='./results/surrogate_compare_result.csv', type=str, help='对比结果保存路径（CSV）')
    return parser.parse_args()

def train_surrogate(
    surrogate_name: str,
    surrogate_func: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader
) -> dict:
    """
    单个替代梯度函数的训练与测试
    Args:
        surrogate_name: 替代梯度名称（用于日志和结果）
        surrogate_func: 替代梯度函数实例
        args: 命令行参数
        device: 设备
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
    Returns:
        result: 训练结果字典（含最大准确率、收敛epoch、训练时间等）
    """
    print(f"\n=== 开始训练替代梯度：{surrogate_name} ===")
    # 1. 初始化模型
    net = CIFAR10CSNN(
        T=args.T,
        channels=args.channels,
        surrogate_func=surrogate_func
    ).to(device)

    # 2. 初始化优化器和调度器
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 3. 混合精度缩放器
    scaler = GradScaler()

    # 4. 初始化日志（每个替代梯度单独一个TensorBoard目录）
    tb_dir = os.path.join(args.log_dir, surrogate_name)
    writer = init_tensorboard(log_dir=tb_dir)

    # 5. 训练循环变量
    start_epoch = 0
    max_test_acc = 0.0
    total_train_time = 0.0
    converge_epoch = -1  # 收敛轮次（首次达到95%最大准确率的epoch）

    # 6. 训练循环
    for epoch in range(start_epoch, args.epochs):
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
                out_fr = net(img)
                label_onehot = F.one_hot(label, 10).float()
                loss = F.mse_loss(out_fr, label_onehot)

            scaler.scale(loss).backward()
            
            # 调试信息：检查梯度
            if epoch == 0 and total_samples == batch_size:  # 只在第一个batch打印
                print(f"\n[DEBUG Training] Loss: {loss.item():.6f}")
                print(f"[DEBUG Training] Output mean: {out_fr.mean().item():.6f}, std: {out_fr.std().item():.6f}")
                
                # 检查模型参数的梯度
                has_grad = False
                grad_norm = 0.0
                param_count = 0
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        has_grad = True
                        grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
                        if param_count <= 3:  # 只打印前3个参数
                            print(f"[DEBUG Training] {name}: grad_norm={param.grad.norm().item():.6f}, param_norm={param.norm().item():.6f}")
                
                grad_norm = grad_norm ** 0.5
                print(f"[DEBUG Training] Has gradient: {has_grad}, Total grad norm: {grad_norm:.6f}, Parameters with grad: {param_count}")
                
                # 检查输出是否有grad_fn
                print(f"[DEBUG Training] out_fr requires_grad: {out_fr.requires_grad}, grad_fn: {out_fr.grad_fn}")
            
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

        # 记录日志
        writer.add_scalar('Train/Avg_Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Avg_Acc', avg_train_acc, epoch)
        writer.add_scalar('Test/Avg_Loss', avg_test_loss, epoch)
        writer.add_scalar('Test/Avg_Acc', avg_test_acc, epoch)
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 更新最大准确率和收敛轮次
        if avg_test_acc > max_test_acc:
            max_test_acc = avg_test_acc
            # 保存最优模型
            save_checkpoint(
                net=net,
                optimizer=optimizer,
                epoch=epoch,
                max_test_acc=max_test_acc,
                save_path=os.path.join(tb_dir, 'checkpoints'),
                is_best=True
            )
            # 记录收敛轮次（首次达到最大准确率的95%）
            if converge_epoch == -1 and max_test_acc > 0.1:  # 避免初始低准确率干扰
                converge_epoch = epoch

        # 打印epoch总结
        print(f"Epoch [{epoch+1}/{args.epochs}] | {surrogate_name} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")

    # 训练结束，关闭日志
    writer.close()

    # 整理结果
    result = {
        'Surrogate_Name': surrogate_name,
        'Max_Test_Accuracy': max_test_acc,
        'Converge_Epoch': converge_epoch,
        'Total_Train_Time': total_train_time,
        'Final_Train_Loss': avg_train_loss,
        'Final_Test_Loss': avg_test_loss
    }
    print(f"\n=== {surrogate_name} 训练结果 ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    return result

def main(args: argparse.Namespace):
    # 1. 初始化设备
    device = get_device(args.device)

    # 2. 加载CIFAR10数据（所有替代梯度共享同一数据加载器）
    print("\n=== 加载CIFAR10数据集 ===")
    train_loader, test_loader = load_cifar10(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )

    # 3. 定义待对比的替代梯度函数
    # NOTE: 
    surrogates = {
        'ATan_Default': ATan(),  # 默认
        'Sigmoid': SigmoidSurrogate(beta=5.0),
        'Esser': EsserSurrogate(beta=1.0),
        'SuperSpike': SuperSpikeSurrogate(beta=2.0)
    }

    # 4. 逐个训练替代梯度并记录结果
    all_results = []
    for name, func in surrogates.items():
        result = train_surrogate(
            surrogate_name=name,
            surrogate_func=func,
            args=args,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader
        )
        all_results.append(result)

    # 5. 保存结果为CSV（用于作业报告表格）
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.save_result, index=False, encoding='utf-8')
    print(f"\n=== 所有替代梯度对比结果已保存到：{args.save_result} ===")
    print("\n对比结果汇总：")
    print(result_df[['Surrogate_Name', 'Max_Test_Accuracy', 'Converge_Epoch', 'Total_Train_Time']].to_string(index=False))

    # 6. 输出最优替代梯度
    best_result = max(all_results, key=lambda x: x['Max_Test_Accuracy'])
    print(f"\n=== 最优替代梯度 ===")
    print(f"名称：{best_result['Surrogate_Name']}")
    print(f"最大测试准确率：{best_result['Max_Test_Accuracy']:.4f}")
    print(f"收敛轮次：{best_result['Converge_Epoch']}")
    print(f"总训练时间：{best_result['Total_Train_Time']:.2f}s")

if __name__ == '__main__':
    args = parse_args()
    main(args)
