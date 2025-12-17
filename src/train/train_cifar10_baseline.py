"""
CIFAR10数据集的激活驱动SNN基础训练（任务1核心）
使用默认ATan替代梯度函数，验证基础网络的可行性
"""
import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler  # 混合精度训练
from spikingjelly.activation_based.surrogate import ATan  # 默认替代梯度

# 添加项目根目录到 Python 路径，确保可以导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入自定义模块
from src.models.model_cifar10_csnn import CIFAR10CSNN
from src.data.data_cifar10 import load_cifar10
from src.utils.utils import get_device, init_tensorboard, save_checkpoint, calculate_metrics

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CIFAR10基础SNN训练（激活驱动）')
    # 设备参数
    parser.add_argument('--device', default='cuda:0', type=str, help='设备ID（如cuda:0、cpu）')
    # 训练超参数
    parser.add_argument('--T', default=4, type=int, help='时间步长（脉冲序列长度）')
    parser.add_argument('--epochs', default=64, type=int, help='训练轮次')
    parser.add_argument('--batch_size', default=128, type=int, help='批量大小')
    parser.add_argument('--lr', default=0.1, type=float, help='初始学习率')
    parser.add_argument('--channels', default=32, type=int, help='第一层卷积输出通道数')
    parser.add_argument('--min_lr', default=1e-4, type=float, help='最小学习率（余弦退火的下界）')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='梯度裁剪阈值（0表示不裁剪）')
    parser.add_argument('--patience', default=10, type=int, help='早停耐心值（连续多少个epoch无提升则停止，0表示不早停）')
    # 数据与日志
    parser.add_argument('--data_dir', default='./data/CIFAR10', type=str, help='CIFAR10数据集目录')
    parser.add_argument('--log_dir', default='./logs/cifar10_baseline', type=str, help='日志保存目录')
    parser.add_argument('--resume', default='', type=str, help='恢复训练的checkpoint路径（可选）')
    return parser.parse_args()

def train_one_epoch(
    net: CIFAR10CSNN,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter
) -> tuple[float, float]:
    """训练一个epoch"""
    net.train()  # 训练模式
    total_train_loss = 0.0
    total_train_acc = 0.0
    total_samples = 0.0
    start_time = time.time()

    for batch_idx, (img, label) in enumerate(train_loader):
        # 数据移至设备
        img, label = img.to(device), label.to(device)
        batch_size = img.shape[0]
        total_samples += batch_size

        # 清零梯度
        optimizer.zero_grad()
        # 混合精度训练（加速训练，减少显存占用）
        with torch.cuda.amp.autocast():
            # 前向传播：计算平均发放率
            out_fr = net(img)
            # 激活驱动损失：MSE损失（最小化发放率与目标one-hot向量的差距）
            label_onehot = F.one_hot(label, num_classes=10).float()  # 转为one-hot
            loss = F.mse_loss(out_fr, label_onehot)

        # 反向传播与优化
        scaler.scale(loss).backward()  # 梯度缩放
        # 梯度裁剪（防止梯度爆炸，提高训练稳定性）
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)  # 取消缩放以进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)  # 优化器更新
        scaler.update()  # 更新缩放器

        # 计算当前批量的准确率和平均损失
        batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
        total_train_acc += batch_acc * batch_size
        total_train_loss += batch_avg_loss * batch_size

        # 重置神经元状态（多步训练后避免状态累积）
        net.reset()

        # 打印批量信息（每10个批量打印一次）
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Batch Loss: {batch_avg_loss:.4f} | Batch Acc: {batch_acc:.4f}")

    # 计算当前epoch的平均损失和准确率
    avg_train_loss = total_train_loss / total_samples
    avg_train_acc = total_train_acc / total_samples
    train_time = time.time() - start_time

    # 记录TensorBoard日志
    writer.add_scalar('Train/Avg_Loss', avg_train_loss, epoch)
    writer.add_scalar('Train/Avg_Acc', avg_train_acc, epoch)

    # 打印epoch总结
    print(f"\nEpoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {avg_train_acc:.4f} | Time: {train_time:.2f}s")

    return avg_train_loss, avg_train_acc

def test_one_epoch(
    net: CIFAR10CSNN,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter
) -> tuple[float, float]:
    """测试一个epoch（无梯度计算）"""
    net.eval()  # 评估模式
    total_test_loss = 0.0
    total_test_acc = 0.0
    total_samples = 0.0

    with torch.no_grad():  # 禁用梯度计算
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            batch_size = img.shape[0]
            total_samples += batch_size

            # 前向传播
            out_fr = net(img)
            # 计算损失
            label_onehot = F.one_hot(label, num_classes=10).float()
            loss = F.mse_loss(out_fr, label_onehot)

            # 计算批量指标
            batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
            total_test_acc += batch_acc * batch_size
            total_test_loss += batch_avg_loss * batch_size

            # 重置神经元状态
            net.reset()

    # 计算平均指标
    avg_test_loss = total_test_loss / total_samples
    avg_test_acc = total_test_acc / total_samples

    # 记录TensorBoard日志
    writer.add_scalar('Test/Avg_Loss', avg_test_loss, epoch)
    writer.add_scalar('Test/Avg_Acc', avg_test_acc, epoch)

    # 打印测试总结
    print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f}\n")

    return avg_test_loss, avg_test_acc

def main(args: argparse.Namespace):
    # 1. 初始化设备
    device = get_device(args.device)

    # 2. 加载数据
    print("\n=== 加载CIFAR10数据集 ===")
    train_loader, test_loader = load_cifar10(
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )

    # 3. 初始化模型（默认使用ATan替代梯度）
    print("\n=== 初始化模型 ===")
    surrogate_func = ATan()  # 默认替代梯度
    net = CIFAR10CSNN(
        T=args.T,
        channels=args.channels,
        surrogate_func=surrogate_func
    ).to(device)
    # 打印模型基本信息（避免使用 repr 导致递归错误）
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型配置：T={args.T}, channels={args.channels}, surrogate=ATan")
    print(f"模型参数：总参数={total_params:,}, 可训练参数={trainable_params:,}")

    # 4. 初始化优化器和学习率调度
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,  # 动量加速收敛
        weight_decay=5e-4  # L2正则化，防止过拟合
    )
    # 余弦退火学习率调度（随epoch降低学习率，但设置最小学习率避免过小）
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # 5. 初始化混合精度训练缩放器
    scaler = GradScaler()

    # 6. 恢复训练（若指定checkpoint）
    start_epoch = 0
    max_test_acc = 0.0
    if args.resume != '':
        from src.utils.utils import load_checkpoint
        checkpoint = load_checkpoint(
            net=net,
            optimizer=optimizer,
            checkpoint_path=args.resume,
            device=device
        )
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    # 7. 初始化TensorBoard日志
    writer = init_tensorboard(log_dir=args.log_dir)

    # 8. 开始训练循环
    print("\n=== 开始训练 ===")
    # 早停机制变量
    best_epoch = start_epoch
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            net=net,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer
        )

        # 测试一个epoch
        test_loss, test_acc = test_one_epoch(
            net=net,
            test_loader=test_loader,
            device=device,
            epoch=epoch,
            writer=writer
        )

        # 更新学习率
        lr_scheduler.step()
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # 保存checkpoint（判断是否为最优模型）
        is_best = test_acc > max_test_acc
        if is_best:
            max_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1
        
        save_checkpoint(
            net=net,
            optimizer=optimizer,
            epoch=epoch,
            max_test_acc=max_test_acc,
            save_path=os.path.join(args.log_dir, 'checkpoints'),
            is_best=is_best
        )
        
        # 早停检查
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n早停触发：连续 {args.patience} 个epoch测试准确率无提升")
            print(f"最优模型在 epoch {best_epoch+1}，测试准确率：{max_test_acc:.4f}")
            break

    # 训练结束
    print(f"\n=== 训练结束 ===")
    print(f"最大测试准确率：{max_test_acc:.4f}")
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
