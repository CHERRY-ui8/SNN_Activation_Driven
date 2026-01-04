"""
CIFAR10DVS SNN Training
"""
import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import GradScaler

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.model_cifar10dvs_csnn import CIFAR10DVSCSNN
from src.data.data_cifar10dvs import load_cifar10dvs
from src.surrogate.surrogate_custom import EsserSurrogate
from src.utils.utils import get_device, init_tensorboard, save_checkpoint, calculate_metrics

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CIFAR10DVS SNN Training')
    parser.add_argument('--device', default='cuda:0', type=str, help='device ID')
    parser.add_argument('--frame_num', default=16, type=int, help='number of frames')
    parser.add_argument('--epochs', default=64, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping')
    parser.add_argument('--channels', default=16, type=int, help='first conv channels')
    parser.add_argument('--data_dir', default='./datasets/CIFAR10DVS', type=str, help='dataset directory')
    parser.add_argument('--split_modes', default='number,time', type=str, help='split modes')
    parser.add_argument('--log_dir', default='./logs/cifar10dvs_train', type=str, help='log directory')
    parser.add_argument('--save_result', default='./results/cifar10dvs_preprocess_compare.csv', type=str, help='result CSV')
    return parser.parse_args()

def train_dvs_split_mode(
    split_mode: str,
    args: argparse.Namespace,
    device: torch.device
) -> dict:
    print(f"\n=== Training split mode: {split_mode} (T={args.frame_num}) ===")
    train_loader, test_loader, T = load_cifar10dvs(
        frame_num=args.frame_num,
        batch_size=args.batch_size,
        split_by=split_mode,
        data_dir=args.data_dir
    )

    surrogate_func = EsserSurrogate(beta=1)
    net = CIFAR10DVSCSNN(
        T=T,
        channels=args.channels,
        surrogate_func=surrogate_func
    ).to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model config: T={T}, channels={args.channels}, surrogate=Esser")
    print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-3)

    scaler = GradScaler()

    tb_dir = os.path.join(args.log_dir, f'split_{split_mode}')
    writer = init_tensorboard(log_dir=tb_dir)

    max_test_acc = 0.0
    total_train_time = 0.0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

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
                loss = F.cross_entropy(out_fr, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
            total_train_acc += batch_acc * batch_size
            total_train_loss += batch_avg_loss * batch_size
            net.reset()

        avg_train_loss = total_train_loss / total_samples
        avg_train_acc = total_train_acc / total_samples
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
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
                loss = F.cross_entropy(out_fr, label)

                batch_acc, batch_avg_loss = calculate_metrics(out_fr, label, loss)
                total_test_acc += batch_acc * batch_size
                total_test_loss += batch_avg_loss * batch_size
                net.reset()

        avg_test_loss = total_test_loss / total_test_samples
        avg_test_acc = total_test_acc / total_test_samples

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)

        writer.add_scalar('Train/Avg_Loss', avg_train_loss, epoch)
        writer.add_scalar('Train/Avg_Acc', avg_train_acc, epoch)
        writer.add_scalar('Test/Avg_Loss', avg_test_loss, epoch)
        writer.add_scalar('Test/Avg_Acc', avg_test_acc, epoch)

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

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] | Split: {split_mode} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

    writer.close()

    result = {
        'Split_Mode': split_mode,
        'Frame_Number': args.frame_num,
        'Max_Test_Accuracy': max_test_acc,
        'Total_Train_Time': total_train_time,
        'Final_Train_Loss': avg_train_loss,
        'Final_Test_Loss': avg_test_loss
    }
    print(f"\n=== {split_mode} results ===")
    for k, v in result.items():
        print(f"{k}: {v}")
    return result

def main(args: argparse.Namespace):
    device = get_device(args.device)

    split_modes = args.split_modes.split(',')
    split_modes = [mode.strip() for mode in split_modes if mode.strip() in ['number', 'time']]
    if not split_modes:
        raise ValueError("Please specify valid split modes: number or time")
    print(f"\n=== Split modes to compare: {split_modes} ===")

    all_results = []
    for mode in split_modes:
        result = train_dvs_split_mode(
            split_mode=mode,
            args=args,
            device=device
        )
        all_results.append(result)

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.save_result, index=False, encoding='utf-8')
    print(f"\n=== Results saved to: {args.save_result} ===")
    print("\nComparison summary:")
    print(result_df[['Split_Mode', 'Frame_Number', 'Max_Test_Accuracy', 'Total_Train_Time']].to_string(index=False))

    best_result = max(all_results, key=lambda x: x['Max_Test_Accuracy'])
    print(f"\n=== Best split mode ===")
    print(f"Mode: {best_result['Split_Mode']}")
    print(f"Max test accuracy: {best_result['Max_Test_Accuracy']:.4f}")
    print(f"Total training time: {best_result['Total_Train_Time']:.2f}s")

if __name__ == '__main__':
    args = parse_args()
    main(args)
