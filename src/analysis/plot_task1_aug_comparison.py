"""
Task1 数据增强前后对比可视化脚本
绘制有/无数据增强的训练曲线对比（accuracy和loss）
完全遵循 plot_task1_baseline 的视觉风格
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def get_tb_data(log_dir, tag_name):
    """从单个 TensorBoard 目录中提取特定 tag 的数据"""
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    
    if 'scalars' not in ea.Tags() or tag_name not in ea.Tags()['scalars']:
        print(f"Warning: Tag '{tag_name}' not found in {log_dir}")
        return None
        
    events = ea.Scalars(tag_name)
    return pd.DataFrame([(e.step, e.value) for e in events], columns=['Epoch', 'Value'])

def find_aug_comparison_experiments(logs_dir='./logs', T=20, weight_decay=0.0001):
    """
    找到数据增强对比实验的日志目录
    查找所有符合 T=20, weight_decay=0.0001 的实验
    返回找到的所有实验路径列表
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise ValueError(f"Logs directory not found: {logs_dir}")
    
    # Task1特征：Sigmoid_beta4.0, T=20, weight_decay=0.0001
    experiments = []  # 存储所有找到的实验路径
    
    for root, dirs, files in os.walk(logs_path):
        # 检查是否有TensorBoard日志
        if any(f.startswith('events.out.tfevents') for f in files):
            parent_dir = str(Path(root).parent)
            dir_name = Path(parent_dir).name
            
            # 检查是否是task1实验：Sigmoid_beta4.0, 不是SigmoidPrime
            if 'Sigmoid_beta4.0' in dir_name and 'SigmoidPrime' not in dir_name:
                # 检查T值和weight_decay
                if f'_T{T}_' in dir_name:
                    # 检查weight_decay（可能是wd0.0001或wd0.0）
                    wd_str = f'wd{weight_decay}' if weight_decay > 0 else 'wd0'
                    if wd_str in dir_name or (weight_decay == 0.0001 and 'wd0.0' in dir_name):
                        experiments.append(root)
    
    # 按路径排序（最新的在前）
    experiments.sort(reverse=True)
    return experiments

def plot_aug_comparison(no_aug_log_dir, with_aug_log_dir, output_path='./results/task1_aug_comparison.png'):
    """
    绘制数据增强前后的对比曲线
    类似 plot_task1_baseline 的风格，但对比"无数据增强"和"有数据增强"
    
    Args:
        no_aug_log_dir: 无数据增强实验的TensorBoard日志目录
        with_aug_log_dir: 有数据增强实验的TensorBoard日志目录
        output_path: 输出图片路径
    """
    # 设置学术风格
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义要查找的 Tag 模式
    tag_patterns = {
        "acc": {"train": "Train/Avg_Acc", "test": "Test/Avg_Acc"},
        "loss": {"train": "Train/Avg_Loss", "test": "Test/Avg_Loss"}
    }
    
    # 为不同的实验分配颜色和标签
    colors = sns.color_palette("tab10", 2)
    experiments = [
        ("No Augmentation", no_aug_log_dir, colors[0]),
        ("With Augmentation", with_aug_log_dir, colors[1])
    ]
    
    # 绘制每个实验的曲线
    for label, log_dir, color in experiments:
        if log_dir is None:
            print(f"Warning: {label} log directory is None, skipping...")
            continue
            
        # --- 绘制 Accuracy 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["acc"][mode]
            df = get_tb_data(log_dir, tag)
            if df is not None:
                # 平滑处理 (alpha越小越平滑)
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax1.plot(df['Epoch'], smooth_val, 
                         label=f"{label} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)
        
        # --- 绘制 Loss 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["loss"][mode]
            df = get_tb_data(log_dir, tag)
            if df is not None:
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax2.plot(df['Epoch'], smooth_val, 
                         label=f"{label} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)
    
    # 细节修饰
    ax1.set_title("Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=9, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9, loc='upper right', ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Task1 Data Augmentation Comparison (T=20, wd=0.0001)", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    """主函数：找到数据增强对比实验并生成图表"""
    parser = argparse.ArgumentParser(description='Task1 数据增强前后对比可视化')
    parser.add_argument('--logs_dir', default='./logs', type=str, 
                       help='日志目录路径')
    parser.add_argument('--no_aug_dir', type=str, default=None,
                       help='无数据增强实验的TensorBoard日志目录（如果未指定，将自动查找）')
    parser.add_argument('--with_aug_dir', type=str, default=None,
                       help='有数据增强实验的TensorBoard日志目录（如果未指定，将自动查找）')
    parser.add_argument('--T', default=20, type=int,
                       help='时间步长T（默认20）')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                       help='权重衰减（默认0.0001）')
    parser.add_argument('--output', default='./results/task1_aug_comparison.png', type=str,
                       help='输出图片路径')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Task1 数据增强前后对比可视化")
    print("=" * 80)
    
    # 查找符合条件的实验
    print(f"\n查找 T={args.T}, weight_decay={args.weight_decay} 的实验...")
    experiments = find_aug_comparison_experiments(
        logs_dir=args.logs_dir, 
        T=args.T, 
        weight_decay=args.weight_decay
    )
    
    if not experiments:
        print(f"\n未找到符合条件的实验（T={args.T}, weight_decay={args.weight_decay}）")
        print("请检查logs目录或使用 --no_aug_dir 和 --with_aug_dir 手动指定实验路径")
        return
    
    print(f"\n找到 {len(experiments)} 个符合条件的实验：")
    for i, exp_dir in enumerate(experiments, 1):
        print(f"  {i}. {exp_dir}")
    
    # 确定要使用的实验路径
    no_aug_dir = args.no_aug_dir
    with_aug_dir = args.with_aug_dir
    
    if no_aug_dir is None or with_aug_dir is None:
        if len(experiments) >= 2:
            # 如果找到2个或更多实验，使用最新的两个
            # 较旧的是无数据增强，较新的是有数据增强
            print(f"\n自动选择实验进行对比：")
            print(f"  无数据增强: {experiments[-1]} (较早的实验)")
            print(f"  有数据增强: {experiments[0]} (最新的实验)")
            
            if no_aug_dir is None:
                no_aug_dir = experiments[-1]  # 较旧的是无数据增强
            if with_aug_dir is None:
                with_aug_dir = experiments[0]  # 较新的是有数据增强
        elif len(experiments) == 1:
            # 如果只有一个实验，尝试在同一个目录下找时间戳子目录
            base_dir = Path(experiments[0]).parent
            timestamp_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('20')], reverse=True)
            if len(timestamp_dirs) >= 2:
                print(f"\n在同一目录下找到多个时间戳实验：")
                print(f"  无数据增强: {timestamp_dirs[-1]} (较早的实验)")
                print(f"  有数据增强: {timestamp_dirs[0]} (最新的实验)")
                if no_aug_dir is None:
                    no_aug_dir = str(timestamp_dirs[-1])
                if with_aug_dir is None:
                    with_aug_dir = str(timestamp_dirs[0])
            else:
                print(f"\n只找到一个实验，需要手动指定另一个实验路径")
                print("请使用 --no_aug_dir 和 --with_aug_dir 参数")
                return
        elif len(experiments) == 1:
            print(f"\n只找到一个实验，需要手动指定另一个实验路径")
            print("请使用 --no_aug_dir 和 --with_aug_dir 参数")
            return
        else:
            print(f"\n需要手动指定实验路径")
            print("请使用 --no_aug_dir 和 --with_aug_dir 参数")
            return
    
    # 验证路径是否存在
    if not Path(no_aug_dir).exists():
        print(f"错误：无数据增强实验路径不存在: {no_aug_dir}")
        return
    if not Path(with_aug_dir).exists():
        print(f"错误：有数据增强实验路径不存在: {with_aug_dir}")
        return
    
    print(f"\n使用实验路径：")
    print(f"  无数据增强: {no_aug_dir}")
    print(f"  有数据增强: {with_aug_dir}")
    
    # 生成图表
    print("\n生成对比曲线图...")
    plot_aug_comparison(no_aug_dir, with_aug_dir, args.output)
    
    print("\n" + "=" * 80)
    print("图表生成完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

