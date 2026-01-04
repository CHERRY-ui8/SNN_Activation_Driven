"""
Task2 实验结果可视化脚本
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
from collections import defaultdict

def get_tb_data(log_dir, tag_name, max_epoch=32):
    """从单个 TensorBoard 目录中提取特定 tag 的数据
    
    Args:
        log_dir: TensorBoard日志目录
        tag_name: 要提取的tag名称
        max_epoch: 最大epoch数，超过此值的会被截断（默认32）
    """
    # size_guidance 设置为 0 表示加载所有数据点
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    if tag_name not in ea.Tags()['scalars']:
        # 尝试一些常见的变体，比如 'Accuracy/train' vs 'train/accuracy'
        print(f"Warning: Tag '{tag_name}' not found in {log_dir}")
        return None
        
    events = ea.Scalars(tag_name)
    # 只保留前max_epoch个epoch的数据（step从0开始，所以保留step < max_epoch）
    filtered_events = [e for e in events if e.step < max_epoch]
    return pd.DataFrame([(e.step, e.value) for e in filtered_events], columns=['Epoch', 'Value'])

def extract_config_from_path(log_path):
    """从日志路径中提取训练配置"""
    # 获取路径的最后一部分（目录名）
    config_str = Path(log_path).name
    if not config_str:
        return None
    
    config = {}
    
    try:
        # 提取surrogate和beta
        if 'SigmoidPrime_beta' in config_str:
            config['surrogate'] = 'SigmoidPrime'
            beta_part = config_str.split('SigmoidPrime_beta')[1].split('_')[0]
            config['surrogate_beta'] = float(beta_part)
        elif 'Esser_beta' in config_str:
            config['surrogate'] = 'Esser'
            beta_part = config_str.split('Esser_beta')[1].split('_')[0]
            config['surrogate_beta'] = float(beta_part)
        elif 'SuperSpike_beta' in config_str:
            config['surrogate'] = 'SuperSpike'
            beta_part = config_str.split('SuperSpike_beta')[1].split('_')[0]
            config['surrogate_beta'] = float(beta_part)
        else:
            return None  # 不是task2实验
        
        # 提取lr
        if '_lr' in config_str:
            lr_part = config_str.split('_lr')[1].split('_')[0]
            config['lr'] = float(lr_part)
        else:
            return None
        
    except Exception as e:
        print(f"Error parsing config from {config_str}: {e}")
        return None
    
    return config

def find_all_experiments(logs_dir='./logs'):
    """扫描logs目录，找到所有task2实验并组织数据"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise ValueError(f"Logs directory not found: {logs_dir}")
    
    # 组织数据：{surrogate: {beta: {lr: log_path}}}
    experiments = defaultdict(lambda: defaultdict(dict))
    
    # 遍历所有目录
    for root, dirs, files in os.walk(logs_path):
        # 检查是否有TensorBoard日志
        if any(f.startswith('events.out.tfevents') for f in files):
            # 提取配置 - root是包含events文件的目录（可能是时间戳子目录）
            # 需要从父目录提取配置
            parent_dir = str(Path(root).parent)
            config = extract_config_from_path(parent_dir)
            if config:
                surrogate = config['surrogate']
                beta = config['surrogate_beta']
                lr = config['lr']
                
                # 如果该配置已经有数据，选择最新的（时间戳最大的目录）
                if lr not in experiments[surrogate][beta] or root > experiments[surrogate][beta][lr]:
                    experiments[surrogate][beta][lr] = root
    
    return experiments

def find_best_lr_for_each_beta(experiments_dict):
    """找到每个beta值对应的最佳lr（基于test accuracy）"""
    best_configs = {}
    
    for beta in experiments_dict.keys():
        best_acc = -1
        best_lr = None
        best_path = None
        
        for lr in experiments_dict[beta].keys():
            log_path = experiments_dict[beta][lr]
            # 获取test accuracy的最大值
            df = get_tb_data(log_path, "Test/Avg_Acc")
            if df is not None and len(df) > 0:
                max_acc = df['Value'].max()
                if max_acc > best_acc:
                    best_acc = max_acc
                    best_lr = lr
                    best_path = log_path
        
        if best_lr is not None:
            best_configs[beta] = {
                'lr': best_lr,
                'path': best_path,
                'max_acc': best_acc
            }
    
    return best_configs

def plot_surrogate_configs(surrogate, experiments_dict, output_path):
    """
    绘制某个surrogate下每个beta值的最佳配置对比图
    完全遵循 plot_surrogate_comparison 的风格
    """
    # 找到每个beta值的最佳lr
    best_configs = find_best_lr_for_each_beta(experiments_dict)
    
    if not best_configs:
        print(f"Warning: No valid experiments found for {surrogate}")
        return
    
    # 设置学术风格
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义要查找的 Tag 模式
    tag_patterns = {
        "acc": {"train": "Train/Avg_Acc", "test": "Test/Avg_Acc"},
        "loss": {"train": "Train/Avg_Loss", "test": "Test/Avg_Loss"}
    }
    
    # 收集最佳配置用于颜色分配
    all_configs = []
    for beta in sorted(best_configs.keys()):
        config = best_configs[beta]
        all_configs.append((beta, config['lr'], config['path']))
    
    colors = sns.color_palette("tab10", len(all_configs))
    
    for i, (beta, lr, log_path) in enumerate(all_configs):
        color = colors[i]
        config_label = f"beta={beta}, lr={lr:.0e}"
        
        # --- 绘制 Accuracy 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["acc"][mode]
            df = get_tb_data(log_path, tag)
            if df is not None:
                # 平滑处理 (alpha越小越平滑)
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax1.plot(df['Epoch'], smooth_val, 
                         label=f"{config_label} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)
        
        # --- 绘制 Loss 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["loss"][mode]
            df = get_tb_data(log_path, tag)
            if df is not None:
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax2.plot(df['Epoch'], smooth_val, 
                         label=f"{config_label} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)
    
    # 细节修饰
    ax1.set_title("Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 32)  # 确保x轴范围对齐到32个epoch
    ax1.legend(fontsize=9, loc='lower right', ncol=2)
    
    ax2.set_title("Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_xlim(0, 32)  # 确保x轴范围对齐到32个epoch
    ax2.legend(fontsize=9, loc='upper right', ncol=2)
    
    plt.suptitle(f"{surrogate} - Best Configuration for Each Beta", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def find_best_configs(all_experiments):
    """找到每种surrogate的最佳配置（基于test accuracy的最大值）"""
    best_configs = {}
    
    for surrogate in all_experiments.keys():
        best_acc = -1
        best_config = None
        best_path = None
        
        for beta in all_experiments[surrogate].keys():
            for lr in all_experiments[surrogate][beta].keys():
                log_path = all_experiments[surrogate][beta][lr]
                # 获取test accuracy的最大值
                df = get_tb_data(log_path, "Test/Avg_Acc")
                if df is not None and len(df) > 0:
                    max_acc = df['Value'].max()
                    if max_acc > best_acc:
                        best_acc = max_acc
                        best_config = {'surrogate': surrogate, 'beta': beta, 'lr': lr}
                        best_path = log_path
        
        if best_config:
            best_configs[surrogate] = {
                'config': best_config,
                'path': best_path,
                'max_acc': best_acc
            }
    
    return best_configs

def plot_best_configs_comparison(best_configs, output_dir='./results'):
    """
    绘制最优配置的对比图（accuracy和loss在同一张图的2个子图中）
    完全遵循 plot_surrogate_comparison 的风格
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要查找的 Tag 模式
    tag_patterns = {
        "acc": {"train": "Train/Avg_Acc", "test": "Test/Avg_Acc"},
        "loss": {"train": "Train/Avg_Loss", "test": "Test/Avg_Loss"}
    }
    
    # 设置学术风格
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = sns.color_palette("tab10", len(best_configs))
    
    for i, (surrogate, info) in enumerate(best_configs.items()):
        color = colors[i]
        log_path = info['path']
        config = info['config']
        label_prefix = f"{surrogate} (β={config['beta']}, lr={config['lr']:.0e})"
        
        # --- 绘制 Accuracy 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["acc"][mode]
            df = get_tb_data(log_path, tag)
            if df is not None:
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax1.plot(df['Epoch'], smooth_val, 
                       label=f"{label_prefix} ({mode.capitalize()})", 
                       color=color, linestyle=linestyle, linewidth=2)
        
        # --- 绘制 Loss 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["loss"][mode]
            df = get_tb_data(log_path, tag)
            if df is not None:
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax2.plot(df['Epoch'], smooth_val, 
                       label=f"{label_prefix} ({mode.capitalize()})", 
                       color=color, linestyle=linestyle, linewidth=2)
    
    # 细节修饰
    ax1.set_title("Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlim(0, 32)  # 确保x轴范围对齐到32个epoch
    ax1.legend(fontsize=9, loc='lower right', ncol=2)
    
    ax2.set_title("Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_xlim(0, 32)  # 确保x轴范围对齐到32个epoch
    ax2.legend(fontsize=9, loc='upper right', ncol=2)
    
    plt.suptitle("Best Configurations Comparison", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "task2_best_configs_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    """主函数：扫描实验并生成所有图表"""
    print("=" * 80)
    print("Task2 实验结果可视化")
    print("=" * 80)
    
    # 扫描所有实验
    print("\n扫描logs目录...")
    all_experiments = find_all_experiments('./logs')
    
    # 统计信息
    surrogates = ['SigmoidPrime', 'Esser', 'SuperSpike']
    for surrogate in surrogates:
        if surrogate in all_experiments:
            count = sum(len(lrs) for lrs in all_experiments[surrogate].values())
            print(f"  {surrogate}: {count} 个实验")
    
    # 生成每种surrogate的配置对比图
    print("\n生成每种surrogate的配置对比图...")
    output_dir = Path('./results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for surrogate in surrogates:
        if surrogate in all_experiments:
            output_path = output_dir / f"task2_{surrogate}_configs.png"
            plot_surrogate_configs(surrogate, all_experiments[surrogate], output_path)
    
    # 找到最佳配置并生成对比图
    print("\n查找最佳配置...")
    best_configs = find_best_configs(all_experiments)
    
    print("\n最佳配置详情:")
    for surrogate, info in best_configs.items():
        config = info['config']
        print(f"  {surrogate}: beta={config['beta']}, lr={config['lr']:.0e}, max_acc={info['max_acc']:.4f}")
    
    # 生成最优配置对比图（accuracy和loss在同一张图）
    print("\n生成最优配置对比图...")
    plot_best_configs_comparison(best_configs, output_dir)
    
    print("\n" + "=" * 80)
    print("所有图表生成完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

