"""
Task1 Baseline 实验结果可视化脚本
绘制task1 baseline的训练曲线（accuracy和loss）
完全遵循 plot_surrogate_comparison 的视觉风格
"""

import os
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

def find_task1_experiments(logs_dir='./logs'):
    """找到task1 baseline实验的日志目录（T=10和T=20）"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise ValueError(f"Logs directory not found: {logs_dir}")
    
    # Task1特征：Sigmoid_beta4.0, weight_decay=0.0001
    # 查找T=10和T=20的实验
    task1_experiments = {}  # {T: log_dir}
    
    for root, dirs, files in os.walk(logs_path):
        # 检查是否有TensorBoard日志
        if any(f.startswith('events.out.tfevents') for f in files):
            parent_dir = str(Path(root).parent)
            dir_name = Path(parent_dir).name
            
            # 检查是否是task1实验：Sigmoid_beta4.0, 不是SigmoidPrime
            if 'Sigmoid_beta4.0' in dir_name and 'SigmoidPrime' not in dir_name:
                # 检查weight_decay（可能是wd0.0001或wd0）
                if 'wd0.0001' in dir_name or 'wd0.0' in dir_name:
                    # 提取T值
                    if '_T10_' in dir_name:
                        T = 10
                    elif '_T20_' in dir_name:
                        T = 20
                    else:
                        continue
                    
                    # 如果该T值已经有数据，选择最新的（路径最大的）
                    if T not in task1_experiments or root > task1_experiments[T]:
                        task1_experiments[T] = root
    
    return task1_experiments

def plot_task1_baseline(task1_experiments, output_path='./results/task1_baseline.png'):
    """
    绘制task1 baseline的训练曲线（T=10和T=20对比）
    完全遵循 plot_surrogate_comparison 的风格
    """
    if not task1_experiments:
        print("Error: No task1 experiment found")
        return
    
    # 设置学术风格
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义要查找的 Tag 模式
    tag_patterns = {
        "acc": {"train": "Train/Avg_Acc", "test": "Test/Avg_Acc"},
        "loss": {"train": "Train/Avg_Loss", "test": "Test/Avg_Loss"}
    }
    
    # 为不同的T值分配颜色
    colors = sns.color_palette("tab10", 2)
    color_map = {10: colors[0], 20: colors[1]}
    
    # 绘制每个T值的曲线
    for T in sorted(task1_experiments.keys()):
        log_dir = task1_experiments[T]
        color = color_map[T]
        
        # --- 绘制 Accuracy 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["acc"][mode]
            df = get_tb_data(log_dir, tag)
            if df is not None:
                # 平滑处理 (alpha越小越平滑)
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax1.plot(df['Epoch'], smooth_val, 
                         label=f"T={T} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)
        
        # --- 绘制 Loss 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["loss"][mode]
            df = get_tb_data(log_dir, tag)
            if df is not None:
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax2.plot(df['Epoch'], smooth_val, 
                         label=f"T={T} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)
    
    # 细节修饰
    ax1.set_title("Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=9, loc='lower right', ncol=2)
    
    ax2.set_title("Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9, loc='upper right', ncol=2)
    
    plt.suptitle("Task1 Baseline - Training Curves (T=10 vs T=20)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    """主函数：找到task1实验并生成图表"""
    print("=" * 80)
    print("Task1 Baseline 实验结果可视化")
    print("=" * 80)
    
    # 查找task1实验（T=10和T=20）
    print("\n查找task1 baseline实验...")
    task1_experiments = find_task1_experiments('./logs')
    
    if task1_experiments:
        for T, log_dir in sorted(task1_experiments.items()):
            print(f"  找到 T={T} 实验: {log_dir}")
        
        # 生成图表
        print("\n生成训练曲线图...")
        output_path = './results/task1_baseline.png'
        plot_task1_baseline(task1_experiments, output_path)
        
        print("\n" + "=" * 80)
        print("图表生成完成！")
        print("=" * 80)
    else:
        print("\n未找到task1 baseline实验，请检查logs目录")

if __name__ == "__main__":
    main()

