import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

def get_tb_data(log_dir, tag_name):
    """从单个 TensorBoard 目录中提取特定 tag 的数据"""
    # size_guidance 设置为 0 表示加载所有数据点
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    if tag_name not in ea.Tags()['scalars']:
        # 尝试一些常见的变体，比如 'Accuracy/train' vs 'train/accuracy'
        print(f"Warning: Tag '{tag_name}' not found in {log_dir}")
        return None
        
    events = ea.Scalars(tag_name)
    return pd.DataFrame([(e.step, e.value) for e in events], columns=['Epoch', 'Value'])

def plot_surrogate_comparison(methods_config, output_name="comparison_plot.png"):
    """
    methods_config: { "方法名": "日志目录路径" }
    """
    # 设置学术风格
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 定义要查找的 Tag 模式 (请根据你代码中 writer.add_scalar 的具体字符串修改)
    # 常见格式为 "Accuracy/train" 或 "train/accuracy"
    tag_patterns = {
        "acc": {"train": "Train/Avg_Acc", "test": "Test/Avg_Acc"},
        "loss": {"train": "Train/Avg_Loss", "test": "Test/Avg_Loss"}
    }
    
    colors = sns.color_palette("tab10", len(methods_config))
    
    for i, (name, log_path) in enumerate(methods_config.items()):
        color = colors[i]
        
        # --- 绘制 Accuracy 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["acc"][mode]
            df = get_tb_data(log_path, tag)
            if df is not None:
                # 平滑处理 (alpha越小越平滑)
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax1.plot(df['Epoch'], smooth_val, 
                         label=f"{name} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)

        # --- 绘制 Loss 图 ---
        for mode, linestyle in [("train", "-"), ("test", "--")]:
            tag = tag_patterns["loss"][mode]
            df = get_tb_data(log_path, tag)
            if df is not None:
                smooth_val = df['Value'].ewm(alpha=0.15).mean()
                ax2.plot(df['Epoch'], smooth_val, 
                         label=f"{name} ({mode.capitalize()})", 
                         color=color, linestyle=linestyle, linewidth=2)

    # 细节修饰
    ax1.set_title("Accuracy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=9, loc='lower right', ncol=2) # ncol=2 可以让图例排成两列，节省空间

    ax2.set_title("Loss", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9, loc='upper right', ncol=2)

    plt.suptitle("Surrogate Gradient Comparison", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    plt.show()

# --- 调用示例 ---
if __name__ == "__main__":
    # 这里填入每个实验运行的根目录
    # 假设你的目录结构是：
    # ./logs/atan_run/events.out.tfevents... (内含 train/loss 和 test/loss)
    my_runs = {
        "baseline": "/home/mulab/data/tpz/SNN_Activation_Driven/logs/cifar10dvs_advanced/preprocess_baseline/20251223_202859",
        "count_norm": "/home/mulab/data/tpz/SNN_Activation_Driven/logs/cifar10dvs_advanced/preprocess_count_norm/20251223_204803",
        "log_scale": "/home/mulab/data/tpz/SNN_Activation_Driven/logs/cifar10dvs_advanced/preprocess_time_surface/20251223_210800"
    }

    plot_surrogate_comparison(my_runs)