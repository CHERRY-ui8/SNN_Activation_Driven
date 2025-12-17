"""
简洁的loss和accuracy分析脚本
用于分析事件驱动训练的结果
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_results(log_dir):
    """分析训练结果"""
    log_path = Path(log_dir)
    
    # 读取JSON文件
    json_path = log_path / 'training_results.json'
    if not json_path.exists():
        print(f"错误: 未找到结果文件 {json_path}")
        print("请确保训练已完成并生成了 training_results.json 文件")
        return
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    epochs_data = results['epochs']
    best_epoch = results.get('best_epoch', 0)
    best_test_acc = results.get('best_test_acc', 0)
    
    # 提取数据
    epochs = [d['epoch'] for d in epochs_data]
    train_loss = [d['train_loss'] for d in epochs_data]
    test_loss = [d['test_loss'] for d in epochs_data]
    train_acc = [d['train_acc'] for d in epochs_data]
    test_acc = [d['test_acc'] for d in epochs_data]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss曲线
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', linewidth=2, color='blue')
    axes[0, 0].plot(epochs, test_loss, label='Test Loss', linewidth=2, color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy曲线
    axes[0, 1].plot(epochs, train_acc, label='Train Acc', linewidth=2, color='blue')
    axes[0, 1].plot(epochs, test_acc, label='Test Acc', linewidth=2, color='red')
    axes[0, 1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss对比（训练vs测试）
    axes[1, 0].plot(epochs, train_loss, label='Train', linewidth=2, alpha=0.7)
    axes[1, 0].plot(epochs, test_loss, label='Test', linewidth=2, alpha=0.7)
    axes[1, 0].fill_between(epochs, train_loss, test_loss, alpha=0.2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Train vs Test Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Accuracy对比（训练vs测试）
    axes[1, 1].plot(epochs, train_acc, label='Train', linewidth=2, alpha=0.7)
    axes[1, 1].plot(epochs, test_acc, label='Test', linewidth=2, alpha=0.7)
    axes[1, 1].fill_between(epochs, train_acc, test_acc, alpha=0.2)
    axes[1, 1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Train vs Test Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = log_path / 'loss_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"分析图表已保存到: {save_path}")
    
    plt.close()
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("训练结果统计")
    print("=" * 60)
    print(f"总训练轮数: {len(epochs)}")
    print(f"最佳测试准确率: {best_test_acc:.2f}% (Epoch {best_epoch})")
    print(f"最终训练Loss: {train_loss[-1]:.4f}")
    print(f"最终测试Loss: {test_loss[-1]:.4f}")
    print(f"最终训练准确率: {train_acc[-1]:.2f}%")
    print(f"最终测试准确率: {test_acc[-1]:.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='分析事件驱动训练结果')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='日志目录路径（包含training_results.json的目录）')
    
    args = parser.parse_args()
    
    analyze_results(args.log_dir)


if __name__ == '__main__':
    main()
