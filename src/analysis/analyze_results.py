"""
训练结果分析脚本

用于分析CIFAR数据集上的训练结果，包括：
1. 准确率曲线
2. 损失曲线
3. 混淆矩阵可视化
4. 各类别性能分析
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, log_dir):
        """
        初始化分析器
        
        Args:
            log_dir: 日志目录路径
        """
        self.log_dir = Path(log_dir)
        if not self.log_dir.exists():
            raise ValueError(f"日志目录不存在: {log_dir}")
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print(f"分析结果目录: {self.log_dir}")
    
    def load_checkpoint(self, checkpoint_name='best.pth'):
        """加载模型检查点"""
        checkpoint_path = self.log_dir / checkpoint_name
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"加载检查点: {checkpoint_name}")
            print(f"  Epoch: {checkpoint['epoch']}")
            print(f"  准确率: {checkpoint.get('accuracy', 'N/A'):.2f}%")
            return checkpoint
        else:
            print(f"检查点不存在: {checkpoint_path}")
            return None
    
    def load_confusion_matrices(self):
        """加载混淆矩阵"""
        confusion_path = self.log_dir / 'confusion_matrices.npy'
        if confusion_path.exists():
            confusion_matrices = np.load(confusion_path)
            print(f"加载混淆矩阵: {confusion_matrices.shape}")
            return confusion_matrices
        else:
            print("混淆矩阵文件不存在")
            return None
    
    def plot_training_curves(self, save_path=None):
        """
        绘制训练曲线
        
        从TensorBoard日志中读取数据并绘制
        """
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            # 查找TensorBoard日志
            event_files = list(self.log_dir.glob('events.out.tfevents.*'))
            if not event_files:
                print("未找到TensorBoard日志文件")
                return
            
            # 加载事件
            ea = EventAccumulator(str(self.log_dir))
            ea.Reload()
            
            # 获取标量数据
            scalars = ea.Tags()['scalars']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制准确率曲线
            if 'Accuracy/Train' in scalars and 'Accuracy/Test' in scalars:
                train_acc = ea.Scalars('Accuracy/Train')
                test_acc = ea.Scalars('Accuracy/Test')
                
                epochs = [s.step for s in train_acc]
                train_values = [s.value for s in train_acc]
                test_values = [s.value for s in test_acc]
                
                axes[0, 0].plot(epochs, train_values, label='Train', linewidth=2)
                axes[0, 0].plot(epochs, test_values, label='Test', linewidth=2)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy (%)')
                axes[0, 0].set_title('Accuracy Curves')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 绘制损失曲线
            if 'Loss/Train' in scalars:
                train_loss = ea.Scalars('Loss/Train')
                epochs = [s.step for s in train_loss]
                values = [s.value for s in train_loss]
                
                axes[0, 1].plot(epochs, values, label='Train Loss', linewidth=2, color='red')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('Training Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].set_yscale('log')
            
            # 绘制学习率曲线
            if 'LearningRate' in scalars:
                lr = ea.Scalars('LearningRate')
                epochs = [s.step for s in lr]
                values = [s.value for s in lr]
                
                axes[1, 0].plot(epochs, values, linewidth=2, color='green')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 绘制准确率对比（训练 vs 测试）
            if 'Accuracy/Train' in scalars and 'Accuracy/Test' in scalars:
                axes[1, 1].plot(epochs, train_values, label='Train', linewidth=2, alpha=0.7)
                axes[1, 1].plot(epochs, test_values, label='Test', linewidth=2, alpha=0.7)
                axes[1, 1].fill_between(epochs, train_values, test_values, alpha=0.2)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Accuracy (%)')
                axes[1, 1].set_title('Train vs Test Accuracy')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"训练曲线已保存到: {save_path}")
            else:
                save_path = self.log_dir / 'training_curves.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"训练曲线已保存到: {save_path}")
            
            plt.close()
            
        except ImportError:
            print("需要安装tensorboard: pip install tensorboard")
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")
    
    def plot_confusion_matrix(self, epoch=-1, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            epoch: 要绘制的epoch（-1表示最后一个）
            save_path: 保存路径
        """
        confusion_matrices = self.load_confusion_matrices()
        if confusion_matrices is None:
            return
        
        # 选择要绘制的混淆矩阵
        if epoch == -1:
            confusion = confusion_matrices[-1]
            epoch_num = len(confusion_matrices)
        else:
            confusion = confusion_matrices[epoch - 1]
            epoch_num = epoch
        
        # 确定类别数量
        n_classes = confusion.shape[0]
        
        # 绘制混淆矩阵
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues', 
                   cbar_kws={'label': 'Normalized Count'})
        plt.title(f'Confusion Matrix (Epoch {epoch_num})', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        else:
            save_path = self.log_dir / f'confusion_matrix_epoch_{epoch_num}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.close()
    
    def analyze_class_performance(self, checkpoint_name='best.pth'):
        """
        分析各类别性能
        
        计算每个类别的准确率、精确率、召回率等指标
        """
        confusion_matrices = self.load_confusion_matrices()
        if confusion_matrices is None:
            return
        
        # 使用最后一个epoch的混淆矩阵
        confusion = confusion_matrices[-1]
        n_classes = confusion.shape[0]
        
        # 计算各类别指标
        class_metrics = []
        
        for i in range(n_classes):
            # 真正例、假正例、真负例、假负例
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            tn = 1 - tp - fp - fn
            
            # 准确率（该类别的准确率）
            accuracy = tp
            
            # 精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # 召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics.append({
                'class': i,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        # 打印结果
        print("\n" + "=" * 80)
        print("各类别性能分析")
        print("=" * 80)
        print(f"{'类别':<8} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1分数':<12}")
        print("-" * 80)
        
        for metrics in class_metrics:
            print(f"{metrics['class']:<8} "
                  f"{metrics['accuracy']*100:>10.2f}% "
                  f"{metrics['precision']*100:>10.2f}% "
                  f"{metrics['recall']*100:>10.2f}% "
                  f"{metrics['f1']*100:>10.2f}%")
        
        # 计算平均指标
        avg_accuracy = np.mean([m['accuracy'] for m in class_metrics])
        avg_precision = np.mean([m['precision'] for m in class_metrics])
        avg_recall = np.mean([m['recall'] for m in class_metrics])
        avg_f1 = np.mean([m['f1'] for m in class_metrics])
        
        print("-" * 80)
        print(f"{'平均':<8} "
              f"{avg_accuracy*100:>10.2f}% "
              f"{avg_precision*100:>10.2f}% "
              f"{avg_recall*100:>10.2f}% "
              f"{avg_f1*100:>10.2f}%")
        print("=" * 80)
        
        # 保存结果
        results_path = self.log_dir / 'class_performance.json'
        with open(results_path, 'w') as f:
            json.dump({
                'class_metrics': class_metrics,
                'averages': {
                    'accuracy': avg_accuracy,
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1
                }
            }, f, indent=2)
        
        print(f"\n结果已保存到: {results_path}")
        
        return class_metrics
    
    def plot_class_performance(self, save_path=None):
        """绘制各类别性能对比图"""
        confusion_matrices = self.load_confusion_matrices()
        if confusion_matrices is None:
            return
        
        confusion = confusion_matrices[-1]
        n_classes = confusion.shape[0]
        
        # 计算各类别准确率
        class_accuracies = [confusion[i, i] for i in range(n_classes)]
        
        # 绘制柱状图
        plt.figure(figsize=(max(12, n_classes * 0.5), 6))
        bars = plt.bar(range(n_classes), [acc * 100 for acc in class_accuracies], 
                      color='steelblue', alpha=0.7)
        
        # 添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
        plt.xticks(range(n_classes))
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 105)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别性能图已保存到: {save_path}")
        else:
            save_path = self.log_dir / 'class_performance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"类别性能图已保存到: {save_path}")
        
        plt.close()
    
    def generate_report(self):
        """生成完整的分析报告"""
        print("\n" + "=" * 80)
        print("生成分析报告")
        print("=" * 80)
        
        # 加载检查点信息
        checkpoint = self.load_checkpoint('best.pth')
        if checkpoint:
            print(f"\n最佳模型信息:")
            print(f"  Epoch: {checkpoint['epoch']}")
            print(f"  准确率: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        
        # 绘制训练曲线
        print("\n绘制训练曲线...")
        self.plot_training_curves()
        
        # 绘制混淆矩阵
        print("\n绘制混淆矩阵...")
        self.plot_confusion_matrix()
        
        # 分析类别性能
        print("\n分析类别性能...")
        self.analyze_class_performance()
        
        # 绘制类别性能图
        print("\n绘制类别性能图...")
        self.plot_class_performance()
        
        print("\n" + "=" * 80)
        print("分析完成！所有图表已保存到日志目录。")
        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析训练结果')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='日志目录路径')
    parser.add_argument('--epoch', type=int, default=-1,
                       help='要分析的epoch（-1表示最后一个）')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ResultAnalyzer(args.log_dir)
    
    # 生成完整报告
    analyzer.generate_report()


if __name__ == '__main__':
    main()

