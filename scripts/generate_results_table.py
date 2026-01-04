#!/usr/bin/env python3
"""
生成Task2实验结果表格
横轴：不同方法（SigmoidPrime, Esser, SuperSpike）
纵轴：surrogate_betas（10.0, 5.0, 2.0, 1.0）
每个方格：不同学习率对应的train和test acc
"""
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_accuracies_from_tb(log_dir):
    """从TensorBoard日志获取train和test accuracy"""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        
        train_tag = 'Train/Avg_Acc' if 'Train/Avg_Acc' in tags else ('Train/Accuracy' if 'Train/Accuracy' in tags else None)
        test_tag = 'Test/Avg_Acc' if 'Test/Avg_Acc' in tags else ('Test/Accuracy' if 'Test/Accuracy' in tags else None)
        
        train_acc = None
        test_acc = None
        
        if train_tag:
            train_events = ea.Scalars(train_tag)
            if train_events:
                train_acc = max([e.value for e in train_events])
        
        if test_tag:
            test_events = ea.Scalars(test_tag)
            if test_events:
                test_acc = max([e.value for e in test_events])
        
        return train_acc, test_acc
    except:
        return None, None

def extract_config_from_path(log_path):
    """从日志路径中提取训练配置"""
    parts = log_path.split('/')
    if len(parts) < 2:
        return None
    
    config_str = parts[-2]
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
            return None
        
        # 提取lr
        if '_lr' in config_str:
            lr_part = config_str.split('_lr')[1].split('_')[0]
            config['lr'] = float(lr_part)
        else:
            return None
        
    except:
        return None
    
    return config

def main():
    logs_dir = './logs'
    
    # 初始化结果字典
    surrogates = ['SigmoidPrime', 'Esser', 'SuperSpike']
    betas = [10.0, 5.0, 2.0, 1.0]
    lrs = [5e-2, 3e-2, 2e-2, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]  # 包含新的高学习率
    
    results = {}
    for surrogate in surrogates:
        results[surrogate] = {}
        for beta in betas:
            results[surrogate][beta] = {}
            for lr in lrs:
                results[surrogate][beta][lr] = {'train': None, 'test': None}
    
    # 从logs中提取结果
    for root, dirs, files in os.walk(logs_dir):
        if any(f.startswith('events.out.tfevents') for f in files):
            config = extract_config_from_path(root)
            if config:
                train_acc, test_acc = get_accuracies_from_tb(root)
                surrogate = config['surrogate']
                beta = config['surrogate_beta']
                lr = config['lr']
                
                if surrogate in results and beta in results[surrogate] and lr in results[surrogate][beta]:
                    results[surrogate][beta][lr]['train'] = train_acc
                    results[surrogate][beta][lr]['test'] = test_acc
    
    # 生成表格
    print("=" * 260)
    print("Task2 实验结果表格")
    print("=" * 260)
    print("\n格式说明：每个单元格显示9个学习率的结果，格式为 (train_acc, test_acc)")
    print("学习率顺序：5e-2, 3e-2, 2e-2, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5\n")
    
    # 表头
    header = f"{'Beta':<8} |"
    for surrogate in surrogates:
        header += f" {surrogate:^80} |"  # 增加列宽以容纳更多学习率
    print(header)
    print("-" * 260)  # 增加分隔线长度
    
    # 每个beta一行
    for beta in betas:
        row = f"{beta:<8} |"
        for surrogate in surrogates:
            cell_content = ""
            for i, lr in enumerate(lrs):
                train = results[surrogate][beta][lr]['train']
                test = results[surrogate][beta][lr]['test']
                
                if train is not None and test is not None:
                    cell_content += f"({train:.4f},{test:.4f})"
                elif test is not None:
                    cell_content += f"(?,{test:.4f})"
                elif train is not None:
                    cell_content += f"({train:.4f},?)"
                else:
                    cell_content += "(?,?)"
                
                if i < len(lrs) - 1:
                    cell_content += " "
            
            row += f" {cell_content:<78} |"  # 增加单元格宽度
        print(row)
        print("-" * 260)  # 增加分隔线长度
    
    # 生成CSV格式（更易读）
    print("\n" + "=" * 150)
    print("CSV格式表格（便于复制到Excel）")
    print("=" * 260)
    
    # CSV表头
    csv_header = "Beta,Surrogate,LR_5e-2_Train,LR_5e-2_Test,LR_3e-2_Train,LR_3e-2_Test,LR_2e-2_Train,LR_2e-2_Test,LR_1e-2_Train,LR_1e-2_Test,LR_1e-3_Train,LR_1e-3_Test,LR_5e-4_Train,LR_5e-4_Test,LR_1e-4_Train,LR_1e-4_Test,LR_5e-5_Train,LR_5e-5_Test,LR_1e-5_Train,LR_1e-5_Test"
    print(csv_header)
    
    for beta in betas:
        for surrogate in surrogates:
            row = f"{beta},{surrogate}"
            for lr in lrs:
                train = results[surrogate][beta][lr]['train']
                test = results[surrogate][beta][lr]['test']
                train_str = f"{train:.4f}" if train is not None else "N/A"
                test_str = f"{test:.4f}" if test is not None else "N/A"
                row += f",{train_str},{test_str}"
            print(row)
    
    # 生成Markdown表格
    print("\n" + "=" * 150)
    print("Markdown格式表格")
    print("=" * 260)
    
    # Markdown表头
    md_header = "| Beta | Surrogate | LR=5e-2 (Train, Test) | LR=3e-2 (Train, Test) | LR=2e-2 (Train, Test) | LR=1e-2 (Train, Test) | LR=1e-3 (Train, Test) | LR=5e-4 (Train, Test) | LR=1e-4 (Train, Test) | LR=5e-5 (Train, Test) | LR=1e-5 (Train, Test) |"
    md_separator = "|------|-----------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|"
    print(md_header)
    print(md_separator)
    
    for beta in betas:
        for surrogate in surrogates:
            row = f"| {beta} | {surrogate} |"
            for lr in lrs:
                train = results[surrogate][beta][lr]['train']
                test = results[surrogate][beta][lr]['test']
                train_str = f"{train:.4f}" if train is not None else "N/A"
                test_str = f"{test:.4f}" if test is not None else "N/A"
                row += f" ({train_str}, {test_str}) |"
            print(row)
    
    # 统计缺失数据
    print("\n" + "=" * 150)
    print("数据完整性统计")
    print("=" * 260)
    total = len(surrogates) * len(betas) * len(lrs)
    complete = 0
    missing_train = 0
    missing_test = 0
    missing_both = 0
    
    for surrogate in surrogates:
        for beta in betas:
            for lr in lrs:
                train = results[surrogate][beta][lr]['train']
                test = results[surrogate][beta][lr]['test']
                if train is not None and test is not None:
                    complete += 1
                elif train is None and test is None:
                    missing_both += 1
                elif train is None:
                    missing_train += 1
                elif test is None:
                    missing_test += 1
    
    print(f"总实验数: {total}")
    print(f"完整数据（train+test）: {complete} ({complete/total*100:.1f}%)")
    print(f"缺失train: {missing_train}")
    print(f"缺失test: {missing_test}")
    print(f"两者都缺失: {missing_both}")

if __name__ == '__main__':
    main()

