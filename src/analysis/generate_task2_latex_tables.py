"""
生成Task2实验结果的LaTeX表格
每个surrogate梯度生成一个表格，横轴是beta，纵轴是lr
"""

import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def get_tb_data(log_dir, tag_name):
    """从单个 TensorBoard 目录中提取特定 tag 的最大值"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
        ea.Reload()
        
        if 'scalars' not in ea.Tags() or tag_name not in ea.Tags()['scalars']:
            return None
            
        events = ea.Scalars(tag_name)
        if events:
            return max([e.value for e in events])
        return None
    except Exception as e:
        return None

def extract_config_from_path(log_path):
    """从日志路径中提取训练配置"""
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
            return None
        
        # 提取lr
        if '_lr' in config_str:
            lr_part = config_str.split('_lr')[1].split('_')[0]
            config['lr'] = float(lr_part)
        else:
            return None
        
    except Exception as e:
        return None
    
    return config

def find_all_experiments(logs_dir='./logs'):
    """扫描logs目录，找到所有task2实验并组织数据"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        raise ValueError(f"Logs directory not found: {logs_dir}")
    
    # 组织数据：{surrogate: {beta: {lr: (log_path, test_acc)}}}
    experiments = defaultdict(lambda: defaultdict(dict))
    
    # 遍历所有目录
    for root, dirs, files in os.walk(logs_path):
        if any(f.startswith('events.out.tfevents') for f in files):
            parent_dir = str(Path(root).parent)
            config = extract_config_from_path(parent_dir)
            if config:
                surrogate = config['surrogate']
                beta = config['surrogate_beta']
                lr = config['lr']
                
                # 获取test accuracy
                test_acc = get_tb_data(root, "Test/Avg_Acc")
                
                # 如果该配置已经有数据，选择test accuracy更高的
                if lr not in experiments[surrogate][beta] or \
                   (test_acc is not None and 
                    (experiments[surrogate][beta][lr][1] is None or 
                     test_acc > experiments[surrogate][beta][lr][1])):
                    experiments[surrogate][beta][lr] = (root, test_acc)
    
    return experiments

def generate_latex_table(surrogate, experiments_dict, output_path):
    """生成LaTeX格式的表格"""
    # 收集所有beta和lr值
    betas = sorted(experiments_dict.keys())
    all_lrs = set()
    for beta in betas:
        all_lrs.update(experiments_dict[beta].keys())
    lrs = sorted(all_lrs)
    
    # 生成LaTeX表格
    latex_lines = []
    latex_lines.append(f"% {surrogate} - Test Accuracy Results")
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{c|" + "c" * len(betas) + "}")
    latex_lines.append("\\hline")
    
    # 表头：第一行是beta值
    header = "LR \\textbackslash{} Beta"
    for beta in betas:
        header += f" & {beta:.1f}"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    # 数据行：每行是一个lr值
    for lr in lrs:
        row = f"{lr:.0e}"
        for beta in betas:
            if lr in experiments_dict[beta]:
                test_acc = experiments_dict[beta][lr][1]
                if test_acc is not None:
                    row += f" & {test_acc:.4f}"
                else:
                    row += " & --"
            else:
                row += " & --"
        row += " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{Test accuracy for {surrogate} surrogate gradient with different beta and learning rate configurations.}}")
    latex_lines.append(f"\\label{{tab:task2_{surrogate.lower()}}}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")
    
    # 写入文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Saved: {output_path}")
    
    # 同时打印到控制台
    print("\n" + "=" * 80)
    print(f"{surrogate} LaTeX Table:")
    print("=" * 80)
    print('\n'.join(latex_lines))
    print("=" * 80 + "\n")

def main():
    """主函数：扫描实验并生成所有LaTeX表格"""
    print("=" * 80)
    print("生成Task2实验结果LaTeX表格")
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
    
    # 生成每种surrogate的LaTeX表格
    print("\n生成LaTeX表格...")
    output_dir = Path('./results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for surrogate in surrogates:
        if surrogate in all_experiments:
            output_path = output_dir / f"task2_{surrogate}_table.tex"
            generate_latex_table(surrogate, all_experiments[surrogate], output_path)
    
    print("\n" + "=" * 80)
    print("所有LaTeX表格生成完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

