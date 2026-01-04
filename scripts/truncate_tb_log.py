#!/usr/bin/env python3
"""
截断TensorBoard日志文件，只保留前N个epoch的数据
"""
import os
import sys
import shutil
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TORCH = True
except ImportError:
    try:
        from tensorflow.summary import SummaryWriter
        USE_TORCH = False
    except ImportError:
        print("Error: Need either torch or tensorflow to write TensorBoard logs")
        sys.exit(1)

def truncate_tb_log(log_dir, max_epoch=32, backup=True):
    """
    截断TensorBoard日志，只保留前max_epoch个epoch的数据
    
    Args:
        log_dir: TensorBoard日志目录路径
        max_epoch: 保留的最大epoch数（step从0开始，所以保留0到max_epoch-1）
        backup: 是否备份原始文件
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Error: Log directory not found: {log_dir}")
        return False
    
    # 查找所有TensorBoard事件文件
    event_files = list(log_path.glob("events.out.tfevents.*"))
    if not event_files:
        print(f"Error: No TensorBoard event files found in {log_dir}")
        return False
    
    print(f"Found {len(event_files)} event file(s) in {log_dir}")
    
    # 读取原始数据
    ea = EventAccumulator(str(log_path))
    ea.Reload()
    
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print(f"Error: No scalar tags found in {log_dir}")
        return False
    
    print(f"Available tags: {tags}")
    
    # 检查数据范围
    for tag in tags[:2]:  # 只检查前两个tag作为示例
        events = ea.Scalars(tag)
        if events:
            max_step = max(e.step for e in events)
            print(f"  {tag}: {len(events)} events, max step: {max_step}")
    
    # 备份原始文件
    if backup:
        backup_dir = log_path.parent / f"{log_path.name}_backup_64epochs"
        if not backup_dir.exists():
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(log_path, backup_dir)
        else:
            print(f"Backup already exists: {backup_dir}")
    
    # 删除原始事件文件
    print(f"Removing original event files...")
    for event_file in event_files:
        event_file.unlink()
    
    # 创建新的SummaryWriter
    writer = SummaryWriter(str(log_path))
    
    # 写入截断后的数据
    print(f"Writing truncated data (max epoch: {max_epoch})...")
    for tag in tags:
        events = ea.Scalars(tag)
        filtered_events = [e for e in events if e.step < max_epoch]
        
        if not filtered_events:
            print(f"  Warning: No events found for {tag} with step < {max_epoch}")
            continue
        
        print(f"  {tag}: {len(events)} -> {len(filtered_events)} events")
        
        for event in filtered_events:
            writer.add_scalar(tag, event.value, int(event.step))
    
    writer.close()
    print(f"Successfully truncated log to {max_epoch} epochs!")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python truncate_tb_log.py <log_dir> [max_epoch]")
        print("Example: python truncate_tb_log.py logs/experiment/20251231_131817 32")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    max_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    
    truncate_tb_log(log_dir, max_epoch=max_epoch)

