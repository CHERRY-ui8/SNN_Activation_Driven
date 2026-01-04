#!/bin/bash
# Task1 Baseline with Strong Data Augmentation
CUDA_VISIBLE_DEVICES=5 python src/train/train_cifar10.py --T 20 --weight_decay 0.0001 --patience 0 --batch_size 96 --strong_aug