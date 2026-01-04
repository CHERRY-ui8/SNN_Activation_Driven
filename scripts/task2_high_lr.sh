#!/bin/bash
# Task2 高学习率实验
# 针对表现最好的组合，测试更大的学习率
# 基于之前的结果，1e-3表现最好，现在测试 2e-3, 3e-3, 5e-3

T=10  # time step (length of spike sequence)

# 重点测试表现最好的组合
# 1. Esser beta=2.0 (最佳: 89.74% @ 1e-3)
# 2. Esser beta=1.0 (85.30% @ 1e-3)
# 3. SuperSpike beta=5.0 (85.82% @ 1e-3)
# 4. SigmoidPrime beta=5.0 (79.47% @ 1e-3)

high_lrs=(2e-2 3e-2 5e-2)

echo "开始高学习率实验..."
echo "测试学习率: ${high_lrs[@]}"
echo ""

# Esser beta=2.0 (最佳组合)
# for lr in ${high_lrs[@]}; do
#     echo "Esser beta=2.0 lr=$lr"
#     CUDA_VISIBLE_DEVICES=0 python src/train/train_cifar10.py \
#         --surrogate Esser \
#         --surrogate_beta 2.0 \
#         --T $T \
#         --lr $lr \
#         --epochs 32 \
#         --batch_size 64 \
#         --patience 0
#     echo ""
# done

# Esser beta=1.0
# for lr in ${high_lrs[@]}; do
#     echo "Esser beta=1.0 lr=$lr"
#     CUDA_VISIBLE_DEVICES=0 python src/train/train_cifar10.py \
#         --surrogate Esser \
#         --surrogate_beta 1.0 \
#         --T $T \
#         --lr $lr \
#         --epochs 32 \
#         --batch_size 64 \
#         --patience 0
#     echo ""
# done

# SuperSpike beta=5.0
for lr in ${high_lrs[@]}; do
    echo "SuperSpike beta=5.0 lr=$lr"
    CUDA_VISIBLE_DEVICES=0 python src/train/train_cifar10.py \
        --surrogate SuperSpike \
        --surrogate_beta 5.0 \
        --T $T \
        --lr $lr \
        --epochs 32 \
        --batch_size 64 \
        --patience 0
    echo ""
done

# SigmoidPrime beta=5.0
for lr in ${high_lrs[@]}; do
    echo "SigmoidPrime beta=5.0 lr=$lr"
    CUDA_VISIBLE_DEVICES=0 python src/train/train_cifar10.py \
        --surrogate SigmoidPrime \
        --surrogate_beta 5.0 \
        --T $T \
        --lr $lr \
        --epochs 32 \
        --batch_size 64 \
        --patience 0
    echo ""
done

echo "高学习率实验完成！"

