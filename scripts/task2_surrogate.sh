#!/bin/bash
T=10  # time step (length of spike sequence)
surrogates=(SigmoidPrime Esser SuperSpike)
surrogate_betas=(10.0 5.0 2.0 1.0)
lrs=(5e-2 3e-2 2e-2 5e-3 3e-3 1e-2 2e-3 1e-3 5e-4 1e-4 5e-5 1e-5)

for surrogate in ${surrogates[@]}; do
    for surrogate_beta in ${surrogate_betas[@]}; do
        for lr in ${lrs[@]}; do
            CUDA_VISIBLE_DEVICES=2 python src/train/train_cifar10.py --surrogate $surrogate --surrogate_beta $surrogate_beta --T $T --lr $lr --epochs 32
        done
    done
done


