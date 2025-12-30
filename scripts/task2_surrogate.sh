#!/bin/bash
surrogates=(SigmoidPrime Esser SuperSpike)
surrogate_betas=(10.0 5.0 2.0 1.0)
lrs=(1e-3 5e-4 1e-4 5e-5 1e-5)

for surrogate in ${surrogates[@]}; do
    for surrogate_beta in ${surrogate_betas[@]}; do
        for lr in ${lrs[@]}; do
            python src/train/train_cifar10.py --surrogate $surrogate --surrogate_beta $surrogate_beta --T $T --lr $lr
        done
    done
done


