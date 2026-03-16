#!/bin/bash
set -v
set -e
set -x

GPU=0
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 5e-5 1e-5 1e-5 1e-5 1e-5)

if [ ! -d "log/full_finetune" ]; then
    mkdir -p log/full_finetune
fi

if [ ! -d "checkpoint/full_finetune" ]; then
    mkdir -p checkpoint/full_finetune
fi

for ((i = 0; i < ${#dataset[@]}; i++)); do
    ASCEND_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=whole \
    --train-dataset=${dataset[i]} \
    --lr=${lr[i]} \
    --ls 0.2 \
    --iterations 1000 \
    --save checkpoint/full_finetune/ \
    --session=${i} \
    >> log/full_finetune/finetuned.log 2>&1
done
