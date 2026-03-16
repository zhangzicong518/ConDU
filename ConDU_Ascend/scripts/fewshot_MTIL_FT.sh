#!/bin/bash
set -v
set -e
set -x

GPU=0
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(5e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5 5e-5 1e-5 1e-5 1e-5)

if [ ! -d "log/fewshot_full_finetune" ]; then
    mkdir -p log/fewshot_full_finetune
fi

if [ ! -d "checkpoint/fewshot_full_finetune" ]; then
    mkdir -p checkpoint/fewshot_full_finetune
fi

for ((i = 0; i < ${#dataset[@]}; i++)); do
    ASCEND_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=whole \
    --train-dataset=${dataset[i]} \
    --lr=${lr[i]} \
    --ls 0.2 \
    --iterations 500 \
    --few_shot 5 \
    --save checkpoint/fewshot_full_finetune/ \
    --session=${i} \
    >> log/fewshot_full_finetune/finetuned.log 2>&1
done
