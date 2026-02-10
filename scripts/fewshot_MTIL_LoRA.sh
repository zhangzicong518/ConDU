#!/bin/bash
set -v
set -e
set -x

GPU=0,1,2,3
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
lr=(1e-3 5e-4 2e-3 2e-3 1.5e-4 1e-5 2e-3 1e-3 1.5e-4 2e-3 1.5e-3 1e-3)

if [ ! -d "log/fewshot_lora" ]; then
    mkdir -p log/fewshot_lora
fi

if [ ! -d "checkpoint/fewshot_lora" ]; then
    mkdir -p checkpoint/fewshot_lora
fi

for ((i = 0; i < ${#dataset[@]}; i++)); do
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main \
    --train-mode=whole \
    --train-dataset=${dataset[i]} \
    --lr=${lr[i]} \
    --ls 0.2 \
    --iterations 500 \
    --few_shot 5 \
    --lora True \
    --save checkpoint/fewshot_lora/ \
    --session=${i} \
    >> log/fewshot_lora/finetuned.log 2>&1
done
