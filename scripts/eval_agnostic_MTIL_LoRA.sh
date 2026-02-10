#!/bin/bash
set -v
set -e
set -x

GPU=0,1,2,3
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)

for ((i = 0; i < ${#dataset[@]}; i++)); do
    all_datasets=$(IFS=,; echo "${dataset[*]}")
    # semantic voting for prediction
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval \
        --lora True \
        --task_agnostic \
        --eval-datasets=${all_datasets} \
        --session=${i} \
        --save checkpoint/full_lora/ \
        >> log/full_lora/session_${i}.log 2>&1
done