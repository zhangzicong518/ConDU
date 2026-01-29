#!/bin/bash
set -v
set -e
set -x

GPU=0,1
dataset=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)

for ((i = 0; i < ${#dataset[@]}; i++)); do
    eval_datasets_current=$(IFS=,; echo "${dataset[*]:0:$((i+1))}")
    # use the corresponding model
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval \
        --eval-datasets=${eval_datasets_current} \
        --session=${i} \
        --save checkpoint/full_finetune/ \
        >> log/full_finetune/session_${i}.log 2>&1

    eval_datasets_future=$(IFS=,; echo "${dataset[*]:$((i+1))}")
    # semantic voting for prediction
    CUDA_VISIBLE_DEVICES=${GPU} python -m src.main --eval \
        --task_agnostic \
        --eval-datasets=${eval_datasets_future} \
        --session=${i} \
        --save checkpoint/full_finetune/ \
        >> log/full_finetune/session_${i}.log 2>&1
done