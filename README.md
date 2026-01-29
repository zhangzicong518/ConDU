# Enhanced Continual Learning of Vision-Language Model

This is the official implementation of paper "[Enhanced Continual Learning of Vision-Language Models with Model Fusion](https://arxiv.org/abs/2503.10705)". We propose Continual Decoupling-Unifying (ConDU), a novel approach, by introducing model fusion into continual learning for VLMs. ConDU maintains a unified model along with task triggers and prototype sets, employing an iterative process of decoupling task-specific models for previous tasks and unifying them with the model for the newly learned task.

## Dataset Preparation

We follow the setting of **Multi-domain Task Incremental Learning (MTIL)** Benchmark. There are 11 required datasets to be donwloaded and placed as following structure. You can refers to [datasets.md](./data/datasets.md) for more details.

~~~
data
├── caltech101
├── cifar-100-python
├── dtd
├── eurosat
├── fgvc-aircraft-2013b
├── flowers-102
├── food-101
├── MNIST
├── oxford-iiit-pet
├── stanford_cars
├── SUN397
~~~

## Training Stage 

If you want to replicate our results in the paper, you can directly run the corresponding scripts to finish the training stage. 
- For benchmark MTIL and task-agnostic MTIL, run `bash scripts/MTIL_FT.sh` to use the method ConDU (FT) or run `bash scripts/MTIL_LoRA.sh` to use the method ConDU (LoRA)
- For benchmark few-shot MTIL, run `bash scripts/fewshot_MTIL_FT.sh` to use the method ConDU (FT) or run `bash scripts/fewshot_MTIL_LoRA.sh` to use the method ConDU (LoRA)


## Evaluation Stage

If you want to replicate our results in the paper, you can directly run the corresponding scripts to finish the evaluation stage. 
- For benchmark MTIL and fewshot MTIL, run `bash scripts/eval_MTIL_FT.sh` to evaluate the method ConDU (FT) or run `bash scripts/eval_MTIL_LoRA.sh` to evaluate the method ConDU (LoRA)
- For benchmark task-agnostic MTIL, run `bash scripts/eval_agnostic_MTIL_FT.sh` to evaluate the method ConDU (FT) or run `bash scripts/eval_agnostic_MTIL_LoRA.sh` to evaluate the method ConDU (LoRA)

## Citation

~~~
@article{gao2025enhanced,
  title={Enhanced Continual Learning of Vision-Language Models with Model Fusion},
  author={Gao, Haoyuan and Zhang, Zicong and Wei, Yuqi and Zhao, Linglan and Li, Guilin and Li, Yexin and Kong, Linghe and Huang, Weiran},
  journal={arXiv preprint arXiv:2503.10705},
  year={2025}
}
~~~

## Acknowledgement

Part of our code is built on [ZSCL](https://github.com/Thunderbeee/ZSCL). We thank the authors for sharing their codes.
