import copy
import os
from random import random
import clip
import torch

from . import utils
from .args import parse_arguments
from .models import evaluate, finetune, few_shot_finetune, agnostic_evaluate
from .models.modeling import create_image_classifier
from .delta_models import DeltaModel, unify_delta_models, apply_delta_model

def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)

def torch_load(model, save_path, device=None): 
    model = torch.load(save_path, weights_only=False)
    if device is not None:
        model = model.to(device)
    return model

def main(args):
    print(args)
    utils.seed_all(args.seed)

    assert args.train_mode in ["whole", "text", "image"]

    if args.eval:
        model, _, val_preprocess = clip.load(args.model, jit=False, use_lora=args.lora)
        if args.task_agnostic:
            agnostic_evaluate(model, args, val_preprocess)
        else :
            evaluate(model, args, val_preprocess)
    else :
        if args.few_shot > 0 :
            current_model, current_prototype = few_shot_finetune(args)
        else:
            current_model, current_prototype = finetune(args)
        
        # save pretrained model
        model, _, val_preprocess = clip.load(args.model, jit=False, use_lora=args.lora)
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        if not os.path.exists(os.path.join(args.save, "pretrained.pth")):
            torch_save(model, os.path.join(args.save, "pretrained.pth"))

        save_path = os.path.join(args.save, f"session_{args.session}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # decoupling unified model
        delta_models = []
        models_list = []
        task_list = []
        pretrained_model = torch_load(model, os.path.join(args.save, "pretrained.pth"))
        pretrained_model = pretrained_model.to(args.device)
        current_model = current_model.to(args.device)
        if args.session != 0:
            store_path = os.path.join(args.save, f"session_{args.session-1}")
            unified_delta_model = torch.load(os.path.join(store_path, "unified.pth"))

            # load the task triggers
            for file in os.listdir(store_path):
                if file.endswith(".pth") and file != "unified.pth" and file != "prototypes.pth":
                    task_list.append(file.replace(".pth", ""))
                    task_triggers = torch.load(os.path.join(store_path, file))
                    masks = task_triggers["masks"]
                    rescaler = task_triggers["rescalers"]
                    delta_model_recon = {}
                    for n in unified_delta_model:
                        delta_model_recon[n] = unified_delta_model[n] * masks[n] * rescaler
                    image_encoder = apply_delta_model(delta_model_recon, pretrained_model)
                    image_encoder = image_encoder.to(args.device)
                    models_list.append(image_encoder)
                    delta_models.append(DeltaModel(pretrained_model, image_encoder))

        # add the current model
        task_list.append(args.train_dataset)
        models_list.append(current_model)
        delta_models.append(DeltaModel(pretrained_model, current_model))

        # unifying the model
        delta_model_unified, masks, rescalers = unify_delta_models(delta_models, args.device)
        torch.save(delta_model_unified, os.path.join(save_path, "unified.pth"))
        for i, dataset in enumerate(task_list):
            save_file_path = os.path.join(save_path, f"{dataset}.pth")
            task_masks = {}
            for n in masks:
                task_masks[n] = masks[n][i]
                
            torch.save({
                "masks": task_masks,
                "rescalers": rescalers[i]
            }, save_file_path)

        # load the prototypes
        if args.session != 0:
            store_path = os.path.join(args.save, f"session_{args.session-1}")
            prototype_sets = torch.load(os.path.join(store_path, "prototypes.pth"))
        else:
            prototype_sets = {}
        prototype_sets[args.train_dataset] = current_prototype
        torch.save(prototype_sets, os.path.join(save_path, "prototypes.pth"))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
