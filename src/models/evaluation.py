import clip

import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

import src.datasets as datasets
from src.datasets.common import get_dataloader, maybe_dictionarize
import src.delta_models as delta_models
from src.delta_models import DeltaModel, apply_delta_model, unify_delta_models

def torch_save(model, save_path):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)

def torch_load(model, save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@torch.no_grad()
def zeroshot_classifier(classnames, templates, model):
    if not isinstance(templates, list):
        templates = [templates]
    zeroshot_weights = []
    for classname in classnames:
        texts = [template(classname) for template in templates]  # format with class
        texts = clip.tokenize(texts).cuda()  # tokenize
        class_embeddings = model.encode_text(texts)  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def zeroshot_eval(model, loader, zeroshot_weights):
    top1, top5, n = 0.0, 0.0, 0.0
    for i, data in enumerate(tqdm(loader)):

        data = maybe_dictionarize(data)
        images = data["images"].cuda()
        target = data["labels"].cuda()

        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100
    return top1, top5


def eval_single_dataset(image_classifier, dataset, args):
    model = image_classifier
    input_key = "images"
    image_enc = None

    model.eval()

    zeroshot_weights = zeroshot_classifier(
        dataset.classnames, dataset.templates, model
    )

    dataloader = get_dataloader(
        dataset, is_train=False, args=args, image_encoder=image_enc
    )

    top1, top5 = zeroshot_eval(model, dataloader, zeroshot_weights)

    print(f"Top-1 accuracy: {top1:.2f}") 

def evaluate(image_classifier, args, val_preprocess):
    if args.eval_datasets is None:
        return
    model = image_classifier
    current_models = {}
    pretrained_model = torch_load(model, os.path.join(args.save, "pretrained.pth"))
    pretrained_model = pretrained_model.to(args.device)
    if args.session != 0:
        store_path = os.path.join(args.save, f"{args.session}")
        unified_delta_model = troch.load(os.path.join(store_path, "unified.pth"))

        # load the task triggers
        for file in os.listdir(store_path):
            if file.endswith(".pth") and file != "unified.pth" and file != "prototypes.pth":
                task_name = file.replace(".pth", "")
                task_triggers = torch.load(os.path.join(store_path, file))
                masks = task_triggers["masks"]
                rescaler = task_triggers["rescalers"]
                delta_model_recon = {}
                for n in unified_delta_model:
                    delta_model_recon[n] = unified_delta_model[n] * masks[n] * rescaler
                image_encoder = apply_delta_model(delta_model_recon, pretrained_model)
                image_encoder = image_encoder.to(args.device)
                current_models[task_name] = image_encoder

    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )
        image_classifier = current_models.get(dataset_name, pretrained_model)
        image_classifier = image_classifier.to(args.device)
        eval_single_dataset(image_classifier, dataset, args)

def agnostic_evaluate(image_classifier, args, val_preprocess):
    # decoupling unified model
    logits_num = args.logits_num
    model = image_classifier
    current_models = []
    task_list = []
    pretrained_proto_list = []
    pretrained_model = torch_load(model, os.path.join(args.save, "pretrained.pth"))
    pretrained_model = pretrained_model.to(args.device)
    store_path = os.path.join(args.save, f"session_{args.session}")
    unified_delta_model = torch.load(os.path.join(store_path, "unified.pth"))
    prototype_sets = torch.load(os.path.join(store_path, "prototypes.pth"))

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
            current_models.append(image_encoder)
            pretrained_proto_list.append(prototype_sets[file.replace(".pth", "")])
    

    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        target_dataset = dataset_class(
            val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            batch_size_eval=args.batch_size_eval,
        )

        current_zero_weights_list = [
            zeroshot_classifier(target_dataset.classnames, target_dataset.templates, pretrained_model)
        ]
        for i, current_model in enumerate(current_models):
            current_zero_weights_list.append(
                zeroshot_classifier(target_dataset.classnames, target_dataset.templates, current_model)
            )
        
        test_loader = DataLoader(
            target_dataset.test_dataset, 
            batch_size=args.batch_size_eval,
            shuffle=False, 
            num_workers=8
        )
        
        logits_acc = 0.0
        n_samples = 0
        
        with torch.no_grad():
            all_protos = [torch.stack(proto_set, dim=0).cuda() for proto_set in pretrained_proto_list]

            #print(task_list)
            #print(len(pretrained_proto_list))
            #assert 1==2
            
            for batch in tqdm(test_loader):
                batch_data = maybe_dictionarize(batch)
                images = batch_data['images'].cuda()
                labels = batch_data['labels'].cuda()
                batch_size = images.size(0)
                n_samples += batch_size
                
                pretrained_image_features = pretrained_model.encode_image(images)
                pretrained_image_features /= pretrained_image_features.norm(dim=-1, keepdim=True)
                
                session_sims = []
                for session_protos in all_protos:
                    similarities = torch.nn.functional.cosine_similarity(
                        pretrained_image_features.unsqueeze(1),  # (B, 1, dim)
                        session_protos.unsqueeze(0),             # (1, num_protos, dim)
                        dim=2
                    )
                    session_sims.append(similarities.max(dim=1)[0])
                session_sims = torch.stack(session_sims, dim=1)  # (B, num_sessions)
                
                sorted_sessions = torch.argsort(session_sims, dim=1, descending=True)
                model_order = torch.cat([
                    torch.zeros((batch_size, 1), dtype=torch.long, device=images.device),
                    sorted_sessions + 1
                ], dim=1)  # (B, num_sessions+1)
            
                logits_list = []
                for model_idx, model in enumerate([pretrained_model] + current_models):
                    zero_weights = current_zero_weights_list[model_idx]
                    image_features = model.encode_image(images)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100.0 * image_features @ zero_weights
                    logits_list.append(logits)
                
                logits_all = torch.stack(logits_list)    # (M, B, C)
                batch_idx = torch.arange(batch_size, device=images.device)  # (B,)
                
                k_logits = min(logits_num + 1, model_order.shape[1])  # 动态调整k_logits
                model_indices = model_order[:, :k_logits]  # (B, k)
                row_idx = batch_idx.view(-1, 1).expand(-1, k_logits)  # (B, k)
                selected_logits = logits_all[model_indices, row_idx]   # (B, k, C)
                cum_logits = selected_logits.mean(dim=1)  # (B, C)
                
                acc1, _ = accuracy(cum_logits, labels, topk=(1,5))
                logits_acc += acc1

        logits_acc_avg = (logits_acc / n_samples) * 100
        print(f"Top-1 accuracy: {logits_acc_avg:.2f}")