import copy
import os

import clip.clip as clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src import datasets, templates, utils, models
from src.models.evaluation import evaluate, zeroshot_classifier
from src.datasets.common import get_dataloader, maybe_dictionarize

def finetune(args):
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, use_lora=args.lora)
    if args.load is not None:
        utils.torch_load(model, args.load)

    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )
    classnames = dataset.classnames
    templates = dataset.templates
    num_classes = len(classnames)
    train_loader = dataset.train_loader
    pretrained_model, train_preprocess, val_preprocess = clip.load(args.model, jit=False, use_lora=args.lora)

    # prepare template
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params
    if args.train_mode == "text":
        print("[Training mode] Text Encoder")
        visual_params_name = [k for k, v in model.visual.named_parameters()]
        exclude_params_name = visual_params_name + ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]
    elif args.train_mode == "image":
        print("[Training mode] Image Encoder")
        params = model.visual.parameters()
    else:
        assert args.train_mode == "whole"
        print("[Training mode] Both Encoders")
        exclude_params_name = ["logit_scale"]

        if args.lora:
            params = [
                v for k, v in model.named_parameters() if 'lora' in k
            ]
        else:
            params = [
                v for k, v in model.named_parameters() if k not in exclude_params_name
            ]

    total_params_size = sum(p.numel() * p.element_size() for p in params)
    print('The number of Total Trainable Parameters:', sum(p.numel() for p in params))
    print(f"Total Trainable Parameters Memory Size: {total_params_size / 1024 / 1024:.2f} MB")

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count())) 
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # Method         
    if args.train_mode == "text":
        embeddings = zeroshot_classifier(dataset.classnames, dataset.templates, model)

    for iteration in tqdm(range(total_iterations + 1)):
        # evaluation
        if eval_iterations is not None and iteration % eval_iterations == 0:
            evaluate(model.module, args, val_preprocess)

        # training
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        # prepare model
        model.train()
        scheduler(iteration)

        # prepare data
        if args.train_dataset == 'ImageNet':
            try:
                train_batch = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                train_batch = next(data_iter)
            images, labels = train_batch["images"], train_batch["labels"]
        else:
            try:
                images, labels = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()

        # ce loss
        # -- get text embedding --
        if args.train_mode != "text":
            embeddings = model(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())
    
    # compute prototype
    feature_dim = 512  # CLIP ViT-B/16 feature dim is 512
    pretrained_features_sum = torch.zeros(num_classes, feature_dim).cuda()
    sample_counts = torch.zeros(num_classes, dtype=torch.int).cuda()
    
    print("Computing image features for all classes...")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch_data = maybe_dictionarize(batch)
            images = batch_data['images'].cuda()
            labels = batch_data['labels'].cuda()
            
            pretrained_image_features = pretrained_model(images, None)
            pretrained_image_features /= pretrained_image_features.norm(dim=-1, keepdim=True)
            
            pretrained_features_sum.index_add_(0, labels, pretrained_image_features)
            
            for label in labels:
                sample_counts[label] += 1
        
        print("Computing text features and prototypes...")
        pretrained_proto = []
        
        for class_label, classname in enumerate(classnames):
            texts = [template(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            pretrained_text_features = pretrained_model(None, texts)
            pretrained_text_features /= pretrained_text_features.norm(dim=-1, keepdim=True)
            pretrained_text_feature_avg = pretrained_text_features.mean(dim=0)
            
            if sample_counts[class_label] > 0:
                pretrained_image_feature_avg = pretrained_features_sum[class_label] / sample_counts[class_label]
                pretrained_prototype = (pretrained_image_feature_avg + pretrained_text_feature_avg) / 2
            else:
                print(f"Warning: No samples found for class {classname} (label {class_label})")
                pretrained_prototype = pretrained_text_feature_avg
            
            pretrained_prototype /= pretrained_prototype.norm()
            pretrained_proto.append(pretrained_prototype)

    to_save_model = model.module
    return to_save_model, pretrained_proto