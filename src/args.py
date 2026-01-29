import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()

    # hyper parameters
    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-size-eval", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--warmup_length", type=int, default=100)
    parser.add_argument("--beta2", type=float, default=0.999)

    # logging setting
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--loss-interval", type=int, default=1000)
    parser.add_argument("--eval-every-epoch", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--session", type=int, default=None)

    # exp setting
    parser.add_argument(
        "--train-mode",
        type=str,
        default="whole",
        choices=["whole", "text", "image"],
        help="Train mode to use.",
    )
    parser.add_argument("--data-location", type=str, default="./data")
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--eval-datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--text-datasets", default=None, type=lambda x: x.split(","))
    parser.add_argument("--template", type=str, default=None)

    # save & load
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--load", type=str, default=None)

    # Model freeze
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help="Whether or not to freeze the image encoder. Only relevant for fine-tuning.",
    )
    parser.add_argument(
        "--freeze-fc",
        type=int,
        default=0,
        help="Whether or not to freeze the fully connection layers. Only relevant for fine-tuning.",
    )

    # few-shot
    parser.add_argument("--few_shot", type=int, default=0)

    # LoRA
    parser.add_argument("--lora", type=bool, default=False)

    # task agnostic
    parser.add_argument(
        "--task_agnostic",
        action="store_true",
        help="Whether or not to use task agnostic training.",
    )
    parser.add_argument(
        "--logits_num",
        type=int,
        default=4,
        help="Number of logits to use for task agnostic training.",
    )

    # temperature
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (
        args.epochs is None or args.iterations is None
    ), "Cannot specify both epoch and iterations."
    assert (
        args.eval_interval is None or not args.eval_every_epoch
    ), "Cannot specify both eval_interval and eval_every_epoch."

    return args
