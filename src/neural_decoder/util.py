from torch.nn import Linear
from torch import nn
from transformers.activations import ACT2FN
import torch
from typing import Literal

ACTIVATION_FUNCTION = Literal[
    "gelu",
    "gelu_10",
    "gelu_fast",
    "gelu_new",
    "gelu_python",
    "gelu_pytorch_tanh",
    "gelu_accurate",
    "laplace",
    "linear",
    "mish",
    "quick_gelu",
    "relu",
    "relu2",
    "relu6",
    "sigmoid",
    "silu",
    "swish",
    "tanh",
]


def create_fully_connected(
    input_size: int,
    output_size: int,
    hidden_sizes=[],
    activation: ACTIVATION_FUNCTION = "gelu",
):
    classifier_layers = []
    for i in range(-1, len(hidden_sizes)):
        is_last = i + 1 == len(hidden_sizes)
        is_first = i == -1
        in_size = input_size if is_first else hidden_sizes[i]
        out_size = output_size if is_last else hidden_sizes[i + 1]
        classifier_layers.append(Linear(in_size, out_size))
        if not is_last:
            classifier_layers.append(ACT2FN[activation])
    return nn.Sequential(*classifier_layers)


def calc_seq_len(index_seq: torch.Tensor):
    for i in range(len(index_seq)):
        j = len(index_seq) - 1 - i
        if index_seq[j].item() > 0:
            return j + 1
    return 0


def compute_ctc_loss(
    model_input: torch.Tensor,
    out_log_softmaxed_batch: torch.Tensor,
    targets: torch.Tensor,
    loss: nn.CTCLoss,
):
    """
    x: (batch_size, seq_len, *) - assuming all items of last dimension to be zero when padded
    out_log_softmaxed_batch: (batch_size, seq_len, vocab_size)
    """
    device = targets.device

    target_lens = torch.tensor([calc_seq_len(seq) for seq in targets])
    # out shape: (batch_size, seq_len, vocab_size)

    # non padded mask
    mask = model_input != 0
    # seq lens without padding
    # mask shape: (batch_size, seq_len, 256)
    in_seq_lens = mask.any(-1)
    # in_seq_lens shape: (batch_size, seq_len)
    in_seq_lens = in_seq_lens.sum(-1)
    # in_seq_lens shape: (batch_size)
    in_seq_lens = in_seq_lens.clamp(max=out_log_softmaxed_batch.shape[1])
    out = out_log_softmaxed_batch.transpose(0, 1)
    # out shape: (seq_len, batch_size, vocab_size)
    ctc_loss = loss(
        out,
        targets,
        in_seq_lens.to(device),
        target_lens.to(device),
    )
    if ctc_loss.item() < 0:
        print(
            f"\nWarning: loss is negative, this might be due to prediction lens ({in_seq_lens.tolist()}) being smaller than target lens {target_lens.tolist()}\n"
        )
    return ctc_loss


import os
import json


class Config:
    def __init__(self):
        if not os.path.exists("config.json"):
            raise Exception(
                "config.json not found. Please create a config.json as described in the README.md"
            )
        with open("config.json") as f:
            config = json.load(f)
            self.dataset_path = config["datasetPath"]
            self.lm_3gram_dir = config["lm_3gram_dir"]
            self.lm_5gram_dir = config["lm_5gram_dir"]
            self.output_dir = config["outputDir"]
            self.cache_dir = config["cacheDir"]
