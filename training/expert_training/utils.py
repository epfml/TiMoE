from datasets import load_dataset, Dataset
from multiprocessing import cpu_count
import torch.nn.init as init
import multiprocessing
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import tiktoken
import random
import torch
import json
import os
from torch.distributed import init_process_group
import torch
import math
from torch.optim import Optimizer
from typing import Optional
import warnings
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

SEED = 42
random.seed(SEED)
print("RANDOM SEED SET TO", SEED)
SEQUENCE_LENGTH = 1024
num_proc = max(4, cpu_count())


def get_fineweb_edu_100BT_dataset(nb_points, date=None):
    """
    Load the fineweb dataset already tokenized. The dates are processed to be between 1 and 6

    Args:
        nb_points (int): the number of points to load
        num_proc (int): the number of processes to use
    
    Returns:
        tuple: the train and test datasets, the minimum and maximum dates in the dataset
    """
    
    if date is not None:
        bucket = int((date - 2013) // 2 + 1)
        dataset_path = f"../data/fineweb_edu_100BT_preprocessed_filtered_{bucket}"
        min_date = date
        max_date = date
    else:
        dataset_path = "../data/fineweb_edu_100BT_preprocessed_"
        min_date = 2013
        max_date = 2024
    dataset = Dataset.load_from_disk(dataset_path)
    dataset = dataset.shuffle(seed=SEED)

    if nb_points == -1:
        nb_points = len(dataset)

    dataset = dataset.select(range(nb_points))

    split_dataset = dataset.train_test_split(test_size=0.0001, seed=SEED, shuffle=True)

    return split_dataset, min_date, max_date


def save_checkpoint(model, opt, scheduler, itr, ckpt_path, **extra_args):
    """Save the model checkpoint

    Args:
        model (nn.Module): the model to save    
        opt (torch.optim.Optimizer): the optimizer to save
        scheduler (torch.optim.lr_scheduler): the scheduler to save
        itr (int): the current iteration
        ckpt_path (str): the path to save the checkpoint to
        **extra_args: any extra arguments to save
    """
    checkpoint = dict({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "itr": itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)


def print_model_architecture(model):
    """ Print the model architecture and the number of parameters in each block
    
    Args:
        model (nn.Module): the model to print the architecture of
    """
    print("Model architecture:\n")
    print(model)

    print("\n\nNumber of parameters in the model:\n")
    total_params = 0
    indent_level = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params = param.numel()
            block_hierarchy = name.split('.')[:-1]
            while len(block_hierarchy) < indent_level:
                indent_level -= 1
            while len(block_hierarchy) > indent_level:
                print(f"{'    ' * indent_level}Block: {block_hierarchy[indent_level]}")
                indent_level += 1
            print(f"{'    ' * indent_level}Layer: {name} --> Number of parameters: {layer_params}")
            total_params += layer_params
    print(f"\nTotal number of parameters: {total_params}\n")

def _get_wsd_scheduler_lambda(
    current_step: int, #determines which phase of the schedule we are in (warmup, stable, or decay)
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int, #these steps are calculated below according to the miniCPM paper
    warmup_type: str, #linear according to miniCPM (but cosine,1-sqrt are also defined as per huggingface repo)
    decay_type: str, #linear cosine or 1-sqrt, all correspond to miniCPM
    min_lr_ratio: float, #the min learning rate as a ratio of the max rate (ensures eta never goes to zero)
    num_cycles: float, #used for cos decay, controls the number of cosine waves in the decay period
):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        if warmup_type == "linear":
            factor = progress #equivalent to s/W corresponding to the miniCPM paper
        elif warmup_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif warmup_type == "1-sqrt":
            factor = 1.0 - math.sqrt(1.0 - progress)
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio #scales the factor to be between min_lr_ratio and 1.0
        return max(0.0, factor) #safety check to guarantee the factor is never negative

    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0 #keep learning rate at max value during the stable phase (after warmup before decay)

    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        if decay_type == "linear":
            factor = 1.0 - progress
        elif decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        elif decay_type == "1-sqrt":
            factor = 1.0 - math.sqrt(progress)
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)
    return min_lr_ratio

def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: Optional[int] = None,
    num_stable_steps: Optional[int] = None,
    warmup_type: str = "linear",
    decay_type: str = "cosine",
    min_lr_ratio: float = 0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that has three stages:
    1. warmup: increase from min_lr_ratio times the initial learning rate to the initial learning rate following a warmup_type.
    2. stable: constant learning rate.
    3. decay: decrease from the initial learning rate to min_lr_ratio times the initial learning rate following a decay_type.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_decay_steps (`int`):
            The number of steps for the decay phase.
        num_training_steps (`int`, *optional*):
            The total number of training steps. This is the sum of the warmup, stable and decay steps. If `num_stable_steps` is not provided, the stable phase will be `num_training_steps - num_warmup_steps - num_decay_steps`.
        num_stable_steps (`int`, *optional*):
            The number of steps for the stable phase. Please ensure that `num_warmup_steps + num_stable_steps + num_decay_steps` equals `num_training_steps`, otherwise the other steps will default to the minimum learning rate.
        warmup_type (`str`, *optional*, defaults to "linear"):
            The type of warmup to use. Can be 'linear', 'cosine' or '1-sqrt'.
        decay_type (`str`, *optional*, defaults to "cosine"):
            The type of decay to use. Can be 'linear', 'cosine' or '1-sqrt'.
        min_lr_ratio (`float`, *optional*, defaults to 0):
            The minimum learning rate as a ratio of the initial learning rate.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if num_training_steps is None and num_stable_steps is None:
        raise ValueError("Either num_training_steps or num_stable_steps must be specified.")

    if num_training_steps is not None and num_stable_steps is not None:
        warnings.warn("Both num_training_steps and num_stable_steps are specified. num_stable_steps will be used.")

    if warmup_type not in ["linear", "cosine", "1-sqrt"]:
        raise ValueError(f"Unknown warmup type: {warmup_type}, expected 'linear', 'cosine' or '1-sqrt'")

    if decay_type not in ["linear", "cosine", "1-sqrt"]:
        raise ValueError(f"Unknown decay type: {decay_type}, expected 'linear', 'cosine' or '1-sqrt'")

    if num_stable_steps is None:
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps

    lr_lambda = partial(
        _get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        warmup_type=warmup_type,
        decay_type=decay_type,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch) #lambdaLR is a lrs by pytorch that allows to create custom lrs w lambda func


def initialize_weights(module):
    """Initialize the weights of the model

    Args:
        module (nn.Module): The module to initialize the weights of
    """
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0, std=0.01)
    elif isinstance(module, nn.Conv2d):
        init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        init.ones_(module.weight)
        init.zeros_(module.bias)

def move_to_device(dataset, device):
    """Move the dataset tensors to the device
    
    Args:
        dataset (Dataset): the dataset to move to the device
        device (str): the device to move the dataset to
    Returns:
        Dataset: the dataset with the tensors moved to the device
    """
    fineweb_dataset = dataset.map(
        lambda examples: {"tokens": torch.tensor(examples["tokens"]).to(device),
                      "date": torch.tensor(examples["date"]).to(device)},
        batched=True,
        batch_size=10000
    )
    return fineweb_dataset

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))