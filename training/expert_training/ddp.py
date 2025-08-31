from utils import *
from multiprocessing import cpu_count
from modeling.configuration import TiMoEConfig as Config
from modeling.modeling import GPTBase
from tqdm import tqdm
import datasets
import utils
import torch
import wandb
import json
import os
from trainer import Trainer, TrainerConfig
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
import argparse


SEED = 42

available_datasets = {
    "fineweb_edu_100BT": get_fineweb_edu_100BT_dataset,
}

moe_routings_run_names = {
    "None": "GPT2",
    "masked": "time_dependent",
    "standard_gating": "standard_gating_final"
}

def main(moe_routing, nb_points, batch_size, n_embed, n_head, n_layer, get_dataset_function, lr, suffix, save_every, eval_every, log_every, token_ckpt_interval, amp, shared_attention, gradient_acc_steps, date):

    """
    Main function to train and evaluate multiple model variants using data parallelism.

    Parameters:
    :param moe_routing: str
        Type of routing mechanism to use in the model.
    :param nb_points: int
        Total number of data points to generate or load for training and evaluation.
    :param batch_size: int
        Number of samples per batch used during training.
    :param n_embed: int
        Dimensionality of the embedding space used by the model.
    :param n_head: int
        Number of attention heads in the multi-head self-attention mechanism.
    :param n_layer: int
        Number of transformer layers in the model.
    :param get_dataset_function: callable
        A function that returns the dataset in a dictionary format containing 'train' and 'test' splits,
        along with the min and max date values for time-aware expert setup.
    :param lr: float, optional
        Learning rate used by the optimizer during training.
    :param suffix: str, optional
        Optional suffix to append to the run name for tracking different experiments.
    :param save_every: int, optional
        Number of micro-batches to save a snapshot of the model.
    :param eval_every: int, optional
        Number of micro-batches to evaluate the model.
    :param log_every: int, optional
        Number of micro-batches to log the loss.
    :param token_ckpt_interval: float, optional
        Fraction of total tokens to save an independent checkpoint of the model. E.g., 0.1 means every 10% of the total tokens.
    :param amp: bool, optional
        Use automatic mixed precision (AMP) for training.
    :param shared_attention: bool, optional
        Use shared attention across experts.
    :param gradient_acc_steps: int, optional
        Number of gradient accumulation steps.
    """
    print(f"Running with {moe_routing} routing")
    utils.ddp_setup()
    # define the types of models to run
    run_name = moe_routings_run_names[moe_routing] + suffix
    if moe_routing == "None":
        moe_routing = None

    #load the dataset
    if date is not None:
        fineweb_dataset, min_date, max_date = get_dataset_function(nb_points, date)
    else:
        fineweb_dataset, min_date, max_date = get_dataset_function(nb_points)

    train_dataset = fineweb_dataset["train"]
    test_dataset = fineweb_dataset["test"]

    total_tokens = 1025 * len(train_dataset)  # since each sample has 1025 tokens

    if os.environ["RANK"] == "0":
        print(f"Total number of tokens in the training dataset: {total_tokens}")
    
    token_ckpt_interval = int(math.floor(total_tokens * token_ckpt_interval))  # convert to number of tokens


    #for moe_routing, run_name in zip(moe_routings, run_names):

    # define the configuration
    model_config = Config(**{
        "moe_num_experts": (max_date - min_date) // 2 + 1,
        "moe_softmax_order": "softmax_topk",
        "batch_size": batch_size,
        "n_embd": n_embed,
        "n_head": n_head,
        "n_layer": n_layer,
        "moe_routing": moe_routing,
        "moe": moe_routing is not None,
        "shared_attention": shared_attention,
    })


    if os.environ["RANK"] == "0":
        # Initialize wandb
        wandb_id_path = f"wandb/wandb_id_{run_name}.txt"

        if os.path.exists(wandb_id_path):
            with open(wandb_id_path) as f:
                run_id = f.read().strip()
            resume_mode = "must"
        else:
            run_id = wandb.util.generate_id()
            with open(wandb_id_path, "w") as f:
                f.write(run_id)
            resume_mode = "allow"  # oppure "never" se vuoi creare sempre nuovi run

        wandb.init(
            entity="time-moe",
            project="pretraining",
            name=run_name,
            id=run_id,
            resume=resume_mode,
            config={
                "moe_num_experts": (max_date - min_date) // 2 + 1,
                "moe_softmax_order": "softmax_topk",
                "batch_size": batch_size,
                "n_head": n_head,
                "n_layer": n_layer,
                "n_embd": n_embed,
                "moe_routing": moe_routing,
                "moe": moe_routing is not None,
                "shared_attention": shared_attention,
            }
        )

    # Instantiate the model
    moe = GPTBase(model_config)

    # Initialize the weights
    moe.apply(utils.initialize_weights)
    moe.train()

    total_params = sum(p.numel() for p in moe.parameters())
    num_gpus = int(os.environ["WORLD_SIZE"])
    if os.environ["RANK"] == "0":
        print(f"Total number of parameters: {total_params}")


    total_steps = math.floor(len(train_dataset) // batch_size / num_gpus / gradient_acc_steps) #total number of training steps
    #warmup, stable, and decay steps
    W = int(0.05*total_steps) #taking 5% as miniCPM paper does not specify the warmup % like for decay 
    #(wen et al, 2024 just mentions its the standard short phase)
    T = W + int(0.85*total_steps) #T is the end step of the stable phase
    S = total_steps #S is the total number of training steps

    warmup_steps = W #end step of the warmup phase
    stable_steps = T-W
    decay_steps = S-T  #10% of total for decay according to the Hu et al. MiniCPM paper

    optimizer = torch.optim.AdamW(moe.parameters(), lr=lr, weight_decay=0.1)

    scheduler = get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_stable_steps=stable_steps,
        num_decay_steps=decay_steps,
        warmup_type="linear",     # because in the MiniCPM paper we have s/W
        decay_type="cosine",      # one possible implementation of f(s-T) from the miniCPM paper
        min_lr_ratio=0.1,         # lowest learning rate will be 10% of the max
        num_cycles=0.5            # controls the cosine decay behaviour
    )
    
    trainer = Trainer(
        trainer_config=TrainerConfig(
            max_epochs=1,
            batch_size=batch_size,
            save_every=save_every,
            use_amp=amp,
            data_loader_workers=2,
            snapshot_path=f"outputs/snapshots/snapshot_{run_name}.pt",
            ckpt_folder=f"outputs/checkpoints/checkpoints_{run_name}/",
            token_ckpt_interval=token_ckpt_interval,
            eval_every=eval_every,
            log_every=log_every,
            grad_norm_clip=1.0,
            gradient_accumulation_steps=gradient_acc_steps,
        ),
        model=moe,
        model_config=model_config,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        wandb=wandb
    )

    trainer.train()

    if os.environ["RANK"] == "0":
        wandb.finish()

    del moe
    del trainer
    torch.cuda.empty_cache()

    destroy_process_group()


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--moe_routing", type=str, default="None", choices=moe_routings_run_names.keys(), help="Type of routing mechanism to use in the model.")
    parser.add_argument("--nb_points", type=int, default=-1, help="Number of data points to generate or load for training and evaluation. If -1, the entire dataset is used.")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_embed", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--dataset", type=str, required=True, choices=available_datasets.keys(),
                    help="Dataset loader: fineweb_tokenize (raw data to be tokenized and preprocessed), fineweb_edu (preprocessed 10B tokens), fineweb_100BT (preprocessed 100B tokens)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--token_ckpt_interval", type=float, default=1e-1, help="Fraction of total tokens to save an independent checkpoint of the model. E.g., 0.1 means every 10% of the total tokens.")
    parser.add_argument("--amp", type = bool, help="Use automatic mixed precision (AMP) for training.", default=True)
    parser.add_argument("--shared_attention", action="store_true", help="Use shared attention across experts.")
    parser.add_argument("--gradient_acc_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--date", type=int, default=None, help="The year of the date to filter the dataset by (as an integer). If None, the entire dataset is used.")
    args = parser.parse_args()
    get_dataset_function = available_datasets[args.dataset]
    main(args.moe_routing, args.nb_points, args.batch_size, args.n_embed, args.n_head, args.n_layer, get_dataset_function, args.lr, args.suffix, args.save_every, args.eval_every, args.log_every, args.token_ckpt_interval, args.amp, args.shared_attention, args.gradient_acc_steps, args.date)

