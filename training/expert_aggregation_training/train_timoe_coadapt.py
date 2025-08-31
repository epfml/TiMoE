from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import get_fineweb_edu_100BT_dataset
from tqdm import tqdm
import argparse
import os

def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def main(lr):
    rank, world_size, local_rank = setup_ddp()

    dataset, min_date, max_date = get_fineweb_edu_100BT_dataset(nb_points=1000000)
    train_dataset = dataset["train"]
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(train_dataset, batch_size=1, sampler=sampler)

    model_name = "robinfaro/molm_log_prob_router"
    save_path = "/mloscratch/homes/faro/thesis/time-moe/outputs/checkpoints/checkpoints/molm_router_advanced_log_probs_1B"

    if rank == 0:
        print("Loading model and tokenizer...")

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # turn on all gradients
    for param in model.parameters():
        param.requires_grad = True

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    expert_optimizers = {}

    router_params = list(model.module.router.parameters())

    for bucket in range(1, 7):
        expert_params = [
            p for n, p in model.named_parameters()
            if f"experts.{bucket}" in n
        ]
        expert_optimizers[bucket] = torch.optim.AdamW(
            router_params + expert_params,
            lr=lr
        )
    losses = []
    dates_count = {i: 0 for i in range(1, 7)}

    for i, batch in enumerate(tqdm(dataloader, desc=f"[Rank {rank}] Training", position=rank)):
        batch["tokens"] = torch.stack(batch["tokens"], dim=1).to(local_rank)
        input_ids = batch["tokens"][:, :-1].clone()
        labels = batch["tokens"][:, 1:].clone()
        date = batch["date"].to(local_rank)

        time_bucket = int(date.long())
        dates_count[time_bucket] += 1
        optimizer = expert_optimizers[time_bucket]

        outputs = model(input_ids, targets=labels, date=date)
        loss_to_log = outputs["loss_to_log"]
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss_to_log)

        if i % 1000 == 0 and rank == 0:
            print(f"[Rank {rank}] Batch {i}, Loss: {loss_to_log}")

    if rank == 0:
        print("Training completed.")
        print("Dates count:", dates_count)

        model.module.save_pretrained(save_path)
        with open(f"{save_path}/losses.txt", "w") as f:
            for loss in losses:
                f.write(f"{loss}\n")

    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the router model with DDP.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    main(args.lr)