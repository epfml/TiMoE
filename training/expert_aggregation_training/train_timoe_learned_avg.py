import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_fineweb_edu_100BT_dataset
from tqdm import tqdm

def setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def main(lr, save_path):
    setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Load dataset and wrap in distributed sampler
    dataset, min_date, max_date = get_fineweb_edu_100BT_dataset(nb_points=200000)
    sampler = DistributedSampler(dataset["train"], num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset["train"], batch_size=4, sampler=sampler)

    if rank == 0:
        print("Loading model and tokenizer...")

    model = AutoModelForCausalLM.from_pretrained("robinfaro/molm_log_prob_router", trust_remote_code=True).to(local_rank)
    tokenizer = AutoTokenizer.from_pretrained("robinfaro/molm_log_prob_router", trust_remote_code=True)

    model.router.train()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.module.router.parameters(), lr=lr)

    losses = []
    for i, batch in enumerate(tqdm(dataloader, disable=rank != 0, desc=f"[Rank {rank}] Training the router")):
        batch["tokens"] = torch.stack(batch["tokens"], dim=1).to(local_rank)
        input_ids = batch["tokens"][:, :-1].clone()
        labels = batch["tokens"][:, 1:].clone()
        date = batch["date"].to(local_rank)

        outputs = model(input_ids, targets=labels, date=date)
        loss_to_log = outputs["loss_to_log"]
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss_to_log)

        if i % 100 == 0 and rank == 0:
            print(f"[Rank {rank}] Batch {i}, Loss: {loss_to_log}")

    # Save only from rank 0
    if rank == 0:
        print("Training complete. Saving model...")
        os.makedirs(save_path, exist_ok=True)
        model.module.save_pretrained(save_path)
        with open(f"{save_path}/losses.txt", "w") as f:
            for loss in losses:
                f.write(f"{loss}\n")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the router model with DDP.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_path", type=str, default="simple_router_model_ddp")
    args = parser.parse_args()

    main(args.lr, args.save_path)