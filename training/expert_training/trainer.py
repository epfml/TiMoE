"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Optional, Any, Dict
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import boto3
from urllib.parse import urlparse
import fsspec
import io
import wandb
import torch.distributed as dist
import math
import gc
from torch.utils.data import Sampler

@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    ckpt_folder: Optional[str] = None
    token_ckpt_interval: int = None
    save_every: int = None  # Numbers of MICRO BATCH to save a snapshot
    eval_every: int = None  # Numbers of MICRO BATCH to evaluate the model
    log_every: int = None  # Numbers of MICRO BATCH to log the loss  
    use_amp: bool = None
    gradient_accumulation_steps: int = 2

@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    finished_step: int
    token_seen: int 

def upload_to_s3(obj, dst):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    dst = urlparse(dst, allow_fragments=False)
    boto3.client('s3').upload_fileobj(buffer, dst.netloc, dst.path.lstrip('/'))

class Trainer:

    def __init__(self, trainer_config: TrainerConfig, model, model_config ,optimizer, scheduler, train_dataset, wandb, test_dataset=None):
        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])  
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.step_run = 0
        # initialize train states
        self.epochs_run = 0
        self.step_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer     
        self.scheduler = scheduler   
        self.start_iter = 0
        # set up logging, snapshotting and evaluation
        self.save_every = self.config.save_every
        self.eval_every = self.config.eval_every
        self.log_every = self.config.log_every
        if self.config.use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        #self._load_snapshot()
        # set up checkpointing
        self.ckpt_folder = self.config.ckpt_folder
        self.token_ckpt_interval = self.config.token_ckpt_interval
        if self.ckpt_folder is not None:
            os.makedirs(self.ckpt_folder, exist_ok = True)
        self.next_token_ckpt = self.token_ckpt_interval
        self.token_seen = 0
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])
        self._load_snapshot()
        self.model_config = model_config
        # data stuff
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_train_dataloader(train_dataset)
        self.test_loader = self._prepare_test_dataloader(test_dataset) if test_dataset else None
        self.wandb = wandb
        
    def _prepare_test_dataloader(self, dataset: Dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )
        return dataloader
    

    def _prepare_train_dataloader(self, dataset: Dataset):
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False
        )

        # Compute local number of steps already done
        local_steps_done = self.step_run // self.world_size
        skip_samples = local_steps_done * self.config.batch_size

        # Get full list of indices assigned to this rank
        sampler.set_epoch(0)  # ensure it's initialized
        full_indices = list(sampler)

        if skip_samples >= len(full_indices):
            raise ValueError(f"[GPU {self.global_rank}] skip_samples={skip_samples} exceeds local dataset size={len(full_indices)}")

        # Truncate indices to skip already seen samples
        remaining_indices = full_indices[skip_samples:]

        print(f"[GPU {self.global_rank}] Skipping {skip_samples} samples → using {len(remaining_indices)} samples")

        truncated_dataset = torch.utils.data.Subset(dataset, remaining_indices)

        # Standard DataLoader with no need for custom sampler now
        return DataLoader(
            truncated_dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            num_workers=self.config.data_loader_workers,
            shuffle=False
        )
    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return 

        snapshot = Snapshot(**snapshot_data)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(snapshot.model_state)
        else:
            self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.scheduler.load_state_dict(snapshot.scheduler_state)
        self.step_run = snapshot.finished_step
        self.token_seen = snapshot.token_seen
        # adjust the next token checkpoint to be the next one after the current one
        self.next_token_ckpt = (self.token_seen // self.token_ckpt_interval + 1) * self.token_ckpt_interval
        print(f"Resuming training from snapshot at Step {self.step_run}")


    def _run_batch(self, batch, train: bool = True) -> tuple:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.config.use_amp)):
            input_ids = batch["tokens"][:, :-1].clone()
            date = batch["date"].clone()
            targets = batch["tokens"][:, 1:].clone()
            output = self.model(input_ids, date, targets= targets, get_logits=False, moe=self.model_config.moe)
            return output

    
    def _run_validation(self, dataloader: DataLoader):
        self.model.eval()
        total_tokens = 0
        total_loss_per_token = 0
        total_loss_per_batch= 0
        with torch.no_grad():
            for iter, batch in tqdm(enumerate(dataloader), desc="Validation", total=len(dataloader)):
                batch["tokens"] = torch.stack(batch["tokens"], dim=1).to(f'cuda:{self.local_rank}')
                num_tokens = batch["tokens"].numel() #return the total n of elements in the input tensor
                batch_loss, _ = self._run_batch(batch, train=False)
                total_tokens += num_tokens
                total_loss_per_token += batch_loss * num_tokens #scale loss by tokens
                total_loss_per_batch += batch_loss
        
        avg_loss_per_batch = total_loss_per_batch / len(dataloader)
        avg_nll = total_loss_per_token/total_tokens # average negative log-likelihood per token
        perplexity = math.exp(avg_nll)
        return {
        "loss": avg_loss_per_batch,
        "perplexity": perplexity
    }

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        for iter, batch in tqdm(enumerate(dataloader), desc=f"GPU {self.local_rank} Epoch {epoch}", total=len(dataloader)):
            step_type = "Train" if train else "Eval"
            batch["tokens"] = torch.stack(batch["tokens"], dim=1).to(f'cuda:{self.local_rank}')
            if iter % self.config.gradient_accumulation_steps == 0:
                self.model.require_backward_grad_sync = True
            else:
                self.model.require_backward_grad_sync = False


            output = self._run_batch(batch, train)
            batch_loss = output["loss"] / self.config.gradient_accumulation_steps
            loss_to_log = output["loss_to_log"] / self.config.gradient_accumulation_steps

            if self.config.use_amp:
                self.scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()


            step = ((epoch - 1) * len(dataloader) + iter) * self.world_size
            step += self.step_run
            self.token_seen = step * self.config.batch_size * 1025

            if iter % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp: 
                    self.scaler.unscale_(self.optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.config.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.scaler.update()
                else:
                    norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.config.grad_norm_clip)
                    self.optimizer.step()
                    self.scheduler.step()

                # reset the gradients
                self.optimizer.zero_grad(set_to_none=True)

            if ((iter % self.log_every == 0) or (iter == len(dataloader) - 1)) and self.global_rank == 0:
                print(f"{step_type} Epoch {epoch} Iter {iter} Loss: {loss_to_log * self.config.gradient_accumulation_steps}")
                wandb.log({f"{step_type} Loss": loss_to_log * self.config.gradient_accumulation_steps}, step = step)

                wandb.log({"Learning Rate": self.scheduler.get_last_lr()[0]}, step = step)
                wandb.log({"Tokens Seen": self.token_seen}, step = step)
                if norm is not None:
                    wandb.log({"Gradient Norm": norm}, step = step)
            if ((iter % self.save_every == 0) or (iter == len(dataloader) - 1)) and self.global_rank == 0:

                print(f"Saving snapshot at step {step}")
                self._save_snapshot(step)
            if self.token_seen > self.next_token_ckpt and self.ckpt_folder is not None and self.global_rank == 0:
                print(f"Saving checkpoint at step {step}")
                self._save_token_checkpoint(self.token_seen)
                self.next_token_ckpt += self.token_ckpt_interval
            
    def _save_token_checkpoint(self, token_count):
        with torch.no_grad():
            ckpt_path = f"{self.ckpt_folder}/ckpt_{token_count}.pt"
            model = self.model
            raw_model = model.module if hasattr(model, "module") else model
            checkpoint = {
                "model_state": raw_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "token_count": token_count
            }
            torch.save(checkpoint, ckpt_path)
            del model
            del raw_model
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Checkpoint saved at {ckpt_path}")      

    def _save_snapshot(self, step):
        # capture snapshot
        with torch.no_grad():
            snapshot = Snapshot(
                model_state=self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                finished_step=step,
                token_seen=self.token_seen
            )
            # save snapshot
            snapshot = asdict(snapshot)
            if self.config.snapshot_path.startswith("s3://"):
                upload_to_s3(snapshot, self.config.snapshot_path)
            else:
                torch.save(snapshot, self.config.snapshot_path)
                del snapshot
                torch.cuda.empty_cache()
                gc.collect()
                
            print(f"Snapshot saved at step {step}")

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            self._run_epoch(epoch, self.train_loader, train=True)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)