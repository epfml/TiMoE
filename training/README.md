# 🎓 Training

This folder contains scripts for training **TiMoE** models in two stages:

1. **Expert Training** – train single GPT experts independently on their assigned time slices.
2. **Expert Aggregation Training** – train the router/aggregator on top of pretrained experts.

---

## 📂 Structure

```
training/
├── expert_training/
│   ├── ddp.py          # Multi-GPU expert training launcher
│   └── trainer.py      # Core training loop (checkpointing, logging, AMP, etc.)
│
└── expert_aggregation_training/
    ├── train_timoe_learned_avg.py   # Train router (experts frozen)
    └── train_timoe_coadapt.py       # Train router + active expert jointly
```

---

## 🧑‍🏫 1. Expert Training

Experts are trained **independently** on 2-year slices of the dataset.

**Main script:** `expert_training/ddp.py`

### Key features
- Uses **PyTorch DistributedDataParallel (DDP)** for multi-GPU scaling.
- Dataset provided via `utils.get_fineweb_edu_100BT_dataset` (with optional date filtering).
- Supports **automatic mixed precision (AMP)**.
- Integrated with **Weights & Biases** (`wandb`) for logging and checkpoint resuming.
- Scheduler implements **warmup–stable–decay** strategy inspired by MiniCPM.

### Usage
```bash
torchrun --nproc_per_node=8 training/expert_training/ddp.py \
  --dataset fineweb_edu_100BT \
  --moe_routing None \
  --batch_size 16 \
  --n_embed 768 \
  --n_head 12 \
  --n_layer 12 \
  --lr 1e-4 \
  --save_every 100 \
  --eval_every 100 \
  --log_every 10 \
  --token_ckpt_interval 0.1 \
  --amp True \
  --gradient_acc_steps 2 \
  --date 2017
```

This trains a **single GPT expert** (`GPTBase`) on documents dated 2017, saving snapshots and checkpoints periodically.

---

## 🔗 2. Expert Aggregation Training

Once experts are trained and frozen, the aggregator (`TiMoE`) can be trained to combine their predictions.

The general aggregation rule is:

\[
\log P(x_{t+1}\mid \mathbf{x}_{1:t}) =
\log\left(\sum_{E_k \in \mathcal{E}(t_q)} w_k(x)\,
  \exp(\log P_k(x_{t+1}\mid \mathbf{x}_{1:t}))\right)
\]

- \( \mathcal{E}(t_q) \) = eligible experts up to query timestamp \( t_q \).  
- \( w_k(x) \) = non-negative weights summing to 1.  

We support three strategies:

### 🟦 TiMoE-Avg (no training required)
\[
w_k(x) = \tfrac{1}{|\mathcal{E}(t_q)|}, \quad \forall E_k \in \mathcal{E}(t_q)
\]

- Equal weights across all experts valid for the query date.
- Simple ensemble, requires no further training.

---

### 🟩 TiMoE-LearnedAvg (train router only)

**Script:** `expert_aggregation_training/train_timoe_learned_avg.py`

- Experts remain **frozen**.
- A small router network is trained to assign adaptive weights \( w_k(x) \) given the input.
- Trains only the router parameters using DDP.

**Usage:**
```bash
torchrun --nproc_per_node=8 training/expert_aggregation_training/train_timoe_learned_avg.py \
  --lr 1e-5 \
  --save_path outputs/router_learnedavg
```

---

### 🟨 TiMoE-CoAdapt (train router + active expert)

**Script:** `expert_aggregation_training/train_timoe_coadapt.py`

- Router and **only the expert corresponding to the training time bucket** are updated.
- All other experts remain frozen.
- This enables **co-adaptation** while preserving temporal validity.

**Usage:**
```bash
torchrun --nproc_per_node=8 training/expert_aggregation_training/train_timoe_coadapt.py \
  --lr 1e-4
```

---

## 📝 Notes

- **Stage 1 (Experts)**: Always train experts first with `ddp.py`.  
- **Stage 2 (Aggregator)**: Choose a strategy:
  - `TiMoE-Avg`: no training needed.
  - `TiMoE-LearnedAvg`: run router training script (experts frozen).
  - `TiMoE-CoAdapt`: run co-adaptation training (router + one expert).  

- Both `train_timoe_learned_avg.py` and `train_timoe_coadapt.py` use DDP and require launching with `torchrun`.

---
