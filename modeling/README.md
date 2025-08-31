# 🧩 Modeling – GPT Experts and TiMoE

This folder contains the implementation of **TiMoE (Time-aware Mixture of Experts)** models, compatible with HuggingFace’s `transformers` interface.  
It includes code for single GPT experts, TiMoE aggregation, routing mechanisms, and HuggingFace-friendly configs.

---

## 📂 Structure

```
modeling/
├── gpt.py              # GPT2-based expert backbone (GPTBase)
├── configuration.py    # HuggingFace config for TiMoE (TiMoEConfig)
├── modeling.py         # TiMoE model definition
├── aux_losses.py       # Auxiliary losses (entropy, load balancing, router z-loss)
├── tokenizer.json      # GPT2 tokenizer
├── vocab.json, merges.txt
```

---

## ⚙️ GPT Experts

- Defined in **`gpt.py`** as `GPTBase`.
- Each **expert** is a standalone GPT2-like model, pretrained on a specific **2-year time slice** of the corpus (e.g., 2013–14, 2015–16, ...).
- Experts can be trained independently and later loaded into TiMoE for aggregation.

---

## 🧠 TiMoE Model

Defined in **`modeling.py`** as `TiMoE`, subclassing HuggingFace’s `PreTrainedModel`.

### Core ideas:
- **Aggregation at log-probability level**  
  All active experts produce log-probabilities; TiMoE combines them either with:
  - Equal weights (`use_router=False`)
  - Router-based learned weights (`use_router=True`)

- **Causal Masking by Timestamp**  
  - Each expert is indexed by training window.  
  - For a query with `date`, experts trained on *future data* are masked.  
  - If no `date` is given, all experts are used.

- **Forward pass**  
  ```python
  output = model(input_ids, date=2020)
  print(output.combined_log_probs.shape)  # (batch, seq_len, vocab_size)
  ```

- **Generation**  
  ```python
  text = model.generate_from_string(
      "The pandemic started in", max_new_tokens=50, date=2020
  )
  ```

---

## 📡 Router Mechanism (`use_router=True`)

When enabled, TiMoE learns to **route tokens across experts**:

1. A simple linear layer maps hidden states → router logits.
2. Router logits are masked so only past/current experts are valid.
3. Top-𝑘 experts are selected (`config.top_k_experts`).
4. Softmax-normalized weights are added in **log-prob space**:
   ```python
   combined_log_probs = logsumexp(log_probs + log_weights)
   ```

These are only applied during training with `use_router=True`.

---

## 📑 Config

Defined in **`configuration.py`** (`TiMoEConfig`).

Key parameters:
- `num_experts`: number of experts
- `expert_configs`: list of configs for each GPT expert
- `use_router`: whether to enable router-based aggregation
- `top_k_experts`: how many experts to keep after routing
- `sequence_length`: max context length

Example:
```python
from modeling.configuration import TiMoEConfig
config = TiMoEConfig(
    num_experts=6,
    expert_configs=[...],
    use_router=True,
    top_k_experts=2,
    sequence_length=1024
)
```

---

## 📦 HuggingFace Integration

- `TiMoE` extends `PreTrainedModel`, so it supports `.from_pretrained()` and `.push_to_hub()`.
- `configuration.py` ensures that model cards/configs serialize correctly.
- Additional files (`tokenizer.json`, `vocab.json`, `merges.txt`) allow direct use with HF `AutoTokenizer`.

---

## 🔑 Usage Example

```python
from transformers import AutoTokenizer
from modeling.modeling import TiMoE
from modeling.configuration import TiMoEConfig

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Build config
config = TiMoEConfig(
    num_experts=6,
    expert_configs=[...],  # list of GPT2 configs
    use_router=True,
    top_k_experts=2,
    sequence_length=1024,
)

# Init model
model = TiMoE(config)

# Forward pass with timestamp
inputs = tokenizer("The year is 2019 and", return_tensors="pt")
out = model(**inputs, date=2019)
print(out.combined_log_probs.shape)
```

---

## 📝 Notes

- **Equal-weight mode (`use_router=False`)**  
  All experts up to the query date contribute equally.
- **Router mode (`use_router=True`)**  
  A learned router dynamically selects a sparse subset of experts.
- **Pretrained weights** can be frozen and reused to build different TiMoE variants.
