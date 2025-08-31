# ⏳ TiMoE: Time-Aware Mixture of Language Experts

Official codebase for the COLM 2025 XTempLLMs workshop paper:  
📄 *TiMoE: Time-Aware Mixture of Language Experts*  
by Robin Faro, Dongyang Fan, Tamar Alphaidze, and Martin Jaggi  
**EPFL, Switzerland**

---

## 🌐 Overview

**TiMoE** is a modular framework for building *temporally grounded language models*.  
Instead of conflating all web data in one LLM, we:

- Train **disjoint GPT experts** on non-overlapping 2-year slices.  
- At inference, route queries only to **eligible experts** (up to query timestamp).  
- Aggregate outputs at **log-probability level** to enforce causal validity and prevent future leakage.  

Aggregation strategies:
- **TiMoE-Avg** – equal weights (no training needed)  
- **TiMoE-LearnedAvg** – trainable router, experts frozen  
- **TiMoE-CoAdapt** – router + active expert jointly trained  

---

## 📂 Repository Structure

```
timoe/
├── preprocessing/       # Dataset filtering, tokenization, windowing
│   └── README.md
├── modeling/            # GPT experts, TiMoE model, router
│   └── README.md
├── training/            # Expert training + aggregator training
│   └── README.md
```

---

## 🚀 Quickstart

### 1. Install
```bash
git clone https://github.com/robinfaro/timoe.git
cd timoe
pip install -r requirements.txt
```

### 2. Preprocess Data
See [preprocessing/README.md](./preprocessing/README.md).

### 3. Train Experts
```bash
torchrun --nproc_per_node=8 training/expert_training/ddp.py \
  --dataset fineweb_edu_100BT --date 2017
```

### 4. Train Aggregator
```bash
torchrun --nproc_per_node=8 training/expert_aggregation_training/train_timoe_learned_avg.py \
  --lr 1e-5
```

---

## 📦 Resources

- **TSQA dataset**: [HuggingFace](https://huggingface.co/datasets/anonymous-789/TSQA)  

---

## 📜 Citation

```bibtex
@misc{faro2025timoetimeawaremixturelanguage,
      title={TiMoE: Time-Aware Mixture of Language Experts}, 
      author={Robin Faro and Dongyang Fan and Tamar Alphaidze and Martin Jaggi},
      year={2025},
      eprint={2508.08827},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.08827}, 
}
```

---

## 🤝 Acknowledgements

This project is supported by the Swiss National Science Foundation (SNSF).  
We thank the COLM 2025 reviewers for their valuable feedback.
