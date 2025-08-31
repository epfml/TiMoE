# 🛠 Preprocessing

The preprocessing pipeline transforms timestamped text (e.g. from **FineWeb-Edu**) into fixed-length token windows annotated with a **date** (the CommonCrawl dump year). It consists of three main stages plus an optional filtering step.

---

## 0) Requirements

```bash
pip install datasets zarr numcodecs tiktoken numpy tqdm
```

- Tokenization: **GPT-2** tokenizer via `tiktoken` (EOT id = 50256).
- Output: HuggingFace datasets stored on disk.

---

## 1) Tokenize corpus into per-date Zarr arrays

**Script:** `preprocessing/tokens_per_year.py`

- Loads the dataset (`HuggingFaceFW/fineweb-edu`, `sample-100BT`).
- Tokenizes each `text`, appends EOT.
- Groups tokens **by year** (from the `dump` field, e.g. `CC-MAIN-2022 → 2022`).
- Saves each year’s token stream into **Zarr arrays**.
- Supports sharding via `--start/--end` to parallelize the pass over the dataset.

**Example run (sharding by index):**
```bash
# Slice 1
python preprocessing/tokens_per_year.py \
  --output_dir /path/preprocessing_results/fineweb_edu/tokens_per_year/0-35000000 \
  --start 0 --end 35000000

# Slice 2
python preprocessing/tokens_per_year.py \
  --output_dir /path/preprocessing_results/fineweb_edu/tokens_per_year/35000000-70000000 \
  --start 35000000 --end 70000000

# Slice 3
python preprocessing/tokens_per_year.py \
  --output_dir /path/preprocessing_results/fineweb_edu/tokens_per_year/70000000-97270686 \
  --start 70000000
```

**Output layout:**
```
.../tokens_per_year/{START-END}/
├── 2013   # Zarr array of tokens for 2013
├── 2014
└── ...
```

> Implementation notes:
> - Tokens stored as `np.uint16` (GPT-2 vocab < 65536).
> - Multicore tokenization via `datasets.map(..., num_proc=cpu_count())`.

---

## 2) Chunk tokens into fixed windows (parallelized)

**Script:** `preprocessing/generate_dataset_parallel.py`

- Processes **one date at a time** (`--date YYYY`).
- Virtually concatenates all Zarr shards for that date from step 1.
- Splits into non-overlapping windows of length `SEQ_LEN=1025` (stride = 1025).
- Pads the last window with `50256` (GPT-2 EOT).
- Saves HuggingFace `Dataset` batches to disk:
  `SAVE_ROOT/date_{YYYY}/block_{i}/batch_{j}`.
- Parallelism: splits the date stream into `num_cpu` blocks; each block in its own process.

**Update paths inside the script:**
```python
ZARR_PATHS = [
  "/path/preprocessing_results/fineweb_edu/tokens_per_year/0-35000000",
  "/path/preprocessing_results/fineweb_edu/tokens_per_year/35000000-70000000",
  "/path/preprocessing_results/fineweb_edu/tokens_per_year/70000000-97270686",
]
SAVE_ROOT = "/path/preprocessing_results/shards_parallel_fineweb_edu"
```

**Run (per date):**
```bash
python preprocessing/generate_dataset_parallel.py --date 2019 --max-cpu 32
python preprocessing/generate_dataset_parallel.py --date 2020 --max-cpu 32
```

**Output layout:**
```
.../shards_parallel_fineweb_edu/
└── date_2019/
    ├── block_0/
    │   ├── batch_0/  # HuggingFace dataset
    │   └── ...
    ├── block_1/
    └── ...
```

Each example:
```python
{"tokens": List[int] (len=1025), "date": int}
```

> Recommended tweaks:
> - Add a guard for missing dates to fail fast:
>   ```python
>   if not shards: raise ValueError(f"No shards found for date={date}. Check ZARR_PATHS.")
>   ```

---

## 3) Merge all shards into one dataset

**Script:** `preprocessing/merge_all_shards.py`

- Scans `SHARDS_ROOT` for all `batch_*` datasets (across all dates/blocks).
- Loads & concatenates them into a single HuggingFace dataset.
- Casts `tokens → Sequence(int32)`, `date → int16`.
- Saves to `FINAL_DATASET_PATH`.

**Update paths in script:**
```python
SHARDS_ROOT = "/path/preprocessing_results/shards_parallel_fineweb_edu"
FINAL_DATASET_PATH = "/path/data/fineweb_edu_100BT_preprocessed"
```

**Run:**
```bash
python preprocessing/merge_all_shards.py
```

**Output:**
```
/path/data/fineweb_edu_100BT_preprocessed/
```

Usage:
```python
from datasets import load_from_disk
ds = load_from_disk("/path/data/fineweb_edu_100BT_preprocessed")
print(ds[0])  # {"tokens": [...], "date": 2019}
```

---

## 4) (Optional) Filter by single date

**Script:** `preprocessing/filter_by_date.py`

**Run:**
```bash
python preprocessing/filter_by_date.py 2019
```

**Output:**
```
../data/fineweb_edu_100BT_preprocessed_filtered_2019/
```

