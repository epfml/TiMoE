import os
import zarr
import argparse
import numpy as np
from datasets import Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === CONFIG ===
SEQ_LEN = 1025
STRIDE = 1025
BATCH_SIZE = 100_000
PAD_TOKEN_ID = 50256  # GPT-2 <|endoftext|>
ZARR_PATHS = [
    "../preprocessing_results/fineweb_edu/tokens_per_year/0-35000000",
    "../preprocessing_results/fineweb_edu/tokens_per_year/35000000-70000000",
    "../preprocessing_results/fineweb_edu/tokens_per_year/70000000-97270686",
]
SAVE_ROOT = "../preprocessing_results/shards_parallel_fineweb_edu"

# === FUNZIONI BASE ===
def load_zarr_shards_for_date(date):
    return [zarr.open(path, mode='r')[str(date)] for path in ZARR_PATHS if str(date) in zarr.open(path, mode='r')]

def compute_global_index_map(shards):
    lengths = [shard.shape[0] for shard in shards]
    cumulative = np.cumsum([0] + lengths)
    return cumulative

def get_window_from_global(shards, cum_lens, global_start, seq_len):
    tokens = []
    remaining = seq_len
    global_pos = global_start

    for i in range(len(shards)):
        shard_start = cum_lens[i]
        shard_end = cum_lens[i + 1]
        if global_pos < shard_end:
            local_start = max(global_pos - shard_start, 0)
            local_end = min(local_start + remaining, shards[i].shape[0])
            tokens.extend(shards[i][local_start:local_end])
            remaining -= (local_end - local_start)
            global_pos += (local_end - local_start)
            if remaining == 0:
                break

    # Pad se necessario
    if len(tokens) < seq_len:
        tokens.extend([PAD_TOKEN_ID] * (seq_len - len(tokens)))

    return tokens

# === ELABORAZIONE DI UN BLOCCO ===
def process_block(args):
    date, block_start, block_end, block_id = args
    #print(f"[PID {os.getpid()}] Processing date {date} | block {block_id}")

    out_dir = os.path.join(SAVE_ROOT, f"date_{date}", f"block_{block_id}")
    os.makedirs(out_dir, exist_ok=True)

    if len(os.listdir(out_dir)) > 0:
        print(f"Block {block_id} already processed, skipping.")
        return

    shards = load_zarr_shards_for_date(date)
    cum_lens = compute_global_index_map(shards)
    total_len = cum_lens[-1]

    examples = []
    batch_id = 0

    for i in tqdm(range(block_start, block_end, STRIDE)):
        tokens = get_window_from_global(shards, cum_lens, i, SEQ_LEN)
        examples.append({"tokens": tokens, "date": date})

        if len(examples) == BATCH_SIZE:
            Dataset.from_list(examples).save_to_disk(os.path.join(out_dir, f"batch_{batch_id}"))
            examples = []
            batch_id += 1

    if examples:
        Dataset.from_list(examples).save_to_disk(os.path.join(out_dir, f"batch_{batch_id}"))

    print(f"[PID {os.getpid()}] Done date {date} | block {block_id}")

# === CREAZIONE DEI BLOCCHI ===
def create_blocks(date, num_blocks):
    shards = load_zarr_shards_for_date(date)
    cum_lens = compute_global_index_map(shards)
    total_len = cum_lens[-1]

    block_size = total_len // num_blocks
    blocks = []
    for block_id in range(num_blocks):
        start = block_id * block_size
        end = (block_id + 1) * block_size if block_id < num_blocks - 1 else total_len
        blocks.append((date, start, end, block_id))
    return blocks

# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=int, required=True, help="Date to process")
    parser.add_argument("--max-cpu", type=int, default=cpu_count(), help="Number of parallel processes to use")
    args = parser.parse_args()

    date = args.date
    num_cpu = min(args.max_cpu, cpu_count())

    print(f"📅 Processing date {date} with {num_cpu} parallel processes")

    blocks = create_blocks(date, num_cpu)

    with Pool(processes=num_cpu) as pool:
        list(tqdm(pool.imap_unordered(process_block, blocks), total=len(blocks)))

if __name__ == "__main__":
    main()
