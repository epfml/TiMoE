import os
from datasets import load_from_disk, concatenate_datasets, Dataset, Sequence, Value
from tqdm import tqdm

SHARDS_ROOT = "../preprocessing_results/shards_parallel_fineweb_edu"
FINAL_DATASET_PATH = "../data/fineweb_edu_100BT_preprocessed"  



def load_all_batches():
    all_batches = []

    print("📦 Scanning all batches...")
    for date_folder in sorted(os.listdir(SHARDS_ROOT)):
    #for date_folder in dates_path:
        date_path = os.path.join(SHARDS_ROOT, date_folder)
        if not os.path.isdir(date_path):
            continue
        for block_folder in sorted(os.listdir(date_path)):
            block_path = os.path.join(date_path, block_folder)
            if not os.path.isdir(block_path):
                continue
            for batch_folder in sorted(os.listdir(block_path)):
                batch_path = os.path.join(block_path, batch_folder)
                if os.path.isdir(batch_path):
                    all_batches.append(batch_path)

    print(f"✅ Found {len(all_batches)} batches")

    return all_batches

def load_and_merge(all_batch_paths):
    datasets_list = []
    for path in tqdm(all_batch_paths, desc="📚 Loading batches"):
        ds = load_from_disk(path)
        ds = ds.cast_column("tokens", Sequence(Value("int32")))
        ds = ds.cast_column("date", Value("int16"))
        datasets_list.append(ds)
    print("🔗 Concatenating all batches...")
    full_dataset = concatenate_datasets(datasets_list)
    return full_dataset

def main():
    batch_paths = load_all_batches()
    dataset = load_and_merge(batch_paths)

    print(f"💾 Dataset saved in: {FINAL_DATASET_PATH}")
    dataset.save_to_disk(FINAL_DATASET_PATH)

    print("✅ Done!")

if __name__ == "__main__":
    main()
