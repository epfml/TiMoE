from datasets import load_dataset
from tqdm import tqdm
import tiktoken
import multiprocessing
from multiprocessing import cpu_count
import re
import os
import argparse
import zarr
import numpy as np

num_proc = cpu_count()

def save_results_to_zarr(results, output_dir="../preprocessing_results/tokens_per_year"):
    os.makedirs(output_dir, exist_ok=True)
    store = zarr.DirectoryStore(output_dir)
    root = zarr.group(store=store, overwrite=True)

    for year, tokens in tqdm(results.items(), desc="Saving to Zarr"):
        tokens_array = np.array(tokens, dtype=np.uint16)  # use np.uint16 if tokenizer vocabulary size < 65536
        root.create_dataset(str(year), data=tokens_array, compressor=zarr.Blosc(cname='zstd', clevel=5))

    print("✅ Zarr save complete at", output_dir)

def process_data(example, min_date, max_date, tokenizer):
    """ tokenize and encode the text and date of the example

    Args:
        example (dict): the example to process
        min_date (int): the minimum date in the dataset
        max_date (int): the maximum date in the dataset
        tokenizer (tiktoken.Tokenizer): the tokenizer to use
    Returns:
        dict: the processed example
    """
    text_tokens = tokenizer.encode_ordinary(example["text"])
    text_tokens.append(tokenizer.eot_token)
    #date = (int(example["date"][:4]) - min_date) // 2 + 1
    date = int(example["dump"].split("CC-MAIN-")[1].split("-")[0])
    return {"tokens": text_tokens, "date": date}


def process_subset(start, end, dataset, tokenizer, min_date, max_date, output_dir):
    """ Process a subset of the dataset and save the results. """
    if end is None:
        end = len(dataset)
    subset = dataset.select(range(start, end))
    results = {}

    print("Now tokenizing dataset")
    tokenized = subset.map(
        process_data,
        remove_columns=["text"],
        desc="Tokenizing the splits",
        num_proc=num_proc,
        fn_kwargs={"min_date": min_date, "max_date": max_date, "tokenizer": tokenizer},
    )
    
    for example in tqdm(tokenized, desc=f"Processing subset {start} to {end}"):
        date = example["date"]
        tokens = example["tokens"]
        if date not in results:
            results[date] = []
        results[date].extend(tokens)

    output_dir = os.path.join(output_dir, f"{start}-{end}")
    
    save_results_to_zarr(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/mloscratch/homes/faro/thesis/preprocessing_results/fineweb_edu/tokens_per_year", help="Directory to save the preprocessed data.")
    parser.add_argument("--start", type=int, default=0, help="Start index of the dataset.")
    parser.add_argument("--end", type=int, default=None, help="End index of the dataset.")
    #dataset = load_dataset("HuggingFaceFW/fineweb", "sample-100BT", cache_dir="data/fineweb-100BTsample")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", cache_dir="../data/fineweb-edu-sample100BT")
    print("Dataset loaded")
    tokenizer = tiktoken.get_encoding("gpt2")

    min_date = 2013
    max_date = 2024

    args = parser.parse_args()
    process_subset(args.start, args.end, dataset["train"], tokenizer, min_date, max_date, args.output_dir)
    print("Done.")