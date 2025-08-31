from datasets import Dataset
from multiprocessing import cpu_count
import sys
import argparse

dataset = Dataset.load_from_disk("../data/fineweb_edu_100BT_preprocessed")
num_proc = cpu_count()

def main(date):
    """
    Filter the dataset by a specific date.
    """
    date = int(date)
    filtered_dataset = dataset.filter(lambda x: x['date'] == date, num_proc=num_proc)
    filtered_dataset.save_to_disk(f"../data/fineweb_edu_100BT_preprocessed_filtered_{date}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter dataset by date.")
    parser.add_argument("date", type=int, help="The year of the date to filter the dataset by (as an integer).")
    args = parser.parse_args()

    if len(sys.argv) != 2:
        print("Usage: python filter_by_date.py <date>")
        sys.exit(1)

    main(args.date)