import os
import numpy as np
import lmdb
import matplotlib.pyplot as plt

LINE_FILE_PATH = "/storage/plzen1/home/xkiszk00/data/lines.filtered_max_width.all"
DATA_PATH = "/storage/plzen1/home/xkiszk00/data/lmdb.hwr_40-1.0/"
OUT_DIR = "samples"

NUM_AUTHORS = 10
MAX_NUM_SAMPLES_PER_AUTHOR = 10

def prepare_output_directory(dir=OUT_DIR):
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_line_file():
    data = dict()
    authors = []
    with open(LINE_FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            file, author_id, *_ = line.split()
            authors.append(int(author_id))
            data[int(author_id)] = data.get(int(author_id), []) + [file]

    authors = list(set(authors))
    authors.sort()
    return data, authors

def sample_data():
    prepare_output_directory("samples")
    data, authors = read_line_file()

    # Extract samples for the first NUM_AUTHORS authors and save them to the output directory
    env = lmdb.open(DATA_PATH, readonly=True)
    with env.begin() as txn:
        for author_id in authors[:NUM_AUTHORS]:
            files = data[author_id][:MAX_NUM_SAMPLES_PER_AUTHOR]
            for file in files:
                key = file.encode("utf-8")
                value = txn.get(key)
                if value is not None:
                    prepare_output_directory(f"samples/{author_id}")
                    with open(f"samples/{author_id}/{file}", "wb") as out_file:
                        out_file.write(value)

def compute_statistics():
    prepare_output_directory("stats")
    data, authors = read_line_file()

    data_lens = [len(data[author_id]) for author_id in authors]
    data_mean = sum(data_lens) / len(data_lens)
    data_variance = sum((x - data_mean) ** 2 for x in data_lens) / len(data_lens)
    outliers = [x for x in data_lens if x > data_mean + 2 * data_variance ** 0.5 or x < data_mean - 2 * data_variance ** 0.5]
    ok_data = [x for x in data_lens if x not in outliers and x > 9]  # also filter out authors with less than 10 samples

    ok_mean = sum(ok_data) / len(ok_data)
    ok_variance = sum((x - ok_mean) ** 2 for x in ok_data) / len(ok_data)

    prepare_output_directory("splits")
    with open("splits/ok_authors.txt", "w", encoding="utf-8") as f:
        for author_id in authors:
            if len(data[author_id]) > 9 and len(data[author_id]) not in outliers:
                f.write(f"{author_id}\n")

    # Save statistics about the number of samples per author
    with open("stats/authors.csv", "w", encoding="utf-8") as stat_f:
        stat_f.write("author_id,num_samples\n")
        for author_id in authors:
            num_samples = len(data[author_id])
            stat_f.write(f"{author_id},{num_samples}\n")

    with open("stats/summary.txt", "w", encoding="utf-8") as summary_f:
        summary_f.write(f"---------------\n")
        summary_f.write(f"Total authors: {len(authors)}\n")
        summary_f.write(f"Mean number of samples per author: {data_mean:.2f}\n")
        summary_f.write(f"Variance of number of samples per author: {data_variance:.2f}\n")
        summary_f.write(f"Number of outliers (authors with too many or too few samples): {len(outliers)}\n")
        summary_f.write(f"Outlier sample counts: {outliers}\n\n")

        summary_f.write(f"---------------\n")
        summary_f.write(f"Total ok authors (after removing outliers and those with less than 10 samples): {len(ok_data)}\n")
        summary_f.write(f"Mean number of samples per author (without outliers): {ok_mean:.2f}\n")
        summary_f.write(f"Variance of number of samples per author (without outliers): {ok_variance:.2f}\n")
        summary_f.write(f"Min number of samples per author (without outliers): {min(ok_data)}\n")
        summary_f.write(f"Max number of samples per author (without outliers): {max(ok_data)}\n")

    # Create histogram of number of samples per author
    plt.figure(figsize=(10, 6))
    plt.hist(ok_data, bins=20, color="blue", edgecolor="black")
    plt.title("Distribution of Number of Samples per Author (removed outliers, min 10 samples)")
    plt.xlabel("Number of Samples")
    plt.ylabel("Number of Authors")
    plt.savefig("stats/samples_per_author_histogram.png")

def shuffle_and_split_authors(authors, train_ratio=0.8, val_ratio=0.1, max_per_split=1000000):
    np.random.shuffle(authors)
    num_train = min(int(len(authors) * train_ratio), max_per_split)
    num_val = min(int(len(authors) * val_ratio), max_per_split)
    train_authors = np.random.choice([int(a) for a in authors], num_train, replace=False)
    val_authors = np.random.choice([int(a) for a in authors if a not in train_authors], num_val, replace=False)
    test_authors = np.random.choice([int(a) for a in authors if a not in train_authors and a not in val_authors], max_per_split, replace=False)
    return train_authors, val_authors, test_authors

def prepare_triplets(data, authors, num_triplets_per_author=100000):
    triplets = []
    for author_id in authors:
        author_data = np.random.choice(data[author_id], num_triplets_per_author, replace=False)
        for sample in author_data:
            anchor = sample
            positive = np.random.choice([s for s in data[author_id] if s != anchor])
            negative_author = np.random.choice([a for a in authors if a != author_id])
            negative = np.random.choice(data[negative_author])
            triplets.append((anchor, positive, negative))
    return triplets

def save_authors(authors, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for author_id in authors:
            f.write(f"{author_id}\n")

def save_triplets(triplets, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for anchor, positive, negative in triplets:
            f.write(f"{anchor} {positive} {negative}\n")

def prepare_splits():
    prepare_output_directory("splits")
    data, authors = read_line_file()

    with open("splits/ok_authors.txt", "r", encoding="utf-8") as f:
        ok_authors = set(line.strip() for line in f)

    train_authors, val_authors, test_authors = shuffle_and_split_authors(list(ok_authors), max_per_split=10)
    save_authors(train_authors, "splits/train_authors.txt")
    save_authors(val_authors, "splits/val_authors.txt")
    save_authors(test_authors, "splits/test_authors.txt")


    train_triplets = prepare_triplets(data, train_authors, num_triplets_per_author=10)
    val_triplets = prepare_triplets(data, val_authors, num_triplets_per_author=10)
    test_triplets = prepare_triplets(data, test_authors, num_triplets_per_author=10)

    save_triplets(train_triplets, "splits/train_triplets.txt")
    save_triplets(val_triplets, "splits/val_triplets.txt")
    save_triplets(test_triplets, "splits/test_triplets.txt")

def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Data exploration for HWR dataset")
    p.add_argument("--variant", default="all", choices=["all", "data", "stats", "splits"], help="Type of exploration to perform")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(42)
    if args.variant == "data":
        sample_data()
    elif args.variant == "stats":
        compute_statistics()
    elif args.variant == "splits":
        prepare_splits()
    else:
        sample_data()
        compute_statistics()
        prepare_splits()

