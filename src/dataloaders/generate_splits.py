import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_line_file
from src.env_vars import TMP_DIR, ALL_LINE_FILE_PATH, NP_RANDOM_SEED, TRAIN_LINE_FILE_PATH, VAL_LINE_FILE_PATH



def prune_authors(data, authors, min_samples=2):
    safe_authors = [author for author in authors if len(data[author]) >= min_samples]
    pruned_authors = [author for author in authors if len(data[author]) < min_samples]
    return pruned_authors, safe_authors


def generate_splits(data, authors, train_size=0.8):
    # Each author has a list of samples. We want to split so that all samples from one author go to the same split.
    # Split size is by size of samples, not by number of authors.
    # So we need to shuffle authors and then split them until we reach the desired size for each split.
    np.random.shuffle(authors)
    len_samples = sum([len(data[author]) for author in authors])
    train_authors = []
    val_authors = []
    train_samples = 0
    val_samples = 0
    for author in authors:
        num_samples = len(data[author])
        if train_samples + num_samples <= train_size * len_samples:
            train_authors.append(author)
            train_samples += num_samples
        else:
            val_authors.append(author)
            val_samples += num_samples

    return train_authors, val_authors


def plot_sample_distribution(train, val, name):
    train, val = np.log(train), np.log(val)
    bins=40
    plt.hist(train, bins=bins, alpha=0.5, label="train")
    plt.hist(val, bins=bins, alpha=0.5, label="val")

    # Plot gaussian distribution for train and val
    train_mean, train_std = np.mean(train), np.std(train)
    val_mean, val_std = np.mean(val), np.std(val)
    x = np.linspace(min(min(train), min(val)), max(max(train), max(val)), 100)
    train_gaussian = (1 / (train_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - train_mean) / train_std) ** 2)
    val_gaussian = (1 / (val_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - val_mean) / val_std) ** 2)
    plt.plot(x, train_gaussian * len(train) * (x[1] - x[0]), label="train gaussian")
    plt.plot(x, val_gaussian * len(val) * (x[1] - x[0]), label="val gaussian")

    # Add vertical lines for mean and std
    plt.axvline(train_mean, color="blue", linestyle="dashed", label="train mean")
    plt.axvline(train_mean + train_std, color="blue", linestyle="dotted", label="train std")
    plt.axvline(train_mean - train_std, color="blue", linestyle="dotted")
    plt.axvline(val_mean, color="orange", linestyle="dashed", label="val mean")
    plt.axvline(val_mean + val_std, color="orange", linestyle="dotted", label="val std")
    plt.axvline(val_mean - val_std, color="orange", linestyle="dotted")

    plt.title(f"Sample distribution for train and test split")
    plt.xlabel("Number of Samples (log scale)")
    plt.ylabel("Number of Authors")
    plt.legend()
    plt.savefig(f"{TMP_DIR}/{name}_sample_distribution.png")
    plt.clf()


def eval_splits(train_authors, val_authors, pruned_authors, data):
    train_samples = [len(data[author]) for author in train_authors]
    val_samples = [len(data[author]) for author in val_authors]

    train_min, train_max, train_mean, train_std = np.min(train_samples), np.max(train_samples), np.mean(train_samples), np.std(train_samples)
    val_min, val_max, val_mean, val_std = np.min(val_samples), np.max(val_samples), np.mean(val_samples), np.std(val_samples)

    total_aurhors = len(train_authors) + len(val_authors) + len(pruned_authors)
    total_safe_authors = len(train_authors) + len(val_authors)
    total_samples = sum(train_samples) + sum(val_samples)

    stats = f"""All authors: {total_aurhors}
All safe authors: {total_safe_authors} ({total_safe_authors/total_aurhors*100:.2f}%)
Pruned authors: {len(pruned_authors)} ({len(pruned_authors)/total_aurhors*100:.2f}%)

All samples: {total_samples}

Train set:
Num authors: {len(train_authors)} ({len(train_authors)/total_safe_authors*100:.2f}%)
Num samples: {sum(train_samples)} ({sum(train_samples)/total_samples*100:.2f}%)
min={train_min}
max={train_max}
mean={train_mean:.2f}, std={train_std:.2f}

Val set:
Num authors: {len(val_authors)} ({len(val_authors)/total_safe_authors*100:.2f}%)
Num samples: {sum(val_samples)} ({sum(val_samples)/total_samples*100:.2f}%)
min={val_min}
max={val_max}
mean={val_mean:.2f}, std={val_std:.2f}
"""
    print(stats)
    with open(f"{TMP_DIR}/split_stats.txt", "w", encoding="utf-8") as f:
        f.write(stats)

    plot_sample_distribution(train_samples, val_samples, "split")


def save_split(data, authors, filename, shuffle=False):
    lines = []
    for author in authors:
        for sample in data[author]:
            lines.append(f"{sample} {author}\n")

    if shuffle:
        np.random.shuffle(lines)

    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)


def save_splits(data, train_authors, val_authors):
    save_split(data, train_authors, f"{TRAIN_LINE_FILE_PATH}")
    save_split(data, val_authors, f"{VAL_LINE_FILE_PATH}")


if __name__ == "__main__":
    np.random.seed(NP_RANDOM_SEED)
    data, authors = read_line_file(ALL_LINE_FILE_PATH)
    pruned, authors = prune_authors(data, authors, min_samples=2)
    train_authors, val_authors = generate_splits(data, authors, train_size=0.8)
    eval_splits(train_authors, val_authors, pruned, data)
    save_splits(data, train_authors, val_authors)
