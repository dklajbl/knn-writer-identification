#!/usr/bin/env python3
"""
Read one or more whitespace-delimited text files whose lines have the format:

    <filename>  <author_id>  <width>  <height>  [<other fields> ...]

and produce train/val/test splits that are disjoint by author.

Filtering pipeline (applied in this order):
  1. Drop malformed lines (fewer than 2 fields).
  2. Drop lines whose width field is missing or below --width_min.
  3. Randomly undersample each input file independently via --ratios.
  4. Drop authors with fewer than --min_count samples (after undersampling).
  5. Cap each author at --max_count samples (optional).
  6. Randomly subsample down to --max_authors authors (optional).
  7. Assign surviving authors to train / val / test by greedy sample-count
     balancing and write the output files.
  8. From test.txt, sample --gallery_query_k lines per author (default 25)
     and write gallery.txt and query.txt with disjoint but equal-sized sets.
     Authors with fewer than 2×k lines are skipped for these two files.

Usage:
    python3 generate_splits.py \
        --files file1.txt file2.txt file3.txt \
        --ratios 0.7 0.5 1.0 \
        --min_count 50 --max_count 2000 \
        --max_authors 500 \
        --width_min 300 \
        --train_size 0.8 --val_size 0.1 \
        --seed 42 \
        --output_dir ./splits

Outputs (written to --output_dir):
    train.txt   val.txt   test.txt   stats.txt
    gallery.txt   query.txt  (sampled from test authors, k samples each)
"""

import argparse
import random
import sys
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",      nargs="+", required=True,  help="Input files")
    parser.add_argument("--ratios",     nargs="+", type=float, required=True, help="Undersample ratio per file (0-1)")
    parser.add_argument("--min_count",  type=int,  default=5,     help="Min samples per author after undersampling")
    parser.add_argument("--max_count",   type=int,  default=None, help="Max samples per author (cap, optional)")
    parser.add_argument("--max_authors", type=int,  default=None, help="Max number of authors to keep (randomly sampled, optional)")
    parser.add_argument("--width_min",  type=int,  default=None,  help="Min width value per line (lines below this are dropped)")
    parser.add_argument("--train_size", type=float, default=0.8,  help="Train split ratio (by sample count)")
    parser.add_argument("--val_size",   type=float, default=0.1,  help="Val split ratio (by sample count)")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--output_dir",       type=str, default=".",  help="Directory to store output files")
    parser.add_argument("--gallery_query_k",  type=int, default=25,
                        help="Samples per author in gallery.txt and query.txt (drawn from test split, default 25)")
    return parser.parse_args()


def assign_splits_by_samples(author_counts, train_size, val_size, rng):
    """Shuffle authors randomly, then greedily fill train → val → test
    by accumulated sample count so each split hits its target ratio."""
    authors = list(author_counts.keys())
    rng.shuffle(authors)
    total_samples = sum(author_counts.values())

    assignment = {}
    train_samples = val_samples = 0

    for author in authors:
        n = author_counts[author]
        if train_samples + n <= train_size * total_samples:
            assignment[author] = "train"
            train_samples += n
        elif val_samples + n <= val_size * total_samples:
            assignment[author] = "val"
            val_samples += n
        else:
            assignment[author] = "test"

    return assignment


def percentile(sorted_vals, p):
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _fmt(v, is_int=False):
    """Format a number for the stats table."""
    if isinstance(v, float) and not is_int:
        return f"{v:,.1f}"
    return f"{int(v):,}"


def three_col_stats(counts, widths, heights):
    """Return a formatted three-column statistics table (Samples | Width | Height).

    Prints Min, p25, Median, p75, p95, Max, Avg for all three columns.
    counts  -- sorted list of per-author sample counts (ints).
    widths  -- flat list of line-width values (ints); may be empty,
               in which case the Width column shows n/a throughout.
    heights -- flat list of line-height values (ints); may be empty,
               in which case the Height column shows n/a throughout.
    """
    if not counts:
        return "  (no data)\n"

    sorted_w = sorted(widths)  if widths  else []
    sorted_h = sorted(heights) if heights else []
    has_w = bool(sorted_w)
    has_h = bool(sorted_h)

    total = sum(counts)
    n     = len(counts)
    w_avg = sum(sorted_w) / len(sorted_w) if has_w else None
    h_avg = sum(sorted_h) / len(sorted_h) if has_h else None

    def wv(p=None, mn=False, mx=False):
        if not has_w:
            return "n/a"
        if mn: return _fmt(sorted_w[0],  is_int=True)
        if mx: return _fmt(sorted_w[-1], is_int=True)
        return _fmt(percentile(sorted_w, p))

    def hv(p=None, mn=False, mx=False):
        if not has_h:
            return "n/a"
        if mn: return _fmt(sorted_h[0],  is_int=True)
        if mx: return _fmt(sorted_h[-1], is_int=True)
        return _fmt(percentile(sorted_h, p))

    rows = [
        ("Min",    _fmt(counts[0],  is_int=True), wv(mn=True),  hv(mn=True)),
        ("p25",    _fmt(percentile(counts, 25)),  wv(p=25),     hv(p=25)),
        ("Median", _fmt(percentile(counts, 50)),  wv(p=50),     hv(p=50)),
        ("p75",    _fmt(percentile(counts, 75)),  wv(p=75),     hv(p=75)),
        ("p95",    _fmt(percentile(counts, 95)),  wv(p=95),     hv(p=95)),
        ("Max",    _fmt(counts[-1], is_int=True), wv(mx=True),  hv(mx=True)),
        ("Avg",    f"{total/n:,.2f}",
                   f"{w_avg:,.2f}" if has_w else "n/a",
                   f"{h_avg:,.2f}" if has_h else "n/a"),
    ]

    # Compute column widths dynamically so values always right-align neatly.
    lbl_w  = max(len(r[0]) for r in rows)
    samp_w = max(len(r[1]) for r in rows + [("", "Samples", "", "")])
    wid_w  = max(len(r[2]) for r in rows + [("", "", "Width", "")])
    hei_w  = max(len(r[3]) for r in rows + [("", "", "", "Height")])

    sep = f"  {'-' * lbl_w}-+-{'-' * samp_w}-+-{'-' * wid_w}-+-{'-' * hei_w}"
    hdr = f"  {'Stat':<{lbl_w}} | {'Samples':>{samp_w}} | {'Width':>{wid_w}} | {'Height':>{hei_w}}"

    lines = [hdr, sep]
    for label, sv, wv_, hv_ in rows:
        lines.append(f"  {label:<{lbl_w}} | {sv:>{samp_w}} | {wv_:>{wid_w}} | {hv_:>{hei_w}}")
    return "\n".join(lines) + "\n"


def split_stats(split_name, per_author_counts, total_authors, total_samples, widths, heights):
    """Format a labelled statistics block for one split (train / val / test).

    Reports author and sample counts as both absolute numbers and percentages
    of the overall totals, followed by the three-column per-author sample,
    width, and height distribution table.
    """
    counts = sorted(per_author_counts.values())
    if not counts:
        return f"\n{split_name}:\n  No authors.\n"
    total = sum(counts)
    n = len(counts)
    header = (
        f"\n{split_name}:\n"
        f"  Authors : {n:,} ({n/total_authors*100:.1f}%)\n"
        f"  Samples : {total:,} ({total/total_samples*100:.1f}%)\n"
    )
    return header + three_col_stats(counts, widths, heights)


def file_stats(path, author_counts, widths, heights):
    """Format raw per-file statistics (before any filtering or undersampling).

    Shows total line and author counts followed by the three-column per-author
    sample, width, and height distribution table.
    """
    counts = sorted(author_counts.values())
    if not counts:
        return "  (no data)\n"
    total = sum(counts)
    n = len(counts)
    header = (
        f"  Lines   : {total:,}\n"
        f"  Authors : {n:,}\n"
    )
    return header + three_col_stats(counts, widths, heights)


def main():
    """Entry point: parse arguments, run the two-pass pipeline, write outputs."""
    args = parse_args()
    if len(args.files) != len(args.ratios):
        raise ValueError("Number of --files and --ratios must match")
    if args.val_size < 0:
        raise ValueError(f"--val_size must be >= 0, got {args.val_size}")
    if args.train_size + args.val_size > 1.0:
        raise ValueError(f"--train_size + --val_size must be <= 1.0")
    if args.max_count is not None and args.max_count < args.min_count:
        raise ValueError(f"--max_count ({args.max_count}) must be >= --min_count ({args.min_count})")

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    class Tee:
        """Write stdout to both terminal and a file."""
        def __init__(self, path):
            self.terminal = sys.stdout
            self.file = open(path, "w", encoding="utf-8")
        def write(self, msg):
            self.terminal.write(msg)
            self.file.write(msg)
        def flush(self):
            self.terminal.flush()
            self.file.flush()
        def close(self):
            self.file.close()

    tee = Tee(os.path.join(args.output_dir, "stats.txt"))
    sys.stdout = tee

    # --- Echo all parameters to stdout (captured by Tee into stats.txt) ---
    file_list = "\n".join(f"    [{i+1}] {f}  (ratio={r})" for i, (f, r) in enumerate(zip(args.files, args.ratios)))
    print(f"""Run configuration:
  Output dir   : {args.output_dir}
  Seed         : {args.seed}
  Min count    : {args.min_count}
  Max count    : {args.max_count if args.max_count is not None else "unlimited"}
  Width min    : {args.width_min if args.width_min is not None else "unlimited"}
  Max authors  : {args.max_authors if args.max_authors is not None else "unlimited"}
  Train size   : {args.train_size}
  Val size     : {args.val_size}
  Test size    : {1.0 - args.train_size - args.val_size:.2f}
  Gallery/query k : {args.gallery_query_k}
  Input files  :
{file_list}
""")

    rng = random.Random(args.seed)

    # --- Pass 1: scan all input files ---
    # Collect raw per-file statistics and build per-author sample counts
    # after applying the width filter and per-file undersampling ratio.
    print("Pass 1: collecting authors and estimating post-undersample counts...")
    author_counts = Counter()   # merged across all files, after width filter + undersampling
    total_lines = 0

    # Per-reason drop counters for Pass 1 (used in the deletion summary).
    p1_deleted_malformed   = 0  # lines with fewer than 2 whitespace-separated fields
    p1_deleted_width_min   = 0  # lines whose width field is absent or below --width_min
    p1_deleted_undersample = 0  # lines randomly discarded by per-file --ratios

    print("\nInput file statistics (raw, before undersampling/filtering):")
    for path, ratio in zip(args.files, args.ratios):
        raw_counts = Counter()
        raw_widths = []
        raw_heights = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                if total_lines % 1_000_000 == 0:
                    print(f"  {total_lines:,} lines scanned...")
                parts = line.split()
                if len(parts) < 2:
                    p1_deleted_malformed += 1
                    continue
                raw_counts[parts[1]] += 1

                width = None
                if len(parts) >= 3:
                    try:
                        width = int(parts[2])
                        raw_widths.append(width)
                    except ValueError:
                        pass

                # Parse height from the 4th column (index 3).
                if len(parts) >= 4:
                    try:
                        raw_heights.append(int(parts[3]))
                    except ValueError:
                        pass

                # Drop line if width is missing or below --width_min.
                if args.width_min is not None:
                    if width is None or width < args.width_min:
                        p1_deleted_width_min += 1
                        continue
                # Keep line with probability `ratio`; count it toward its author.
                if rng.random() <= ratio:
                    author_counts[parts[1]] += 1
                else:
                    p1_deleted_undersample += 1

        print(f"  {path}  (ratio={ratio})")
        print(file_stats(path, raw_counts, raw_widths, raw_heights), end="")

    print(f"\nMerged after undersampling: {total_lines:,} lines, {len(author_counts):,} authors.")

    print(f"\nPass 1 deletion summary:")
    print(f"  Malformed lines (< 2 fields) : {p1_deleted_malformed:,}")
    print(f"  Dropped by width_min filter  : {p1_deleted_width_min:,}")
    print(f"  Dropped by undersampling     : {p1_deleted_undersample:,}")
    print(f"  Retained for author counting : {sum(author_counts.values()):,}")

    # --- Filter authors by min_count, then cap per-author sample counts ---
    safe_counts = {a: c for a, c in author_counts.items() if c >= args.min_count}
    pruned_min = len(author_counts) - len(safe_counts)
    lines_pruned_min = sum(c for a, c in author_counts.items() if c < args.min_count)

    if args.max_count is not None:
        capped = {a: min(c, args.max_count) for a, c in safe_counts.items()}
        num_capped = sum(1 for a, c in safe_counts.items() if c > args.max_count)
        lines_capped = sum(max(0, c - args.max_count) for c in safe_counts.values())
    else:
        capped = safe_counts
        num_capped = 0
        lines_capped = 0

    print(f"\nAuthor/line filtering summary:")
    print(f"  Authors pruned (< {args.min_count} samples) : {pruned_min:,}  ({lines_pruned_min:,} lines dropped)")
    if args.max_count is not None:
        print(f"  Authors capped  (> {args.max_count} samples) : {num_capped:,}  (~{lines_capped:,} excess lines dropped)")

    # --- Apply max_authors cap ---
    # Randomly subsample the surviving author pool down to --max_authors.
    # Uses the same seeded rng for reproducibility.
    if args.max_authors is not None and len(capped) > args.max_authors:
        chosen = rng.sample(sorted(capped), args.max_authors)
        dropped_authors = len(capped) - args.max_authors
        lines_dropped_authors = sum(capped[a] for a in capped if a not in set(chosen))
        capped = {a: capped[a] for a in chosen}
        print(f"  Authors dropped by max_authors cap : {dropped_authors:,}  ({lines_dropped_authors:,} lines dropped)")
    else:
        dropped_authors = 0

    # --- Assign each surviving author to train / val / test ---
    assignment = assign_splits_by_samples(capped, args.train_size, args.val_size, rng)

    # --- Pass 2: write output splits ---
    # Re-read all input files and apply the identical filter pipeline as Pass 1
    # (same seed → same random decisions), then route each line to the split
    # assigned to its author and enforce the per-author --max_count cap.
    rng = random.Random(args.seed)
    print("\nPass 2: writing splits...")

    per_author = defaultdict(Counter)   # split -> {author: lines written}
    author_written = Counter()          # author -> total lines written (used to enforce max_count)
    split_widths  = defaultdict(list)   # split -> flat list of width  values (for stats)
    split_heights = defaultdict(list)   # split -> flat list of height values (for stats)

    # Per-reason drop counters for Pass 2 (mirroring Pass 1 where applicable).
    p2_deleted_malformed   = 0  # fewer than 2 fields
    p2_deleted_width_min   = 0  # width absent or below --width_min
    p2_deleted_undersample = 0  # randomly discarded by per-file --ratios
    p2_deleted_min_count   = 0  # author absent from assignment (removed by min_count or max_authors)
    p2_deleted_max_count   = 0  # author already reached --max_count written lines

    with open(os.path.join(args.output_dir, "train.txt"), "w", encoding="utf-8") as f_train, \
         open(os.path.join(args.output_dir, "val.txt"),   "w", encoding="utf-8") as f_val, \
         open(os.path.join(args.output_dir, "test.txt"),  "w", encoding="utf-8") as f_test:

        handles = {"train": f_train, "val": f_val, "test": f_test}
        processed = 0

        for path, ratio in zip(args.files, args.ratios):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    processed += 1
                    if processed % 1_000_000 == 0:
                        print(f"  {processed:,} lines processed...")
                    parts = line.split()
                    if len(parts) < 2:
                        p2_deleted_malformed += 1
                        continue
                    author = parts[1]

                    # Parse width field and drop line if below --width_min.
                    width = None
                    if len(parts) >= 3:
                        try:
                            width = int(parts[2])
                        except ValueError:
                            pass
                    if args.width_min is not None:
                        if width is None or width < args.width_min:
                            p2_deleted_width_min += 1
                            continue

                    # Parse height from the 4th column (index 3).
                    height = None
                    if len(parts) >= 4:
                        try:
                            height = int(parts[3])
                        except ValueError:
                            pass

                    # Mirror Pass 1 undersampling (identical seed → identical decisions).
                    if rng.random() > ratio:
                        p2_deleted_undersample += 1
                        continue

                    # Skip authors removed by min_count or max_authors filtering.
                    if author not in assignment:
                        p2_deleted_min_count += 1
                        continue

                    # Enforce --max_count: stop writing once the author's cap is reached.
                    if args.max_count is not None and author_written[author] >= args.max_count:
                        p2_deleted_max_count += 1
                        continue

                    split = assignment[author]
                    handles[split].write(line)
                    author_written[author] += 1
                    per_author[split][author] += 1
                    if width is not None:
                        split_widths[split].append(width)
                    if height is not None:
                        split_heights[split].append(height)

    total_final = sum(sum(c.values()) for c in per_author.values())
    total_authors = sum(len(c) for c in per_author.values())

    print(f"\nPass 2 deletion summary (out of {processed:,} lines read):")
    print(f"  Malformed lines (< 2 fields) : {p2_deleted_malformed:,}")
    print(f"  Dropped by width_min filter  : {p2_deleted_width_min:,}")
    print(f"  Dropped by undersampling     : {p2_deleted_undersample:,}")
    print(f"  Dropped — author below min_count / over max_authors : {p2_deleted_min_count:,}")
    print(f"  Dropped — author hit max_count   : {p2_deleted_max_count:,}")
    total_deleted_p2 = (p2_deleted_malformed + p2_deleted_width_min +
                        p2_deleted_undersample + p2_deleted_min_count + p2_deleted_max_count)
    print(f"  Total dropped : {total_deleted_p2:,}")
    print(f"  Total written : {total_final:,}")

    print(f"\nDone! Total written: {total_final:,} lines, pruned authors: {pruned_min:,}")
    for split in ("train", "val", "test"):
        print(split_stats(
            split, per_author[split], total_authors, total_final,
            split_widths[split], split_heights[split],
        ), end="")

    # --- Build gallery.txt and query.txt from the test split ---
    # Read test.txt back, group lines by author, then for each author that has
    # at least 2*k lines draw two disjoint random samples of size k.
    k = args.gallery_query_k
    test_path = os.path.join(args.output_dir, "test.txt")
    test_lines_by_author: dict[str, list[str]] = defaultdict(list)
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                test_lines_by_author[parts[1]].append(line)

    # Use a fresh seeded RNG so gallery/query sampling is reproducible
    # independently of any RNG state left over from Pass 2.
    gq_rng = random.Random(args.seed)
    gallery_lines: list[str] = []
    query_lines:   list[str] = []
    skipped_authors = 0
    included_authors = 0
    # For each test author with enough lines, draw 2*k distinct samples
    # and split them evenly: first half → gallery, second half → query.
    for author, lines in sorted(test_lines_by_author.items()):
        if len(lines) < 2 * k:
            skipped_authors += 1
            continue
        sampled = gq_rng.sample(lines, 2 * k)
        gallery_lines.extend(sampled[:k])   # indices 0..k-1
        query_lines.extend(sampled[k:])     # indices k..2k-1 (disjoint from gallery)
        included_authors += 1

    gallery_path = os.path.join(args.output_dir, "gallery.txt")
    query_path   = os.path.join(args.output_dir, "query.txt")
    with open(gallery_path, "w", encoding="utf-8") as fg, \
         open(query_path,   "w", encoding="utf-8") as fq:
        fg.writelines(gallery_lines)
        fq.writelines(query_lines)

    print(f"\nGallery / Query:")
    print(f"  k (samples per author) : {k}")
    print(f"  Authors included       : {included_authors:,}")
    print(f"  Authors skipped (< {2*k} test samples) : {skipped_authors:,}")
    print(f"  Lines in gallery.txt   : {len(gallery_lines):,}")
    print(f"  Lines in query.txt     : {len(query_lines):,}")


if __name__ == "__main__":
    main()
    if hasattr(sys.stdout, "close"):
        sys.stdout.close()
        sys.stdout = sys.__stdout__
