#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter


def load_experiments(root_dir: Path, prefix: str):
    experiments = []

    for exp_dir in sorted(root_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name

        if not exp_name.startswith(prefix):
            continue

        args_file = exp_dir / f"{exp_name}_args.json"
        metrics_file = exp_dir / f"{exp_name}_test_metrics_full.json"

        if not args_file.exists() or not metrics_file.exists():
            continue

        try:
            with open(args_file, "r") as f:
                args_data = json.load(f)

            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            row = {
                "experiment": exp_name,
                "mAP": metrics_data["csi_metrics"]["mAP"],
                "rank_1_acc": metrics_data["csi_metrics"]["rank_1_acc"],
                "rank_5_acc": metrics_data["csi_metrics"]["rank_5_acc"],
                "mrr": metrics_data["csi_metrics"]["mrr"],
                "roc_auc": metrics_data["osi_metrics"]["roc_curve"]["auc"],
                "eer": metrics_data["osi_metrics"]["eer"]["val"],
                "epoch": metrics_data["epoch"],
                "raw_metrics": metrics_data,
            }

            row.update(args_data)

            patch_count = args_data.get("patch_count", 1)
            patch_h = args_data.get("patch_height", 1)
            patch_w = args_data.get("patch_width", 1)

            row["seen_pixels"] = patch_count * patch_h * patch_w

            experiments.append(row)

        except Exception as e:
            print(f"[WARN] Failed to load {exp_name}: {e}")

    return pd.DataFrame(experiments)


def keep_only_varying_columns(df, always_keep):
    cols = []

    for col in df.columns:
        if col in always_keep:
            cols.append(col)
            continue

        if df[col].nunique(dropna=False) > 1:
            cols.append(col)

    return df[cols]


def pareto_front(df, x_col, y_col):
    """
    Minimalizuje x_col
    Maximalizuje y_col
    """

    sorted_df = df.sort_values(
        by=[x_col, y_col],
        ascending=[True, False],
    )

    pareto_rows = []
    best_y = -1

    for _, row in sorted_df.iterrows():
        if row[y_col] > best_y:
            pareto_rows.append(True)
            best_y = row[y_col]
        else:
            pareto_rows.append(False)

    return sorted_df[pareto_rows]


def save_table(df, out_csv, out_md):
    df.to_csv(out_csv, index=False)

    with open(out_md, "w") as f:
        f.write(df.to_markdown(index=False))


def plot_pareto(df, pareto_df, out_png):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["seen_pixels"],
        df["mAP"],
    )

    for _, row in df.iterrows():
        plt.annotate(
            row["experiment"],
            (row["seen_pixels"], row["mAP"]),
            fontsize=8,
        )

    pareto_sorted = pareto_df.sort_values("seen_pixels")

    plt.plot(
        pareto_sorted["seen_pixels"],
        pareto_sorted["mAP"],
        linewidth=2,
    )

    plt.xlabel("Seen pixels = patch_count * patch_height * patch_width")
    plt.ylabel("mAP")
    plt.title("Pareto Front")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print(f"[OK] Saved Pareto plot: {out_png}")


def plot_oscr_curves(experiments, out_png):
    """
    Publication-quality OSCR plot.
    """

    plt.figure(figsize=(5.2, 4.0))

    # seřadit podle mAP
    experiments = sorted(
        experiments,
        key=lambda x: x["mAP"],
        reverse=True,
    )

    labels = {
        "resnet_512_gem": "Baseline - ResNet",
        "random_pc_50": "Random (50 patchů 48x48)",
        "sift_pc_50": "SIFT (50 patchů 48x48)",
        "grid_best": "Grid (patche 48x48)",
    }

    for exp in experiments:
        metrics = exp["raw_metrics"]

        curve = metrics["osi_metrics"]["oscr_curve"]

        if "x_fpir" not in curve or "y_tpir" not in curve:
            continue

        x = curve["x_fpir"]
        y = curve["y_tpir"]

        # log-scale často vhodnější pro FPIR
        x = [max(v, 1e-4) for v in x]

        label = (
            labels.get(exp["experiment"], exp["experiment"])
        )

        if len(y) >= 11:
            y_smooth = savgol_filter(
                y,
                window_length=11,
                polyorder=2,
            )
        else:
            y_smooth = y

        plt.plot(
            x,
            y_smooth,
            linewidth=1.5,
            label=label,
        )

    # logaritmická osa FPIR
    plt.xlim(0.0001, 0.1)

    plt.xlabel("FPIR", fontsize=12)
    plt.ylabel("TPIR", fontsize=12)

    # plt.title("OSCR Curves", fontsize=13)

    ax = plt.gca()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # jemnější grid
    plt.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )

    # menší tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # legenda mimo graf
    plt.legend(
        fontsize=8,
        loc="lower right",
        frameon=True,
    )

    # lepší spacing
    plt.tight_layout()

    # publication-quality export
    plt.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
    )

    # doporučeno i PDF
    pdf_path = str(out_png).replace(".png", ".pdf")

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[OK] Saved OSCR plot: {out_png}")
    print(f"[OK] Saved OSCR plot: {pdf_path}")


def plot_roc_curves(experiments, out_png):
    plt.figure(figsize=(10, 8))

    for exp in experiments:
        metrics = exp["raw_metrics"]

        curve = metrics["osi_metrics"]["roc_curve"]

        if "x_fpr" not in curve or "y_tpr" not in curve:
            continue

        x = curve["x_fpr"]
        y = curve["y_tpr"]

        label = (
            f'{exp["experiment"]} '
            f'(mAP={exp["mAP"]:.4f}, '
            f'AUC={curve["auc"]:.4f})'
        )

        plt.plot(x, y, label=label)

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")

    plt.grid(True)
    plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print(f"[OK] Saved ROC plot: {out_png}")


def plot_det_curves(experiments, out_png):
    plt.figure(figsize=(10, 8))

    for exp in experiments:
        metrics = exp["raw_metrics"]

        curve = metrics["osi_metrics"]["det_curve"]

        if "x_fpr" not in curve or "y_fnr" not in curve:
            continue

        x = curve["x_fpr"]
        y = curve["y_fnr"]

        label = (
            f'{exp["experiment"]} '
            f'(mAP={exp["mAP"]:.4f}, '
            f'AUC={curve["auc"]:.4f})'
        )

        plt.plot(x, y, label=label)

    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.title("DET Curves")

    plt.grid(True)
    plt.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

    print(f"[OK] Saved DET plot: {out_png}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiments_dir",
        type=str,
        help="Adresář se všemi experimenty",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default="",
        help="Prefix experimentů (např. grid, sift, random)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Výstupní adresář",
    )

    args = parser.parse_args()

    root_dir = Path(args.experiments_dir)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    raw_df = load_experiments(root_dir, args.prefix)

    if len(raw_df) == 0:
        print("No experiments found.")
        return

    raw_df = raw_df.sort_values(
        "mAP",
        ascending=False,
    ).reset_index(drop=True)

    experiments = raw_df.to_dict(orient="records")

    df = raw_df.copy()

    important_cols = [
        "experiment",
        "mAP",
        "rank_1_acc",
        "rank_5_acc",
        "mrr",
        "roc_auc",
        "eer",
        "epoch",
        "seen_pixels",
    ]

    remove_cols = {
        "raw_metrics",
        "gt_file",
        "gt_file_gallery",
        "gt_file_query",
        "lmdb",
        "sift_keypoints_lmdb",
        "show_dir",
        "out_checkpoints_dir",
        "out_model_name",
        "logging_level",
        "start_iteration",
    }

    df = df.drop(
        columns=[c for c in remove_cols if c in df.columns]
    )

    df = keep_only_varying_columns(
        df,
        important_cols,
    )

    csv_path = outdir / f"{args.prefix}_summary.csv"
    md_path = outdir / f"{args.prefix}_summary.md"

    save_table(df, csv_path, md_path)

    print(f"[OK] Saved summary CSV: {csv_path}")
    print(f"[OK] Saved summary MD : {md_path}")

    pareto_df = pareto_front(
        df,
        "seen_pixels",
        "mAP",
    )

    pareto_csv = outdir / f"{args.prefix}_pareto.csv"

    pareto_df.to_csv(
        pareto_csv,
        index=False,
    )

    print(f"[OK] Saved Pareto CSV: {pareto_csv}")

    pareto_plot = outdir / f"{args.prefix}_pareto.png"

    plot_pareto(
        df,
        pareto_df,
        pareto_plot,
    )

    plot_oscr_curves(
        experiments,
        outdir / f"{args.prefix}_oscr.png",
    )

    plot_roc_curves(
        experiments,
        outdir / f"{args.prefix}_roc.png",
    )

    plot_det_curves(
        experiments,
        outdir / f"{args.prefix}_det.png",
    )


if __name__ == "__main__":
    main()
