import argparse
import json
import logging
import os
import sys
import time
import torch
from pathlib import Path

from src.train_id_embedding import *

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(usage="Tests trained model.")

    parser.add_argument(
        "--gt-file-gallery",
        required=True,
        help="Test gallery text file with an image file name and id on each line."
    )
    parser.add_argument(
        "--gt-file-query",
        required=True,
        help="Test query text file with an image file name and id on each line."
    )
    parser.add_argument(
        "--lmdb",
        required=True,
        help="Path to LMDB database."
    )

    parser.add_argument(
        "--sift-keypoints-lmdb",
        type=str,
        default=None,
        help=(
            "Path to an LMDB produced by `python -m src.patchers.extract_keypoints` "
            "holding pre-computed SIFT keypoints. Required when --patcher sift."
        ),
    )

    parser.add_argument("--checkpoints-dir", required=True, type=str, help="Checkpoint folder.")
    parser.add_argument("--model-name", required=True, type=str, help="Name of model in checkpoint folder.")

    parser.add_argument("--num-workers", default=4, type=int, help="Number of DataLoader worker processes.")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size (does not influence evaluation metrics).")

    args = parser.parse_args()

    return args

def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    # set these arguments to these specific values otherwise it fails
    args.stop_checkpointing = False
    args.start_iteration = -1

    # get checkpoint dir
    checkpoint_dir = os.path.join(args.checkpoints_dir, f"{args.model_name}")

    # get path to model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model_name}.img.ckpt")

    if Path(checkpoint_dir).is_dir() and Path(checkpoint_path).is_file():
        # load args stored in checkpoint
        checkpoint_args = load_args_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # set args with the loaded arg values
        set_model_args(args, checkpoint_args)
    else:
        print("Checkpoint file does not exist.", file=sys.stderr)
        return 1

    # configure logger
    configure_logging(logging.INFO)

    # log command line arguments
    log_args(args, logger)

    # create model
    image_encoder = create_model(args, device)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # load state of model from the checkpoint
    image_encoder.load_state_dict(checkpoint["model_state_dict"])

    # create test gallery and query dataloader
    gallery_dataloader, query_dataloader = create_eval_dataloaders(args)

    image_encoder.eval()

    with torch.no_grad():
        # caulate test metrics
        test_metrics = eval_identification(
            encoder=image_encoder,
            gallery_dataloader=gallery_dataloader,
            query_dataloader=query_dataloader,
            device=device,
        )

        # log compact result to stdout
        logger.info(test_metrics.to_json_compact())

        # log compact and full result to checkpoint folder
        with open(os.path.join(checkpoint_dir, f"{args.model_name}_test_metrics_compact.json"), "w", encoding="utf-8") as f:
            f.write(test_metrics.to_json_compact())
        with open(os.path.join(checkpoint_dir, f"{args.model_name}_test_metrics_full.json"), "w", encoding="utf-8") as f:
            f.write(test_metrics.to_json())

        # flush the logger
        for handler in logger.handlers:
            handler.flush()

if __name__ == '__main__':
    main()
