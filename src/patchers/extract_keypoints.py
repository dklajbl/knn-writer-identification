import cv2
import lmdb
import pickle
import logging
import argparse
import numpy as np
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# supported detectors (only SIFT is implemented, FAST is scaffolded for later)
SUPPORTED_METHODS = ["sift", "fast"]


@dataclass
class ExtractionStats:

    """
    Running statistics collected during LMDB keypoint extraction.

    Attributes:
        processed (int): number of images successfully processed and written.
        skipped (int): number of images skipped because they already exist in the output LMDB.
        errors (int): number of images that could not be decoded or processed.
        fully_detected (int): images where the detector produced >= keypoint_count keypoints (no random fill).
        partially_detected (int): images where the detector produced 1..(keypoint_count-1) keypoints (partial random fill).
        empty_detected (int): images where the detector produced 0 keypoints (full random fallback).
        total_detected_keypoints (int): sum of "real" (detector-produced) keypoints across all processed images.
        total_target_keypoints (int): sum of requested keypoint slots (processed * keypoint_count).
        interrupted (bool): True if the run was stopped early by a KeyboardInterrupt (Ctrl+C).
    """

    processed: int = 0
    skipped: int = 0
    errors: int = 0

    fully_detected: int = 0
    partially_detected: int = 0
    empty_detected: int = 0

    total_detected_keypoints: int = 0
    total_target_keypoints: int = 0

    interrupted: bool = False


def parse_args() -> argparse.Namespace:

    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: parsed arguments.

    Raises:
        SystemExit: if required arguments are missing or invalid.
    """

    parser = argparse.ArgumentParser(
        description="Precompute keypoints (SIFT/FAST) for every image in an LMDB and store them into a new LMDB. "
                    "If the detector does not produce enough keypoints, the remainder is filled with random coordinates. "
                    "Each stored record remembers the split index between detector and random keypoints."
    )

    parser.add_argument("--input-lmdb", type=str, required=True, help="Path to the source LMDB containing images (keys = image ids, values = encoded image bytes).")
    parser.add_argument("--output-lmdb", type=str, required=True, help="Path to the output LMDB where the keypoint records will be written.")
    parser.add_argument("--method", type=str, default="sift", choices=SUPPORTED_METHODS, help="Keypoint detector to use. Currently only 'sift' is implemented.")
    parser.add_argument("--keypoint-count", type=int, default=100, help="Total number of keypoints per image (detector + random fill).")
    parser.add_argument("--rewrite-existing", action=argparse.BooleanOptionalAction, default=True, help="If set (default), overwrite entries that already exist in the output LMDB. Use --no-rewrite-existing to skip them (resume mode).")
    parser.add_argument("--random-seed", type=int, default=None, help="Optional RNG seed for reproducible random-fill keypoints.")
    parser.add_argument("--map-size-bytes", type=int, default=2 ** 40, help="LMDB map size for the output environment in bytes (default 1 TiB).")
    parser.add_argument("--log-every", type=int, default=1000, help="How often to emit progress log lines (every N processed images).")

    args = parser.parse_args()

    if args.keypoint_count <= 0:
        parser.error("--keypoint-count must be > 0")

    if args.log_every <= 0:
        parser.error("--log-every must be > 0")

    return args


def decode_grayscale_image(raw_bytes: bytes) -> np.ndarray | None:

    """
    Decode raw LMDB bytes (JPEG/PNG) into a grayscale numpy image.

    Parameters:
        raw_bytes (bytes): encoded image bytes retrieved from the input LMDB.

    Returns:
        np.ndarray | None: grayscale image of shape (H, W) with dtype uint8, or None if decoding fails.
    """

    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

    return image


def detect_sift_keypoints(gray: np.ndarray, max_count: int) -> np.ndarray:

    """
    Run OpenCV SIFT on a grayscale image and return the top-`max_count` keypoints
    sorted by response strength (descending).

    Parameters:
        gray (np.ndarray): grayscale image of shape (H, W).
        max_count (int): upper bound on the number of keypoints returned.

    Returns:
        np.ndarray: array of shape (N, 2) with dtype int32 where each row is (y, x) and N <= max_count.
            Returns an empty (0, 2) array if SIFT finds nothing.
    """

    sift = cv2.SIFT_create(nfeatures=max_count)
    keypoints = sift.detect(gray, None)

    if not keypoints:
        return np.empty((0, 2), dtype=np.int32)

    # sort strongest-first and clamp to the requested budget
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_count]

    # convert KeyPoint.pt = (x, y) -> (y, x) so downstream indexing matches image[y, x]
    coords = np.array(
        [(int(round(kp.pt[1])), int(round(kp.pt[0]))) for kp in keypoints],
        dtype=np.int32,
    )

    return coords


def generate_random_keypoints(image_shape: tuple[int, int], count: int, rng: np.random.Generator) -> np.ndarray:

    """
    Generate `count` random keypoints uniformly distributed over the image.

    Parameters:
        image_shape (tuple[int, int]): (H, W) of the source image.
        count (int): number of random keypoints to generate.
        rng (np.random.Generator): random number generator.

    Returns:
        np.ndarray: array of shape (count, 2) with dtype int32 where each row is (y, x).
    """

    if count <= 0:
        return np.empty((0, 2), dtype=np.int32)

    height, width = image_shape

    ys = rng.integers(0, height, size=count, dtype=np.int32)
    xs = rng.integers(0, width, size=count, dtype=np.int32)

    return np.stack([ys, xs], axis=1)


def build_keypoint_record(
    gray: np.ndarray,
    method: str,
    total_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:

    """
    Build the full (total_count, 2) keypoint array for a single image, combining detector keypoints with random fill-ins as needed.

    Parameters:
        gray (np.ndarray): grayscale image of shape (H, W).
        method (str): detector method ("sift" or "fast").
        total_count (int): target number of keypoints in the final array.
        rng (np.random.Generator): random number generator used for the fill-in.

    Returns:
        tuple[np.ndarray, int]: (keypoints, split_index) where
            - keypoints has shape (total_count, 2) and dtype int32
            - keypoints[:split_index] come from the detector
            - keypoints[split_index:] come from the random generator

    Raises:
        NotImplementedError: if `method` is recognized but not implemented yet (e.g. "fast").
        ValueError: if `method` is not a supported value.
    """

    if method == "sift":
        detected = detect_sift_keypoints(gray, total_count)

    elif method == "fast":
        raise NotImplementedError("FAST method is not implemented yet")

    else:
        raise ValueError(f"Unsupported method: {method!r}")

    split_index = int(detected.shape[0])
    remaining = total_count - split_index

    if remaining <= 0:
        return detected[:total_count], total_count

    random_fill = generate_random_keypoints(gray.shape[:2], remaining, rng)

    if split_index == 0:
        return random_fill, 0

    return np.concatenate([detected, random_fill], axis=0), split_index


def serialize_record(
    keypoints: np.ndarray,
    split_index: int,
    method: str,
    image_shape: tuple[int, int],
) -> bytes:

    """
    Serialize a keypoint record for storage in LMDB.

    Parameters:
        keypoints (np.ndarray): (N, 2) int32 array of (y, x) coordinates.
        split_index (int): boundary between detector and random keypoints.
        method (str): detector method that produced the first `split_index` keypoints.
        image_shape (tuple[int, int]): (H, W) of the source image.

    Returns:
        bytes: pickled payload suitable for LMDB `txn.put`.
    """

    payload = {
        "keypoints": keypoints,
        "split_index": split_index,
        "method": method,
        "image_shape": image_shape,
    }

    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _update_detection_stats(stats: ExtractionStats, split_index: int, keypoint_count: int) -> None:

    """
    Update the per-image detection counters in the stats accumulator.

    Parameters:
        stats (ExtractionStats): stats object mutated in place.
        split_index (int): number of detector keypoints for this image.
        keypoint_count (int): target total keypoints per image.
    """

    stats.total_detected_keypoints += split_index
    stats.total_target_keypoints += keypoint_count

    if split_index >= keypoint_count:
        stats.fully_detected += 1
    elif split_index == 0:
        stats.empty_detected += 1
    else:
        stats.partially_detected += 1


def process_lmdb(
    input_lmdb_path: str,
    output_lmdb_path: str,
    method: str,
    keypoint_count: int,
    rewrite_existing: bool,
    random_seed: int | None,
    map_size_bytes: int,
    log_every: int,
) -> ExtractionStats:

    """
    Iterate every key in the input LMDB, compute keypoints for each image, and write them
    to the output LMDB under the same key.

    Parameters:
        input_lmdb_path (str): path to the source (read-only) LMDB containing images.
        output_lmdb_path (str): path to the output LMDB where keypoint records will be written.
        method (str): detector method ("sift" or "fast").
        keypoint_count (int): total keypoints per image (detector + random fill).
        rewrite_existing (bool): if True, overwrite existing entries; if False, skip them.
        random_seed (int | None): RNG seed for the random-fill generator.
        map_size_bytes (int): LMDB map size for the output environment.
        log_every (int): how often (in images) to emit a progress log line.

    Returns:
        ExtractionStats: aggregated statistics for the run.
    """

    rng = np.random.default_rng(random_seed)
    stats = ExtractionStats()

    input_env = lmdb.open(
        input_lmdb_path,
        readonly=True,
        lock=False,
        readahead=False,
    )

    output_env = lmdb.open(
        output_lmdb_path,
        map_size=map_size_bytes,
        subdir=True,
        readonly=False,
        lock=True,
        create=True,
    )

    try:
        with input_env.begin(write=False) as input_txn, output_env.begin(write=True) as output_txn:

            cursor = input_txn.cursor()

            try:
                # keys from the cursor are already bytes - no .encode() needed
                for key, raw_bytes in cursor:

                    # resume mode: don't touch keys that are already in the output LMDB
                    if not rewrite_existing and output_txn.get(key) is not None:
                        stats.skipped += 1
                        continue

                    gray = decode_grayscale_image(raw_bytes)

                    if gray is None:
                        logger.warning(f"Failed to decode image for key: {key!r}")
                        stats.errors += 1
                        continue

                    keypoints, split_index = build_keypoint_record(gray, method, keypoint_count, rng)

                    record_bytes = serialize_record(
                        keypoints=keypoints,
                        split_index=split_index,
                        method=method,
                        image_shape=(int(gray.shape[0]), int(gray.shape[1])),
                    )

                    output_txn.put(key, record_bytes, overwrite=True)

                    stats.processed += 1
                    _update_detection_stats(stats, split_index, keypoint_count)

                    if stats.processed % log_every == 0:
                        logger.info(f"Progress: processed={stats.processed:,} | skipped={stats.skipped:,} | errors={stats.errors:,}")

            except KeyboardInterrupt:
                # catching here (instead of letting it propagate out of the `with`) lets the write transaction commit normally so all work done so far is preserved
                stats.interrupted = True
                logger.warning(
                    f"KeyboardInterrupt received after {stats.processed:,} images. "
                    f"Committing partial progress and printing summary..."
                )

        output_env.sync()

    finally:
        output_env.close()
        input_env.close()

    return stats


def log_summary(stats: ExtractionStats, keypoint_count: int, method: str) -> None:

    """
    Emit a concise human-readable summary of the extraction run.

    Parameters:
        stats (ExtractionStats): collected statistics.
        keypoint_count (int): target total keypoints per image (used in labels only).
        method (str): detector method used (used in labels only).
    """

    total = stats.processed

    if stats.interrupted:
        logger.warning("Run was interrupted (Ctrl+C) - the summary below reflects partial progress.")

    logger.info(f"Processed: {stats.processed:,} | Skipped: {stats.skipped:,} | Errors: {stats.errors:,}")

    # avoid division-by-zero if nothing was processed (e.g. empty input LMDB or all-skipped resume)
    if total == 0:
        logger.info("No images were processed - detection statistics are unavailable.")
        return

    full_pct = 100.0 * stats.fully_detected / total
    partial_pct = 100.0 * stats.partially_detected / total
    empty_pct = 100.0 * stats.empty_detected / total

    logger.info(
        f"Fully detected ({keypoint_count}/{keypoint_count}):            "
        f"{stats.fully_detected:,} / {total:,} ({full_pct:.1f}%)"
    )
    logger.info(
        f"Partial detection:                   "
        f"{stats.partially_detected:,} / {total:,} ({partial_pct:.1f}%)"
    )
    logger.info(
        f"Empty (all random fill):             "
        f"{stats.empty_detected:,} / {total:,} ({empty_pct:.1f}%)"
    )

    if stats.total_target_keypoints > 0:
        coverage_pct = 100.0 * stats.total_detected_keypoints / stats.total_target_keypoints
        logger.info(
            f"{method.upper()} coverage: "
            f"{stats.total_detected_keypoints:,} / {stats.total_target_keypoints:,} "
            f"target keypoints ({coverage_pct:.1f}%)"
        )


def main() -> None:

    """
    Script entry point: parses CLI arguments, runs the extraction over the input LMDB, and logs a summary of the run.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = parse_args()

    logger.info(
        f"Extracting {args.method.upper()} keypoints: "
        f"{args.input_lmdb} -> {args.output_lmdb} "
        f"(keypoint_count={args.keypoint_count}, rewrite_existing={args.rewrite_existing}, seed={args.random_seed})"
    )

    stats = process_lmdb(
        input_lmdb_path=args.input_lmdb,
        output_lmdb_path=args.output_lmdb,
        method=args.method,
        keypoint_count=args.keypoint_count,
        rewrite_existing=args.rewrite_existing,
        random_seed=args.random_seed,
        map_size_bytes=args.map_size_bytes,
        log_every=args.log_every,
    )

    log_summary(stats, keypoint_count=args.keypoint_count, method=args.method)


if __name__ == "__main__":
    main()
