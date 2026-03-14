import os
import cv2
import lmdb
import logging
import argparse
import numpy as np
from typing import Any
from PIL import Image
import xml.etree.ElementTree as ET


logger = logging.getLogger(__name__)

IAM_INDEX_FILE = "iam.lines.all"


def parse_args() -> argparse.Namespace:

    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        usage="Extracts lines from IAM pages and stores into lmdb file. \n" +
              "The pages are stored in the images and xmls paths. Each image (image of a page containing lines of written text) in the images path has its corresponding XML file. \n" +
              "This XML file has coordinates for where each line is located at the certain page. \n" +
              "It is also required to have a mapping file, mapping which page belongs to which author (page -> writer_id)"
    )

    parser.add_argument("--images-path", help="Path to a directory containing image (.png) files.", type=str, required=True)
    parser.add_argument("--xmls-path", help="Path to a directory containing PAGE-XML (.xml) files.", type=str, required=True)
    parser.add_argument("--map-file", help="File (.txt) mapping page to its writer ID.", type=str, required=True)
    parser.add_argument("--lmdb", help="Path to LMDB directory.", type=str, required=True)
    parser.add_argument("--vpad", help="Additional vertical padding for each extracted line", type=int, required=False, default=10)
    parser.add_argument("--jpg-quality", help="JPEG quality for stored line images.", type=int, required=False, default=95)

    return parser.parse_args()


def parse_points(points_str: str) -> list[tuple[int, int]]:

    """
    Convert PAGE-XML points string: 'x1,y1 x2,y2 x3,y3 ...' into a list of (x, y) tuples.

    Parameters:
        points_str (str): PAGE-XML points string

    Returns:
        list[tuple[int, int]]: list of (x, y) tuples
    """

    points = []

    for point_pair in points_str.strip().split():
        x, y = point_pair.split(",")
        points.append((int(x), int(y)))

    return points


def get_namespace(root: ET.Element) -> dict[str, str]:

    """
    Extract the XML namespace from the PAGE-XML root element.

    Parameters:
        root (ET.Element): Root element of the parsed PAGE-XML document.

    Returns:
        dict[str, str]: Namespace mapping usable in ElementTree searches.
    """

    if root.tag.startswith("{"):
        return {"pc": root.tag.split("}")[0].strip("{")}

    return {"pc": ""}


def process_map_file(mapping_file: str) -> dict[str, str] | None:

    """
    Loads the IAM forms mapping file and build a mapping from page/form ID to writer ID.

    The expected file format is the IAM `forms.txt` format, where each non-comment line begins with:
        <page_id> <writer_id> ...
    Example:
        a01-000u 000 2 prt 7 5 52 36

    Parameters:
        mapping_file (str): Path to the mapping text file.

    Returns:
        dict[str, str] | None:
            A dictionary mapping page IDs (for example `a01-000u`) to
            writer IDs (for example `000`) if the file is valid,
            otherwise None.
    """

    if not os.path.exists(mapping_file) or not os.path.isfile(mapping_file) or not mapping_file.endswith(".txt"):
        logger.error("Mapping file path does not exist or is not a file, or not .txt file.")
        return None

    mapped_pages_with_writers = {}

    with open(mapping_file, 'r', encoding="utf-8") as f:

        for raw_line in f:

            line = raw_line.strip()

            if not line or line.startswith("#"):  # skip empty lines
                continue

            parts = line.split()
            if len(parts) < 2:
                logger.warning(f"Skipping malformed mapping line: {raw_line!r}")
                continue

            page_id, writer_id = parts[0], parts[1]

            if page_id in mapped_pages_with_writers:
                logger.warning(f"Duplicate page ID in mapping file, skipping: {page_id}")
                continue

            mapped_pages_with_writers[page_id] = writer_id

    return mapped_pages_with_writers


def extract_images_lines_with_metadata(
    image_path: str,
    xml_path: str,
    page_id: str,
    writer_id: int,
    vpad: int = 10
) -> list[dict[str, Any]] | None:

    """
    Extract individual text-line crops from a page image using coordinates stored in its corresponding PAGE-XML file.

    For each `TextLine` element in the XML, the function reads the coordinates, computes a bounding box, optionally expands it vertically
    by `vpad`, and crops the matching region from the page image.

    Each extracted line is returned together with its metadata:
        writer ID, page ID, line ID, transcription text, and cropped PIL image.

    Parameters:
        image_path (str): Path to the page image (.png).
        xml_path (str): Path to the PAGE-XML file (.xml) corresponding to the page.
        page_id (str): Page/form identifier.
        writer_id (int): Numeric writer identifier assigned to the page.
        vpad (int, optional): Extra vertical padding added above and below each line bounding box. Defaults to 10.

    Returns:
        list[dict[str, Any]] | None:
            A list of extracted line records, or None if the image/XML file is missing or invalid.
    """

    if not os.path.exists(image_path) or not os.path.isfile(image_path) or not image_path.endswith(".png"):
        logger.warning(f"Image path does not exist or is not a .png file: {image_path}")
        return None

    if not os.path.exists(xml_path) or not os.path.isfile(xml_path) or not xml_path.endswith(".xml"):
        logger.warning(f"XML path does not exist or is not a .xml file: {xml_path}")
        return None

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = get_namespace(root)

    # load the image
    image_page = Image.open(image_path).convert("RGB")

    lines = []

    for line_elem in root.findall(".//pc:TextLine", ns):  # go through each TextLine elem

        line_id = line_elem.attrib.get("id", "")

        # get coordinates of the line and what's actually written
        coords_elem = line_elem.find("pc:Coords", ns)
        text_elem = line_elem.find("pc:TextEquiv/pc:Unicode", ns)

        if coords_elem is None:  # no coords - skip
            continue

        # extract the coordinate points
        points_attr = coords_elem.attrib.get("points")
        if not points_attr:
            continue

        # parse the points
        points = parse_points(points_attr)
        if not points:
            continue

        xs, ys = [x for x, _ in points], [y for _, y in points]
        x0, y0, x1, y1 = min(xs), min(ys) - vpad, max(xs), max(ys) + vpad

        # crop the image page to get only the page's line
        image_line = image_page.crop((x0, y0, x1, y1))

        lines.append({
            "writer_id": writer_id,
            "page_id": page_id,
            "line_id": line_id,
            "text": text_elem.text.strip() if text_elem is not None and text_elem.text else "",
            "image_line": image_line,   # keep PIL image in memory
        })

    return lines


def pil_to_jpg_bytes(image: Image.Image, jpg_quality: int = 95) -> bytes | None:

    """
    Encode a PIL image into JPEG bytes for LMDB storage.

    Parameters:
        image (Image.Image): Input PIL image.
        jpg_quality (int): JPEG quality used for encoding.

    Returns:
        bytes | None: Encoded JPEG bytes, or None if encoding fails.
    """

    image_np = np.array(image)

    if image_np.ndim != 3 or image_np.shape[2] != 3:
        logger.warning("Expected RGB image, got shape: %s", image_np.shape)
        return None

    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    )

    if not success:
        logger.warning("Image was not JPEG encoded, skipping.")
        return None

    return encoded.tobytes()

def write_lines_to_lmdb(
    lmdb_path: str,
    lines: list[dict[str, Any]],
    jpg_quality: int = 95,
    map_size: int = 2 ** 40
) -> tuple[int, int]:

    """
    Store extracted line images into LMDB.
    Each line is stored under key: <page_id>-<line_id>.jpg

    If the key already exists, its value is updated.

    Parameters:
        lmdb_path (str): Path to LMDB directory.
        lines (list[dict[str, Any]]): Extracted line records.
        jpg_quality (int): JPEG quality for encoded images.
        map_size (int): LMDB map size in bytes.

    Returns:
        tuple[int, int]: Number of successfully written samples and number of skipped samples.
    """

    written, skipped = 0, 0

    env = lmdb.open(
        lmdb_path,
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        create=True,
    )

    with env.begin(write=True) as txn:

        for line_sample in lines:

            page_id, line_id, image_line = str(line_sample["page_id"]), str(line_sample["line_id"]), line_sample["image_line"]

            key_str = f"{page_id}-{line_id}.jpg"

            if not isinstance(image_line, Image.Image):
                logger.warning(f"Skipping non-PIL image for {key_str}")
                skipped += 1
                continue

            key = key_str.encode("utf-8")

            image_value = pil_to_jpg_bytes(image_line, jpg_quality=jpg_quality)
            if image_value is None:
                logger.warning(f"Failed to encode line image for key: {key_str}")
                skipped += 1
                continue

            # store into lmdb
            txn.put(key, image_value, overwrite=True)
            written += 1

    env.sync()
    env.close()

    return written, skipped

def write_index_file(lines: list[dict[str, Any]]) -> None:

    """
    Write IAM index file in format: <image_key> <writer_id> <text>

    Parameters:
        lines (list[dict[str, Any]]): Extracted line records.
    """

    with open(IAM_INDEX_FILE, 'w', encoding="utf-8") as f:
        for line in lines:
            writer_id = int(line["writer_id"])
            page_id = str(line["page_id"])
            line_id = str(line["line_id"])
            text = str(line["text"])

            image_line_key = f"{page_id}-{line_id}.jpg"
            f.write(f"{image_line_key} {writer_id} {text}\n")


def main() -> None:

    """
    Parse command-line arguments, validate input paths, load the IAM page-to-writer mapping, and
     extract line crops with metadata from all mapped PAGE-XML pages.
    """

    # basic logger config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    args = parse_args()  # parse arguments

    # check for paths
    if not os.path.exists(args.images_path) or not os.path.isdir(args.images_path):
        logger.error("Images path does not exist or is not a directory")
        return

    if not os.path.exists(args.xmls_path) or not os.path.isdir(args.xmls_path):
        logger.error("XMLs path does not exist or is not a directory")
        return

    if not os.path.exists(args.map_file) or not os.path.isfile(args.map_file):
        logger.error("Mapping file does not exist or is not a file")
        return

    # get mapping
    mapped_pages_with_writers = process_map_file(args.map_file)
    if mapped_pages_with_writers is None:
        return

    all_lines = []       # all the lines
    processed_pages = 0  # number of processed pages (pages from which all lines were extracted)

    for page_id, writer_id in mapped_pages_with_writers.items():  # go through each page and extract its lines

        lines_of_page = extract_images_lines_with_metadata(
            image_path=os.path.join(args.images_path, f"{page_id}.png"),
            xml_path=os.path.join(args.xmls_path, f"{page_id}.xml"),
            page_id=page_id,
            writer_id=int(writer_id) + 10_000,  # + 10000, to make sure the writer ID is unique across ALL datasets
            vpad=args.vpad
        )


        if lines_of_page is None:
            continue

        all_lines.extend(lines_of_page)
        processed_pages += 1

    logger.info(f"Extracted {len(all_lines)} lines from {processed_pages}/{len(mapped_pages_with_writers)} pages.")

    # write the extracted lines into lmdb database
    written, skipped = write_lines_to_lmdb(
        lmdb_path=args.lmdb,
        lines=all_lines,
        jpg_quality=args.jpg_quality,
    )

    logger.info(f"LMDB write complete: {written} written, {skipped} skipped.")

    # write also the lines (key, writer_id, text) into an index file
    write_index_file(all_lines)

    logger.info(f"IAM index \"{IAM_INDEX_FILE}\" file created at the root")


if __name__ == "__main__":
    main()
