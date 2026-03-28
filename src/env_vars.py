# Directory paths and file paths for the project.
DATA_DIR = "/storage/plzen1/home/xkiszk00/data"
SPLITS_DIR = f"{DATA_DIR}/splits/real"
TOY_SPLITS_DIR = f"{DATA_DIR}/splits/toy"
TMP_DIR = "tmp"

# Data file paths
IAM_AUTHORS_FILE_PATH = f"{DATA_DIR}/iam.lines.all"
ICDAR_AUTHORS_FILE_PATH = f"{DATA_DIR}/icdar.lines.all"
INTERN_AUTHORS_FILE_PATH = f"{DATA_DIR}/intern.lines.all"

# Split file paths
TRAIN_SPLIT_FILE_PATH = f"{SPLITS_DIR}/train.txt"
VAL_SPLIT_FILE_PATH = f"{SPLITS_DIR}/val.txt"
TEST_SPLIT_FILE_PATH = f"{SPLITS_DIR}/test.txt"
QUERY_SPLIT_FILE_PATH = f"{SPLITS_DIR}/query.txt"
GALLERY_SPLIT_FILE_PATH = f"{SPLITS_DIR}/gallery.txt"

# Toy split file paths
TOY_TRAIN_SPLIT_FILE_PATH = f"{TOY_SPLITS_DIR}/train.txt"
TOY_VAL_SPLIT_FILE_PATH = f"{TOY_SPLITS_DIR}/val.txt"
TOY_TEST_SPLIT_FILE_PATH = f"{TOY_SPLITS_DIR}/test.txt"
TOY_QUERY_SPLIT_FILE_PATH = f"{TOY_SPLITS_DIR}/query.txt"
TOY_GALLERY_SPLIT_FILE_PATH = f"{TOY_SPLITS_DIR}/gallery.txt"

# LMDB path
LMDB_PATH = f"{DATA_DIR}/lmdb.all/"

# Random seed for reproducibility
NP_RANDOM_SEED = 42
