# run with:
#   SING_SCRIPT="${PWD}/singularity/test_model.sh" NOTE="test" bash ./singularity/job.sh

echo ""
echo "==== Testing model ===="

SPLIT="real"
DATA_DIR="/storage/plzen1/home/xkiszk00/data"
OUT_DIR="${DATA_DIR}/checkpoints/${SPLIT}/"

CHECKPOINT_DIR="/storage/brno2/home/xklajb00/my_checkpoints" # MODIFY
MODEL_NAME="resnet_single_600" # MODIFY

cd /workspace

python3 -m src.test_model \
    --gt-file-gallery "${DATA_DIR}/splits/${SPLIT}/tst-gallery.txt" \
    --gt-file-query "${DATA_DIR}/splits/${SPLIT}/tst-query.txt" \
    --sift-keypoints-lmdb "${DATA_DIR}/lmdb.patches" \
    --lmdb "${DATA_DIR}/lmdb.all" \
    --batch-size 64 \
    --num-workers 8 \
    --checkpoints-dir "${CHECKPOINT_DIR}" \
    --model-name "${MODEL_NAME}"
