#!/bin/bash

# Pars args
if [ "$1" == "cpu" ]; then
    REQ_FILE="requirements_cpu.txt"
    IMAGE_FILE="singularity/knn_container_cpu.sif"
elif [ "$1" == "gpu" ]; then
    REQ_FILE="requirements_gpu.txt"
    IMAGE_FILE="singularity/knn_container_gpu.sif"
else
    echo "Usage: $0 [cpu|gpu]"
    exit 1
fi

DEF_TEMPLATE="singularity/knn_container.def"
DEF_TMP="singularity/tmp_knn_container.def"

# Remove old tmp/cache dirs
rm -rf "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
rm "$DEF_TMP"

# Fill in the correct requirements file path to the .def file template
sed "s|{{REQUIREMENTS_FILE}}|$REQ_FILE|" "$DEF_TEMPLATE" > "$DEF_TMP"

# Temporary tmp/cache dirs for this singularity build
export SINGULARITY_TMPDIR="$PWD/singularity/tmp"
export SINGULARITY_CACHEDIR="$PWD/singularity/cache"

mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

# Build the .sif singularity file
sudo -E singularity build "$IMAGE_FILE" "$DEF_TMP"

# Clean tmp/cache dirs
rm -rf "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
rm "$DEF_TMP"
