#!/bin/bash

echo "==== Host and kernel ===="
uname -a

echo ""
echo "==== Memory usage ======="
free -h

echo ""
echo "==== CPU info ==========="
lscpu | grep -Ev '^(Flags:|Vulnerability)'

echo ""
echo "==== GPU info ==========="
module load nvidia
nvidia-smi

echo ""
echo "==== Running program ===="

cd  /workspace
ls -l
#python3 ./main.py
