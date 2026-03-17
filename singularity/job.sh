#!/bin/bash
#PBS -N knn_run
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=1:gpu_mem=32gb:scratch_local=1gb
#PBS -l walltime=00:02:00

# =========================================
# define paths (change to match reality)

# home directory
HOME_DIR=/storage/brno2/home/xklajb00/

# singularity image path
SING_IMAGE="${HOME_DIR}/knn/singularity/knn_container_gpu.sif"

# singularity scrip path (script that will be executed in the singularity container)
SING_SCRIPT="${HOME_DIR}/knn/singularity/singularity_script.sh"

# path to source code folder
CODE_DIR="${HOME_DIR}/knn/src"

# path to log directory
LOG_DIR="${HOME_DIR}/logs"

# create log directory if it does not exist
mkdir -p "$LOG_DIR"

# =========================================
# log file

CUR_TIME=$(date +%Y%m%d_%H%M%S)

LOG_FILE="${LOG_DIR}/job_${CUR_TIME}.log"

# =========================================

echo "JOB START" > $LOG_FILE

echo "" >> $LOG_FILE

echo "PBS job ID:   ${PBS_JOBID}" >> $LOG_FILE

echo "Node:         `hostname -f`" >> $LOG_FILE

echo "Scratch dir:  ${SCRATCHDIR}" >> $LOG_FILE

echo "" >> $LOG_FILE

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

echo "Copying code to SCRATCH" >> $LOG_FILE
cp -R $CODE_DIR $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

# set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOME_DIR
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR

echo "Executing singularity script in container" >> $LOG_FILE

echo "==================================" >> $LOG_FILE

# binding $SCRATCHDIR to /workspace
# executing $SING_SCRIPT inside the container
singularity exec --nv --bind $SCRATCHDIR:/workspace $SING_IMAGE $SING_SCRIPT >> $LOG_FILE 2>&1

echo "==================================" >> $LOG_FILE

# here you should copy results from scratch dir to home dir
# echo "Copying results" >> $PROJECT_DIR/jobs_info.txt
# cp test.out $PROJECT_DIR/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

echo "JOB END" >> $LOG_FILE

# clean the SCRATCH directory
clean_scratch
