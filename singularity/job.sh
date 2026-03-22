#!/bin/bash
#PBS -N knn_run
#PBS -l select=1:ncpus=1:mem=4gb:ngpus=0:gpu_mem=4gb:scratch_local=1gb
#PBS -l walltime=00:00:15
#PBS -o /storage/brno2/home/xklajb00/knn_job_log_latest.out
#PBS -e /storage/brno2/home/xklajb00/knn_job_log_latest.err


# >>> EDIT REQUIRED <<<
# edit the paths above ^ to match reality (for PBS -o and PBS -e)
# this will make sure no extra metacentrum stdout and stderr capture files will be created

# =========================================
# switch between CPU and GPU mode

# >>> EDIT REQUIRED <<<
USE_GPU=0 # set to 1 to enable GPU support (do not forget to change ngpus job parameter as well)

# =========================================
# define paths (edit to match reality)

# >>> EDIT REQUIRED <<<
# home directory
HOME_DIR=/storage/brno2/home/xklajb00/

# >>> EDIT REQUIRED <<<
# project folder name in home directory
PROJ_FOLDER=knn-writer-identification

# >>> EDIT REQUIRED <<<
# singularity CPU image path
SING_CPU_IMAGE="${HOME_DIR}/containers/knn_container_cpu.sif"

# >>> EDIT REQUIRED <<<
# singularity GPU image path
SING_GPU_IMAGE="${HOME_DIR}/containers/knn_container_gpu.sif"

# singularity scrip path (script that will be executed in the singularity container)
SING_SCRIPT="${HOME_DIR}/${PROJ_FOLDER}/singularity/singularity_script.sh"

# path to source code folder
CODE_DIR="${HOME_DIR}/${PROJ_FOLDER}/src"

# =========================================
# log and error files

CUR_TIME=$(date +%Y-%m-%d_%H-%M-%S)

# path to log directory
LOG_DIR="${HOME_DIR}/knn_job_logs/job_${CUR_TIME}"

# create log directory if it does not exist
mkdir -p "$LOG_DIR"

JOB_LOG_OUT="${LOG_DIR}/knn_job.out"

JOB_LOG_ERR="${LOG_DIR}/knn_job.err"

SING_LOG_OUT="${LOG_DIR}/knn_job.sing_out"

SING_LOG_ERR="${LOG_DIR}/knn_job.sing_err"

touch "$SING_LOG_OUT" "$SING_LOG_ERR" "$JOB_LOG_OUT" "$JOB_LOG_ERR"

# redirect all script outputs (stdout and stderr, excluding singularity outputs)
exec > "$JOB_LOG_OUT" 2> "$JOB_LOG_ERR"

# =========================================

echo "JOB START"

echo ""

echo "PBS job ID:   ${PBS_JOBID}"

echo "Node:         `hostname -f`"

echo "Scratch dir:  ${SCRATCHDIR}"

echo ""

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo "Variable SCRATCHDIR is not set!" >&2; exit 1; }

echo "Copying code to SCRATCH"
cp -R $CODE_DIR $SCRATCHDIR || { echo "Error while copying input file(s)!" >&2; exit 2; }

# set SINGULARITY variables for runtime data
export SINGULARITY_CACHEDIR=$HOME_DIR/singularity_cache
export SINGULARITY_TMPDIR=$HOME_DIR/singularity_tmp
export SINGULARITY_LOCALCACHEDIR=$SCRATCHDIR/local_cache

mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_LOCALCACHEDIR" "$SINGULARITY_CACHEDIR"

echo "Executing singularity script in container"

# echo "=================================="

# binding $SCRATCHDIR to /workspace
# executing $SING_SCRIPT inside the container
if [ "$USE_GPU" -eq 1 ]; then
    singularity exec --nv --bind "$SCRATCHDIR:/workspace" "$SING_GPU_IMAGE" "$SING_SCRIPT" >> "$SING_LOG_OUT" 2>> "$SING_LOG_ERR"
else
    singularity exec --bind "$SCRATCHDIR:/workspace" "$SING_CPU_IMAGE" "$SING_SCRIPT" >> "$SING_LOG_OUT" 2>> "$SING_LOG_ERR"
fi

# echo "=================================="

echo "Finished singularity execution"

# >>> EDIT REQUIRED <<<
## here you should copy results from scratch dir to home dir
# echo "Copying results back to home"
# cp output.txt $HOME_FOLDER/... || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

echo "JOB END"

# clean the SCRATCH directory
clean_scratch
