
# Singularity for Metacentrum - Guide


## File description
* `knn_container.def`
    * definition file **template** to build the Singularity container
    * `{{REQUIREMENTS_FILE}}` must be filled in if the user wants to use it without `build_singularity.sh`
* `build_singularity.sh`
    * bash script to automatically build the Singularity container
* `job.sh`
    * Metacentrum job batch script
    * runs `singularity_script.sh` inside Singularity container
    * must be edited to match local paths
* `singularity_script.sh`
    * script to be run inside Singularity container in `job.sh` execution
    * here you write all your commands you want to execute in the Singularity container

---
**Note:** All of the following commands must be run from project root.

## Building singularity container

To build CPU singularity container run following command from project root:

`sudo bash singularity/build_singularity.sh "cpu"`

To build GPU (CUDA 13.0 support) singularity container run following command from project root:

`sudo bash singularity/build_singularity.sh "gpu"`

The created container will be located at `singularity/knn_container_cpu.sif` or `singularity/knn_container_gpu.sif`, depending on your previous CPU vs GPU choice.

---
**Note:** Make sure you have at least 20 GB of free space to build the container. The resulting container will be approximately 3 GB.

**Note:** `requirements_cpu.txt` and `requirements_gpu.txt` have to exist in project root.


## Running code on Metacentrum in Singularity container

1. You must have Singularity `.sif` containers somewhere in your Metacentrum home directory (e.g., in `.../home/login/knn-writer-identification/singularity/` or in `.../home/login/containers/`)
2. You must have `job.sh` and `singularity_script.sh` scripts in `.../home/login/knn-writer-identification/singularity/`
3. Modify `job.sh` script on required places (denoted by `# >>> EDIT REQUIRED <<<`). This includes:
    * modification of local paths to match reality
    * setting flag that enables / disables use of GPU (note, that you have to also modify PBS arguments to ask for GPU)
    * add custom copy commands, that copy what you need to `SCRATCHDIR`
        * code in `knn-writer-identification/src/` is already being copied by default
    * add custom copy commands, that copy program results from `SCRATCHDIR` to somewhere in your home directory (ideally in `$LOG_DIR`)
    * specify what you want to copy to scratch directory and what you want to copy back to home directory
4. Write commands you want to execute in the container in `singularity_script.sh`
    * the container has binded `$SCRATACHDIR` to `/workspace` directory inside the container -> everything that is in `$SCRATACHDIR` will be seen in this folder
    * to access `$SCRATACHDIR` simply write `cd /workspace`
    * next you can run any code you want (e.g. `python3 ./src/main.py`)
5. **!! Important !!** Make sure you have execute permissions for `singularity_script.sh`
    * otherwise use `chmod u+rwx ./singularity/singularity_script.sh`
5. Use command: `qsub ./singularity/job.sh` to submit job
6. After the job has finished, the run logs will be located in `LOG_DIR="${HOME_DIR}/knn_job_logs/job_${CUR_TIME}"`. This includes:
    * `knn_job.out` - stdout outputs of `job.sh`
    * `knn_job.err` - stderr outputs of `job.sh`
    * `knn_job.sing_out` - stdout outputs of `singularity_script.sh`
    * `knn_job.sing_err` - stderr outputs of `singularity_script.sh`

---
---
---

## Manually executing Python program in singularity container
`singularity run --nv -B $PWD:/workspace singularity/knn_container_cpu.sif train.py`
* executes default `%runscript` specified in Singularity `.def` with given parameter (`train.py`)
    * `%runscript` currently executes: `cd /workspace ; python3 "$@"`
* `--nv` = binds all `/dev/nvidia*` devices from the host into the container
* `-B <host_dir_path>:<container_dir_path>` = mounts specified host directory into specified container directory
    * `$PWD` = current host directory from which the command was executed

`singularity exec --nv -B $PWD:/workspace singularity/knn_container_cpu.sif bash -c "(cd /workspace; python3 train.py)"`
* alternative to `singularity run` command
* executes given command from containers root directory (`bash -c "(cd /workspace; python3 train.py)"`)

## Container repositories
* **DockerHub**: https://hub.docker.com/r/pytorch/pytorch/tags
* **Nvidia**: https://catalog.ngc.nvidia.com/?tab=container
