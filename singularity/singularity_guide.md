
## Container repositories
* **DockerHub**: https://hub.docker.com/r/pytorch/pytorch/tags
* **Nvidia**: https://catalog.ngc.nvidia.com/?tab=container

## File description
* `singularity/knn_container.def` - definition file **template** to build the singularity container (`{{REQUIREMENTS_FILE}}` must be filled in if the user wants to use it without the following Bash script)
* `singularity/build_singularity.sh` - bash script to automatically build the singularity container

## Building singularity container

To build CPU singularity container run following command from project root:

`sudo bash singularity/build_singularity.sh "cpu"`

To build GPU (CUDA 13.0 support) singularity container run following command from project root:

`sudo bash singularity/build_singularity.sh "gpu"`

The created container will be located at `singularity/knn_container_cpu.sif` or `singularity/knn_container_gpu.sif`, depending on your previous choice.

---
Note: Make sure you have at least 20 GB of free space to build the container. The resulting container will be approximately 3 GB.

Note: `requirements_cpu.txt` and `requirements_gpu.txt` have to exist in project root.

## Singularity container

## Executing Python program in singularity container
`singularity run --nv -B $PWD:/workspace singularity/knn_container_cpu.sif train.py`
* executes default `%runscript` specified in Singularity `.def` with given parameter (`train.py`)
    * `%runscript` currently executes: `cd /workspace ; python3 "$@"`
* `--nv` = binds all `/dev/nvidia*` devices from the host into the container
* `-B <host_dir_path>:<container_dir_path>` = mounts specified host directory into specified container directory
    * `$PWD` = current host directory from which the command was executed

`singularity exec --nv -B $PWD:/workspace singularity/knn_container_cpu.sif bash -c "(cd /workspace; python3 train.py)"`
* alternative to `singularity run` command
* executes given command from containers root directory (`bash -c "(cd /workspace; python3 train.py)"`)

