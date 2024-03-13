#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=sm3090el8
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=7-00:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH --gres=gpu:RTX3090:1 # Request 1 GPU (can increase for more)

source /home/energy/s222491/diff_env/bin/activate

module load Python/3.11.3-GCCcore-12.3.0
module load foss

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PROJECT_ROOT="/home/energy/s222491"
export SCRATCH_PATH="/home/scratch3/s222491"
export SCRATCH_COMPUTE_PATH="/home/scratch3/s222491"
export PYTHONPATH="$PYTHONPATH:/home/energy/s222491/om-diff"

python /home/energy/s222491/om-diff/src/train.py experiment=$1