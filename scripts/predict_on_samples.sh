#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=sm3090el8
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=08:00:00 # 2 days of runtime (can be set to 7 days)
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

python /home/energy/s222491/om-diff/src/predict_on_samples.py \
--predictor_dir_path="/home/scratch3/s222491/logs/train/runs/2024-03-14_12-20-54/" \
--samples_path="/home/scratch3/s222491/logs/train/runs/2024-03-13_12-28-42/" \
--device="cuda" \
--file_suffix="unconditional_test"