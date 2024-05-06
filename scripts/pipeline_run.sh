#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --partition=xeon24
#SBATCH -N 1-1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --ntasks-per-node=8     # 8 MPI processes per node
#SBATCH --time=01:00:00 # 2 days of runtime (can be set to 7 days)

source /home/energy/s222491/diff_env/bin/activate

module load Python/3.11.3-GCCcore-12.3.0
module load foss

export PROJECT_ROOT="/home/energy/s222491"
export g16root="/home/modules/software/Gaussian"
source $g16root"/g16/bsd/g16.profile"
export GAUSS_SCRDIR="/home/energy/s222491"
export ASE_GAUSSIAN_COMMAND="/home/modules/software/Gaussian/g16/bsd/g16.profile"
export PYTHONPATH="$PYTHONPATH:/home/energy/s222491/om-diff"

python /home/energy/s222491/om-diff/src/vaskas_pipeline.py --samples_path=$1 --barrier_criteria=$2