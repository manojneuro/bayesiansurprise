#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=/scratch/gpfs/mk35/context-prediction/logs/regression-aic-%j.out

#SBATCH --job-name regression_aic
#SBATCH --time 1:02:00
#SBATCH --mem=50000
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Set up the environment
module load anaconda3
conda activate mybrainiak_v11

analyses=$1
model=$2 
stat=$3

echo  ${analyses} $nfeat $model $stat
# Run the python script
srun --mpi=pmi2 python regression_aic.py  --analyses ${analyses}  --model ${model} --stat ${stat}