#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=/scratch/gpfs/mk35/context-prediction/logs/perm-regression-%j.out

#SBATCH --job-name perm_regression
#SBATCH --time 11:30:00
#SBATCH --mem=50000
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Set up the environment
module load anaconda3
conda activate mybrainiak_v11
#analyses='KLDivergence'
analyses=$1
nfeat=$2
model=$3
stat=$4
remove_func_words=$5

echo  ${analyses} $nfeat $model $stat
# Run the python script
srun --mpi=pmi2 python permute_regression.py  --analyses ${analyses} --nfeat ${nfeat} --model ${model} --stat ${stat} --remove-func-words ${remove_func_words} 