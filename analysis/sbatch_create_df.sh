#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=/scratch/gpfs/mk35/context-prediction/logs/create-df-%j.out

#SBATCH --job-name create_df
#SBATCH --time 01:50:00
#SBATCH --mem=30000
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Set up the environment
module load anaconda3
conda activate mybrainiak_v11

analyses_run=$1
moving_window=$2
weight_by_freq=$3
drift=$4
remove_func_words=$5

echo  ${analyses_run} ${moving_window} ${weight_by_freq} ${drift}
# Run the python script
srun --mpi=pmi2 python create_analyses_df.py \
    --analyses ${analyses_run} \
    --moving-window ${moving_window} \
    --weight_by_freq ${weight_by_freq} \
    --drift ${drift} \
    --remove-func-words ${remove_func_words}