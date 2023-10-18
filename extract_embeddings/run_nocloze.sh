#!/usr/bin/env bash

#SBATCH --time=01:40:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o 'logs/%A.log'

set -e

module load anaconda
conda activate 247-main

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Run start time:' `date`

#export TRANSFORMERS_OFFLINE=1
STORY_NAME=tunnel
clength=1024
model=gpt2-xl
python code/extract-gpt2-embeddings_nocloze.py \
    --save-hidden-states \
    --use-previous-state \
    --model-name ${model} \
    --context-length ${clength} \
    --story-name ${STORY_NAME} \
    --sentence-file data/${STORY_NAME}_transcript.txt \
    --datum-file data/${STORY_NAME}Aligned.txt

python code/do-pca.py \
    --k 50 \
    results/${STORY_NAME}/${STORY_NAME}gpt2-xl-c_1024-layer_0.csv \
    results/${STORY_NAME}/${STORY_NAME}gpt2-xl-c_1024-layer_0_pca50d.csv

echo 'Run end time:' `date`
