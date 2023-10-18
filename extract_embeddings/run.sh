#!/usr/bin/env bash

#SBATCH --time=02:00:00
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
STORY_NAME=monkey

python code/extract-gpt2-embeddings.py \
    --save-hidden-states \
    --use-previous-state \
    --model-name gpt2-xl \
    --context-length 1024 \
    --story-name ${STORY_NAME} \
    --sentence-file data/podcast-transcription.txt \
    --datum-file data/podcast-datum-cloze.csv

python code/do-pca.py \
    --k 50 \
    --story-name ${STORY_NAME} \
    results/${STORY_NAME}/${STORY_NAME}gpt2-xl-c_1024-layer_0.csv \
    results/${STORY_NAME}/${STORY_NAME}gpt2-xl-c_1024-layer_0_pca50d.csv

echo 'Run end time:' `date`
