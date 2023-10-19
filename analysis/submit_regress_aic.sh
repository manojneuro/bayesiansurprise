#!/usr/bin/env bash
for model in 'Lasso' ; do 
for stat in 'Pearson' ; do 

#for analyses_run in  'Simulated-Positive+Noise-5' 'Simulated-Positive+Noise-10' 'Simulated-Positive+Noise-20'; do # 'Simulated-Positive+Noise-50' 'Simulated-Positive+Noise-70' 'Simulated-Positive+Noise-90'; do
#for analyses_run in  "EntropyRatio_Average_Reynolds_drift3" "EntropyRatio_Average_Reynolds_drift5" "EntropyRatio_Average_Reynolds_drift10" ; do

#for analyses_run in  "SurpriseRatio_Average_Reynolds_drift3" "SurpriseRatio_Average_Reynolds_drift5" "SurpriseRatio_Average_Reynolds_drift10" ; do
#for analyses_run in   'GPT2Embed-cosineRatio_Average_Reynolds_drift3'  'GPT2Embed-cosineRatio_Average_Reynolds_drift10'; do
#for analyses_run in   'KLDivergenceRatio_Average_Reynolds_drift3'  'KLDivergenceRatio_Average_Reynolds_drift5' 'KLDivergenceRatio_Average_Reynolds_drift10'; do
for analyses_run in 'KLDivergence' 'GPT2Embed-cosine' 'Surprise' 'Entropy' ; do


# Run the python script

echo `date`
echo ${analyses_run} "Features =" ${nfeat} "model=" ${model} "stat =" ${stat}
sbatch ./sbatch_regression_aic.sh ${analyses_run}  ${model} ${stat}
done
done
done