#!/usr/bin/env bash
for model in 'Lasso' ; do 
for stat in 'Pearson' ; do 

#for analyses_run in 'KLDivergenceRatio_Average_Reynolds_drift3' 'KLDivergenceRatio_Average_Reynolds_drift5' 'KLDivergenceRatio_Average_Reynolds_drift10'  ; do 
#for analyses_run in 'GPT2Embed-cosineRatio_Average_Reynolds_drift3' 'GPT2Embed-cosineRatio_Average_Reynolds_drift5' 'GPT2Embed-cosineRatio_Average_Reynolds_drift10'  ; do 
#for analyses_run in 'SurpriseRatio_Average_Reynolds_drift3' 'SurpriseRatio_Average_Reynolds_drift5' 'SurpriseRatio_Average_Reynolds_drift10'  ; do 
#for analyses_run in 'EntropyRatio_Average_Reynolds_drift3' 'EntropyRatio_Average_Reynolds_drift5' 'EntropyRatio_Average_Reynolds_drift10'  ; do 
#for analyses_run in 'Surprise' 'Entropy' 'KLDivergence' 'GPT2Embed-cosine' ; do
for analyses_run in  'GPT2Embed-cosine'; do


for nfeat in   200 ; do

# Run the python script
remove_func_words='N'
echo `date`
echo ${analyses_run} "Features =" ${nfeat} "model=" ${model} "stat =" ${stat} "Remove Func Words =" ${remove_func_words}
sbatch ./sbatch_regression_perm.sh ${analyses_run} ${nfeat} ${model} ${stat} $remove_func_words
done
done
done
done