#!/usr/bin/env bash


analyses_run='GPT2Embed-cosine' #KLDivergence
freq_type='None'
weight_by_freq='N'
remove_func_words='N'


for mw in 0; do # This is not used in any of the analyses, but is kept for consistency with the other scripts
# Run the python script

for drift in 3 5; do


echo `date`
echo  ${analyses_run} "Moving window = " ${mw} 
sbatch ./sbatch_create_df.sh  ${analyses_run} ${mw} ${weight_by_freq} ${drift} ${remove_func_words}
done
done
