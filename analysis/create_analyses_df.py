import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np 

from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.io
from scipy.io import loadmat
import pandas as pd

import matplotlib.style as style 
from matplotlib.offsetbox import AnchoredText
from scipy.spatial.distance import cdist,mahalanobis
from scipy.stats import wasserstein_distance, pearsonr, entropy
from scipy.signal import correlate, correlation_lags

from tqdm import tqdm

import imp
import time
from timeit import default_timer as timer
import matplotlib.patches as patches
import pickle as pickle
import os
import argparse

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('autosave', '5')
sns.set(style = 'whitegrid', context='talk', rc={"lines.linewidth": 2.5})
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.patches as patches
mpl.use('Agg')

from sklearn.linear_model import LinearRegression as lR
#%matplotlib inline 
#%autosave 5
import context_helper
imp.reload(context_helper)
from context_helper import folders,   read_embedding, compute_autocorr, 
from context_helper import   get_human_bounds, compute_cross_corr, pad_df
from context_helper import   read_embedding

from wordfreq import zipf_frequency
#%matplotlib inline 
#%autosave 5

import surprise_helper


from surprise_helper import   average_surp_reynolds
from surprise_helper import compute_cross_entr, make_random, compute_embed_dist


model='GPT2'

home_dir = folders['tiger']
save_plot='Y'
results_dir= home_dir + 'results/'
log_dir = home_dir + 'logs/'

parser = argparse.ArgumentParser()


parser.add_argument('--analyses', type=str, required=True)
parser.add_argument('--moving-window', type=int, required=True)
parser.add_argument('--weight_by_freq', type=str, required=True)
parser.add_argument('--drift', type=int, required=True)
parser.add_argument('--remove-func-words', type=str, required=True)

args= parser.parse_args()
print(args)

smooths=[0]

running_win=args.moving_window
remove_func_words=args.remove_func_words
use_precomputed=True


stories=['monkey','pieman','tunnel']

#z_windows=[50]

analyses=args.analyses


keep_word_order='N'


plot_figs='N'

for smooth in smooths:

    
    for i,pod_name in enumerate(stories):
        print(i, pod_name)
        #counter=0
        embed_file=home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) \
            +'%sgpt2-xl-c_1024-layer_0_pca50d.csv' % pod_name
        datum_file= home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) + 'datum.csv'
        surprise_file= home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) + '%sgpt2-xl-c_1024_surp_entr.csv' % pod_name
        sentence_id_file = home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) + '%s_sentence_file.csv' % pod_name
        df_sentence = pd.read_csv(sentence_id_file)
                #df_embed=pd.read_csv(embed_file,  sep=',')
        datum_data=pd.read_csv(datum_file,  sep=',', header=(0), usecols=range(4))
        df_entr_sur=pd.read_csv(surprise_file,  sep=',', header=(0))

        #Option to use precomputed file to add extra analyses

        if use_precomputed:
            if remove_func_words=='N':
                precomputed_file=home_dir + '/outputs/%s_processed_df.csv' % pod_name
            else:
                precomputed_file=home_dir + '/outputs/%s_processed_df_non-func.csv' % pod_name

            df_precomputed=pd.read_csv(precomputed_file, sep=',', header=(0))
            df_entr_surp_addedevents=df_precomputed
        else:
        #print(df_entr_sur.shape,df_entr_sur.columns)
            if pod_name=='monkey':
                onset=datum_data['onset']/512
            else:
                onset=datum_data['onset']
            df_entr_sur['onsets']=onset
            
            df_entr_sur.drop(columns=['Unnamed: 0'], inplace=True)
            
            
            analyses_str= f"{analyses}"

            if remove_func_words=='Y':
                df_entr_sur['Stop Word']=df_sentence['Stop Word']
                df_entr_sur_nonfunc =df_entr_sur.copy()
                #df_entr_sur_nonfunc['Stop Word']=df_sentence['Stop Word']
                analyses_str=analyses_str+'non-func'
                
                if keep_word_order=='N':
                    df_filtered=df_entr_sur_nonfunc.loc[df_entr_sur_nonfunc['Stop Word']==0]
                    #print(df_filtered.columns,df_entr_sur.columns)
                    df_in=df_filtered
                    df_in['Sentence ID']=df_sentence['Sentence ID'].loc[df_sentence['Stop Word']==0]

                df_entr_sur=df_in

            # Compute K-L Divergence        
            df_entr_sur=compute_cross_entr(df_entr_sur)       

            print(df_entr_sur.shape)
            
        
                
            df_in=df_entr_sur
            df_in['Sentence ID']=df_sentence['Sentence ID'] 

            df_embed,_ = read_embedding(pod_name, model,'Y') # use the pca embeddings
            if remove_func_words=='N':
                cos_dist=compute_embed_dist(df_embed.iloc[1:,:])
                df_in['GPT2Embed-cosine']=cos_dist
            else:
                df_non_func_embed = df_embed.iloc[df_filtered.index]
                cos_dist=compute_embed_dist(df_non_func_embed) # already filtered so 0th row is taken out
                df_in['GPT2Embed-cosine']=cos_dist
              
            # Compute ratio with Average Surprise Reynold's et al. 2007
            df_in=average_surp_reynolds(df_in, args)   
            
            # Pad values for moving window analysis
            
            df_entr_sur_addedevents=pad_df(df_entr_surp_addedevents)
        # Compute distance between embeddings
      
        df_embed,_ = read_embedding(pod_name, model,'Y') # use the pca embeddings
        if remove_func_words=='N':
            cos_dist=compute_embed_dist(df_embed.iloc[1:,:])
            df_entr_surp_addedevents['GPT2Embed-cosine']=cos_dist



        #Recompute Reynolds averaging if it does not already exist
        df_entr_surp_addedevents=average_surp_reynolds(df_entr_surp_addedevents, args)   

         # Simulate random numbers for random noise signal

        #df_entr_surp_addedevents=make_random(args,i,df_entr_surp_addedevents)


        # for running_win in z_windows:
        #     analyses_str= args.analyses + 'RunZ_{0}'.format(running_win)
        #     df_entr_surp_addedevents[analyses_str]=np.nan
        # #Weighted or un-weighted Z computation is performed based on args
        #     df_entr_surp_addedevents = compute_running_avg_entr(df_entr_surp_addedevents,running_win,args)
        # #analyses_str= analyses_str +'using_running_win%i' % running_win
        
        df_entr_sur=df_entr_surp_addedevents
        
 
        if remove_func_words=='N':
            df_entr_sur.to_csv(home_dir + '/outputs/%s_processed_df.csv' % pod_name)
        else:
             df_entr_sur.to_csv(home_dir + '/outputs/%s_processed_df_non-func.csv' % pod_name)