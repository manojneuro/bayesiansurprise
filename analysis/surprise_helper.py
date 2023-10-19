import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
from scipy.io import loadmat
import brainiak

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from tqdm import tqdm
from numpy.random import RandomState
import matplotlib.style as style 
sns.set(style = 'whitegrid', context='poster', rc={"lines.linewidth": 2.5})

import matplotlib
import matplotlib.patches as patches
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.use('Agg')
from scipy.spatial.distance import cdist,mahalanobis
from scipy.stats import wasserstein_distance, pearsonr, entropy, ttest_ind, ks_2samp
from scipy import stats
from scipy.special import softmax
from scipy.signal import correlate, correlation_lags

import pickle as pickle
from scipy.interpolate import UnivariateSpline
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.model_selection import TimeSeriesSplit
from context_helper import folders
from sklearn.linear_model import LinearRegression as lR
import ast

home_dir=folders['tiger']

def assign_event_id(human_boundaries, df_entr_sur_in):  
    """Assign event id to each word
     """

    df_entr_sur_in['HumanEvent_ID']=np.nan
    #print(len(df_entr_sur_in),len(human_boundaries))

    for j in range(len(human_boundaries)):
        
        #print(j)


        if j==0:
            start_indx=int(0) # the first value i.e. the value after padding the dataframe.
        else:
            start_indx=end_indx +1
        #print(type(start_indx))
        indx_find=np.argmax(df_entr_sur_in['onsets']> human_boundaries[j]).astype(int) #argmax only returns the first value.
        #Use this value to find the word before the button press.
        indx_near = indx_find-1
        
        end_indx=indx_near
        #print(j, start_indx, end_indx)
        if end_indx > len(df_entr_sur_in):
            #index exceeds length of story
            df_entr_sur_in.loc[start_indx:,'HumanEvent_ID']=j
            
        if ((end_indx >0) & (end_indx < len(df_entr_sur_in))):
            #index is within the length of story
            df_entr_sur_in.loc[start_indx:round(end_indx), 'HumanEvent_ID']=j
        
       
    return  df_entr_sur_in


def word_frequency_factor(df_in):
    wfw_all=np.empty(len(df_in))
    wfw_all[:]=np.nan
    w_idf=np.empty(len(df_in))
    counter=0
    doc_freq=df_in['Target Word'].str.lower().groupby(df_in['Target Word'].str.lower()).count()
    freq_df=doc_freq.to_frame()
    freq_df.rename(columns={'Target Word':'Count'}, inplace=True)
    freq_df.reset_index(inplace=True)
    
    #Compute idf
    idf=df_in.groupby(df_in['Target Word'].str.lower())['Sentence ID'].count()
    num_sent=df_in['Sentence ID'].max()
    idf=-np.log(idf/num_sent)
    idf=idf.to_frame()
    idf.rename(columns={'Sentence ID':'idf'}, inplace=True)
    idf.reset_index(inplace=True)
    
    for word in df_in['Target Word'].str.lower():
        wfw=freq_df['Count'].loc[freq_df['Target Word'].str.lower()==word].to_numpy()
        wfw_all[counter]=1/wfw
        w_idf[counter]=idf['idf'].loc[idf['Target Word']==word].to_numpy()
        
        counter=counter +1
        #print(word, 1/wfw)
    #print(wfw_all)
    df_in['frequency_weight']=wfw_all
    df_in['idf']=w_idf
    return df_in

def compute_running_avg_entr(df_in, win_avg, args):
    #ma=df_in['Entropy'].rolling(running_win).mean().to_numpy()
    surp=np.empty(len(df_in))
    entrp_base=np.empty(len(df_in))
    surp[:]=np.nan
    entrp_base[:]=np.nan
    
    surp_std_all=np.empty(len(df_in))
    surp_entr_std_all=np.empty(len(df_in))
    surpZratio=np.empty(len(df_in))
    surpRunZ=np.empty(len(df_in))
    analyRunZ=np.empty(len(df_in))
    
    entrZratio=np.empty(len(df_in))
    entrRunZ=np.empty(len(df_in))
    
    
    surp_std_all[:]=np.nan
    surp_entr_std_all[:]=np.nan
    surpZratio[:]=np.nan
    surpRunZ[:]=np.nan
    entrRunZ[:]=np.nan
    entrZratio[:]=np.nan
    analyRunZ[:]=np.nan
    
    
    
    for index in range(win_avg,len(df_in)):
        if args.weight_by_freq=='Y':
            raw_weights=df_in[args.freq_type].iloc[index-win_avg:index+1]
            weights=raw_weights/np.sum(raw_weights)
            ma_surp_includecurrent_word=np.average(df_in['Surprise'].iloc[index-win_avg:index+1], weights=weights)
            std_surp_includecurrent_word=np.sqrt(np.cov(df_in['Surprise'].iloc[index-win_avg:index+1],  aweights=weights))
            
            surpRunZ[index]=(df_in['Surprise'].iloc[index]-ma_surp_includecurrent_word)/std_surp_includecurrent_word
            #print(surpRunZ[index])
            #print('Zweighting', weights)
        else:


            ma_surp=np.nanmean(df_in['Surprise'].iloc[index-win_avg:index])
            ma_entrp=np.nanmean(df_in['Entropy'].iloc[index-win_avg:index])

            std_surp=np.nanstd(df_in['Surprise'].iloc[index-win_avg:index])
            std_entr=np.nanstd(df_in['Entropy'].iloc[index-win_avg:index])

            surp[index]=df_in['Surprise'].iloc[index]/ma_surp
            entrp_base[index]=df_in['Surprise'].iloc[index]/ma_entrp

            surp_std_all[index]=df_in['Surprise'].iloc[index]/std_surp
            surp_entr_std_all[index]=df_in['Surprise'].iloc[index]/std_entr
            
            zz=(df_in['Surprise'].iloc[index]-ma_surp)/std_surp
            #print(zz)
            surpZratio[index]=(df_in['Surprise'].iloc[index]-ma_surp)/std_surp

            ma_surp_includecurrent_word=np.nanmean(df_in['Surprise'].iloc[index-win_avg:index+1])
            std_surp_includecurrent_word=np.nanstd(df_in['Surprise'].iloc[index-win_avg:index+1])

            ma_entrp_includecurrent_word=np.nanmean(df_in['Entropy'].iloc[index-win_avg:index+1])
            std_entrp_includecurrent_word=np.nanstd(df_in['Entropy'].iloc[index-win_avg:index+1])
            # for any precomputed analyses, not just surprirse or entropy.
            #Note the analyses has to be computed in create_analyses_df before you can use...
            # any analyses
            ma_analy=np.nanmean(df_in[args.analyses].iloc[index-win_avg:index+1])
            std_analy=np.nanstd(df_in[args.analyses].iloc[index-win_avg:index+1])
            
            surpRunZ[index]=(df_in['Surprise'].iloc[index]-ma_surp_includecurrent_word)/std_surp_includecurrent_word
            analyRunZ[index]=(df_in[args.analyses].iloc[index] - ma_analy)/std_analy
            entrZratio[index]=(df_in['Surprise'].iloc[index]-ma_entrp)/std_surp
            entrRunZ[index]=(df_in['Entropy'].iloc[index]-ma_entrp_includecurrent_word)/std_entrp_includecurrent_word
        
    df_in['Surprise-Ratio-Running_Average_{0}'.format(win_avg)]=surp
    df_in['Surprise-Running_Average_{0}_Entropy_base'.format(win_avg)]=entrp_base
    
    df_in['Surprise-Running_std_dev_surprise_{0}'.format(win_avg)]=surp_std_all
    df_in['Surprise-Running_std_dev_entropy_{0}'.format(win_avg)]=surp_entr_std_all
    
    df_in['Surprise-Ratio-Running_Z_{0}'.format(win_avg)]=surpZratio
    df_in['Entropy-Ratio-Running_Z_{0}'.format(win_avg)]=entrZratio
    df_in['SurpriseRunZ_{0}'.format(win_avg)]=surpRunZ
    df_in['EntropyRunZ_{0}'.format(win_avg)]=surpRunZ
    df_in[args.analyses+'RunZ_{0}'.format(win_avg)]=analyRunZ
    
    return df_in


def average_surp_reynolds(df_in, args):
    drift=args.drift/100 #0.05
    df_in['Average Surprise']=np.nan
    df_in['Average-{0}'.format(args.analyses)]=np.nan
    avg_surp=np.empty(len(df_in))
    avg_analyses=np.empty(len(df_in))
    analyses_col='{0}'.format(args.analyses)
    for i in range(len(df_in)):
        if i==0:
            avg_surp[i]=np.mean(df_in['Surprise']) #.iloc[i]
            avg_analyses[i]=np.nanmean(df_in[analyses_col].to_numpy()) #.iloc[i]
           
        else:
            avg_surp[i] =avg_surp[i-1]+ drift*(df_in['Surprise'].iloc[i] -avg_surp[i-1])
            avg_analyses[i] =avg_analyses[i-1]+ drift*(df_in[analyses_col].iloc[i] -avg_analyses[i-1])
    df_in['Average Surprise']=avg_surp
    df_in['Surprise_Ratio_Average_Surprise']=df_in['Surprise']/df_in['Average Surprise']
    # we want to divide by the average at time t-1. So shift the average values by 1 and 
    #reset the average for the fist position as the overall average. 
    #The first and second position will have the same average.
    avg_analyses=np.roll(avg_analyses,1)
    avg_analyses[0]=avg_analyses[1]
    df_in['Average-{0}-drift{1}'.format(args.analyses,args.drift)]=avg_analyses
    drift_col=args.analyses+'Ratio_Average_Reynolds_drift{0}'.format(args.drift) 
    df_in[drift_col]=df_in[args.analyses]/df_in['Average-{0}-drift{1}'.format(args.analyses,args.drift)]
    #plt.plot(range(len(df_in)), df_entr_sur['Surprise'])
    #plt.plot(avg_surp)
    return df_in



def compute_cross_entr(df_in):
    """Compute cross entropy with respect to the previous word
    Vocab length is for GPT2. Will need to change if other models are used."""
    prob=np.empty([len(df_in), 50257])
    cross_entr=np.zeros(len(df_in))
    kld=np.zeros(len(df_in))

    for i in tqdm(range(len(df_in))):
        z=df_in['logits'].iloc[i]

        x=np.array(ast.literal_eval(z))
        prob[i]=softmax(x)
        if i>0:
            cross_entr[i]=np.sum(-prob[i]*np.log2(prob[i-1]))
            kld[i]=entropy(prob[i],prob[i-1])
            
            #print(cross_entr[i])
    df_in['Cross_Entropy']=cross_entr
    df_in['KLDivergence']=kld
    
    return df_in

def compute_autocorr(data, pod_name):
    home_dir=folders['tiger']
    results_dir =os.path.join(home_dir,'results/')

    plt.figure(figsize=(5,5))
    plt.title('%s Autocorrelation' % pod_name)
    plot_acf(data, lags =600,alpha=0.05, use_vlines=False)
    plt.savefig(results_dir + '%s_button_autocorr.png' % pod_name, bbox_inches="tight")

def make_random(args,i,df_in):
    """ Generate random number signal for each story.
    The parametr i ensures a random seed for each story
    """
    analyses=args.analyses
    df_entr_sur=df_in
    if 'Simulated-Random' in analyses:
        prng=RandomState(50*i)
        sim_rand=prng.rand(len(df_entr_sur))
        df_entr_sur['Simulated-Random']=sim_rand

    #df=df_entr_sur[[analyses, 'onsets']]
    #df['onsets']=df_entr_sur['onsets']
    return df_entr_sur

def compute_embed_dist(df_embed_data):
    """Compute cosine distance with respect to the previous word
    """

    cos_dist=np.zeros(len(df_embed_data))
    for i in tqdm(range(len(df_embed_data))):
        if i>0:
            cos_dist[i]=cosine(df_embed_data.iloc[i,:],df_embed_data.iloc[i-1,:])
    return cos_dist
    
  