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
from numpy.random import RandomState

import imp
import time
from timeit import default_timer as timer
import matplotlib.patches as patches
import pickle as pickle
import os
import argparse

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('autosave', '5')
sns.set(style = 'whitegrid', context='poster', rc={"lines.linewidth": 2.5})
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import matplotlib.patches as patches
#mpl.use('Agg')

from sklearn.linear_model import  Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
#%matplotlib inline 
#%autosave 5
from context_helper import folders
from surprise_helper import  compute_autocorr

from wordfreq import zipf_frequency
#%matplotlib inline 
#%autosave 5

model='GPT2'

home_dir = folders['tiger']
save_plot='Y'
results_dir= home_dir + 'results/revision1/'
log_dir = home_dir + 'logs/'

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--analyses', type=str)
parser.add_argument('--nfeat', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('--stat', type=str)
parser.add_argument('--remove-func-words', type=str)
args= parser.parse_args()

def get_analyses_data(pod_name,args):

    df_all=pd.DataFrame()
    home_dir = folders['tiger']
    analyses=args.analyses
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

    #print(df_entr_sur.shape,df_entr_sur.columns)
    if pod_name=='monkey':
        onset=datum_data['onset']/512
    else:
        onset=datum_data['onset']
    df_entr_sur['onsets']=onset

    df_entr_sur.drop(columns=['Unnamed: 0'], inplace=True)
    df=df_entr_sur[[analyses, 'onsets']]
    #df['onsets']=df_entr_sur['onsets']


    df_all= pd.concat([df_all, df])
  
    return df_all

def load_analyses_data(pod_name,args,i):
    # New function to load precomputed data
    df_all=pd.DataFrame()
    home_dir = folders['tiger']
    analyses=args.analyses
    print(i, pod_name)
    #counter=0

    datum_file= home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) + 'datum.csv'
    
    surprise_file= home_dir +'/outputs/'  + '%s_processed_df.csv' % pod_name
    datum_data=pd.read_csv(datum_file,  sep=',', header=(0), usecols=range(4))
    if args.remove_func_words =='Y':
        surprise_file= home_dir +'/outputs/'  + '%s_processed_df_non-func.csv' % pod_name

    df_entr_sur=pd.read_csv(surprise_file,  sep=',', header=(0))
    if args.remove_func_words =='Y':
        df_entr_sur.rename(columns={'Unnamed: 0':'Original_Index'}, inplace=True)
        datum_data=datum_data.loc[df_entr_sur['Original_Index']]


    #if pod_name=='tunnel':
        #df_filtered=df_entr_sur.loc[125:]
        #df_entr_sur=df_filtered
    #print(df_entr_sur.shape,df_entr_sur.columns)
    if pod_name=='monkey':
        onset=datum_data['onset']/512
    else:
        onset=datum_data['onset']
    
    if args.remove_func_words =='N':
        # not sure if this will impact anything, but doing
        # this only for the full word case as we have renamed the columns to 
        #Original_index.
        df_entr_sur['onsets']=onset
        df_entr_sur.drop(columns=['Unnamed: 0'], inplace=True)
    else:
        df_entr_sur['onsets']=onset.loc[df_entr_sur['Original_Index']].to_numpy()
    # Simulated Random is precomputed in make_random
   

    if 'Simulated-Positive' in analyses:
        # use the button press values directly as the simulation signal
        df_button= get_button_press_proportion(pod_name)
        word_onset=get_onset_time(df_entr_sur)
        button_extract=df_button[word_onset,1]
        button_extract=button_extract/max(button_extract)
        if 'Simulated-Positive+Noise' in analyses:
            prng=RandomState(50*i)
            sim_rand=prng.rand(len(df_entr_sur))
            noise_pct=int(analyses[-2:])/100
            button_extract=(1-noise_pct)*sim_rand + noise_pct*button_extract
        df_entr_sur[analyses]=button_extract
    df=df_entr_sur[[analyses, 'onsets']]
    #df['onsets']=df_entr_sur['onsets']
  
    return df

def get_onset_time(df):
    if pod_name=='monkey':
        onset_time=df['onsets'].to_numpy()*1000
    if pod_name=='tunnel':
        onset_time=np.round(df['onsets'].to_numpy()*10)
    if pod_name=='pieman':
        onset_time=np.round(df['onsets'].to_numpy()*1000)
    #story_id=df['story_id'].to_numpy().astype(int)
    onset_time=onset_time.astype(int)
    return onset_time

def get_button_press_proportion(pod_name):

    if pod_name=='monkey':
        button_file=home_dir + 'outputs/mturk/%s_button_gaussian.csv' % pod_name
        df_button=pd.read_csv(button_file)
    else:
        button_file=home_dir + 'outputs/%s_button_gaussian.csv' % pod_name
    df_button=pd.read_csv(button_file)
    button=df_button.to_numpy()
    return button

def get_button_press_kde(pod_name):

    button_kde_file=home_dir + 'outputs/%s_button_density.csv' % pod_name
    df_button_density=pd.read_csv(button_kde_file)
    button=df_button_density.to_numpy()

    return button

def build_signal_mat(signal,nfeat):
    feat_win=nfeat
    mat_indx=np.zeros([len(signal),feat_win]).astype(int)
    sig_mat=np.empty([len(signal),feat_win])
    for i in range(feat_win,len(signal)):
        mat_indx[i]=np.linspace(i-feat_win,i-1,feat_win)
        sig_mat[i]=np.squeeze(signal[mat_indx[i,:]])
    return sig_mat

def z_score(signal):
    scaler=StandardScaler()
    signal=signal.reshape(-1,1)
    scaler.fit(signal)
    signal_z=scaler.transform(signal)
    return signal_z

def permute_signal(sig_mat,p):

    sig_perm = np.roll(sig_mat,p)

    return sig_perm


def compute_grid_search_regression(sig_mat_all,story_id_all, button_all, args):
    X, y=sig_mat_all, button_all
    ps = PredefinedSplit(story_id_all)
    corr_score=[]
    betas=[]
    #p_space1 = [10000, 20000, 30000, 40000]#np.logspace(0.01, 20, num=10, endpoint=True)
    #p_space2 = np.logspace(0.00001, 4, num=20, endpoint=True)
    p_space3 = np.logspace(-16, -5, num=10, endpoint=True)
    


    if args.model=='Lasso':
        parameters=[ {
                'clf': [Lasso()],
                'alpha': p_space3,
                'max_iter':[5000],
                'tol':[1e-2]
            }]

    result=[]
    for params in parameters:

        #classifier
        clf = params['clf'][0]
            #getting arguments by
            #popping out classifier
        params.pop('clf')
       
        for train_index, test_index in ps.split():
                # split the data 
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            story_train, story_test=story_id_all[train_index],story_id_all[test_index]
            ps_train=PredefinedSplit(story_train)
            # fit the model on the training set 
            
            search = GridSearchCV(clf,param_grid=params, n_jobs=2, cv=ps_train)

            search.fit(X_train, y_train)
                
            # calculate the accuracy for the hold out run
            best_model=search.best_estimator_
            print(best_model)
            betas.append(best_model.coef_)
            ypred= best_model.predict(X_test)
            if args.stat=='Pearson':
                corr_score.append(np.corrcoef(y_test,ypred)[0,1])

            result.append\
            (   {
            'grid': search,
            'classifier': search.best_estimator_,
            'best params': search.best_params_,
            'cv': search.cv,
            'corr':corr_score
            })

    return corr_score, betas

stories=['monkey','pieman', 'tunnel']
sig_mat_all=[]
button_all=[]
story_id_all=[]
nfeat=args.nfeat
analyses_str=args.analyses
do_zscore=False #True only for non-transient
if do_zscore:
    suffix='yesZscore'
else:
    suffix='noZscore'
if args.remove_func_words=='Y':
    suffix =suffix + '_non-func'
for i,pod_name in enumerate(stories):

    df=load_analyses_data(pod_name,args,i)
    df['story_id']=i
    #Build signal matrix
    signal=df[args.analyses].loc[df['story_id']==i].to_numpy()
    # remove any leading Nans for anlyses with running Z
    indx=np.isfinite(signal)
    num_nan= len(indx[~indx])

    signal=signal[indx]
    if do_zscore:
        signal_z=z_score(signal)
        if pod_name=='monkey':
            signal_monkey = signal_z
        if pod_name=='tunnel':
            signal_tunnel=signal_z
        if pod_name=='pieman':    
            signal_pieman = signal_z        
        
    
        sig_mat=build_signal_mat(signal_z,nfeat)
    else:
        if pod_name=='monkey':
            signal_monkey = signal
        if pod_name=='tunnel':
            signal_tunnel=signal
        if pod_name=='pieman':    
            signal_pieman = signal  
        sig_mat=build_signal_mat(signal,nfeat)    
    
    sig_mat=sig_mat[nfeat:,:]

    
    #get button press proportions
    df_button= get_button_press_kde(pod_name)
    word_onset=get_onset_time(df)[num_nan+nfeat:]
    button_extract=df_button[word_onset,1]
    #compute_autocorr(button_extract, pod_name)
    button_extract=button_extract/max(button_extract)
    #plt.plot(button_extract)
    if i==0:
        sig_mat_all=sig_mat
    else:
        sig_mat_all=np.vstack([sig_mat_all,sig_mat])
    button_all.append(button_extract)
    story_id_all.append(df['story_id'].loc[num_nan+nfeat:].to_list())
    
button_all=np.concatenate(button_all)
#compute_autocorr(button_all, 'All_3_Stories_Button_Concatenated')
story_id_all=np.concatenate(story_id_all)

corr_score, betas =compute_grid_search_regression(sig_mat_all,story_id_all, button_all, args)
corr_perm_all=[]
betas_perm_all=[]
nPerm=4000
lin_s = list(map(int, np.linspace(-nPerm, nPerm, 2*nPerm+1)))
lin_s.remove(0)
for perm in tqdm(lin_s):
    perm_sig_all=[]
    perm_sig_monkey=np.roll(signal_monkey,perm)
    perm_sig_monkey=build_signal_mat(perm_sig_monkey,nfeat)
    perm_sig_monkey=perm_sig_monkey[nfeat:,:]
    print(perm_sig_monkey.shape)
    perm_sig_pieman=np.roll(signal_pieman,perm)
    perm_sig_pieman=build_signal_mat(perm_sig_pieman,nfeat)
    perm_sig_pieman=perm_sig_pieman[nfeat:,:]

    perm_sig_tunnel=np.roll(signal_tunnel,perm)
    perm_sig_tunnel=build_signal_mat(perm_sig_tunnel,nfeat)
    perm_sig_tunnel=perm_sig_tunnel[nfeat:,:]    
    perm_sig_all=np.vstack([perm_sig_monkey,perm_sig_pieman])
    perm_sig_all=np.vstack([perm_sig_all,perm_sig_tunnel])
    print(perm,perm_sig_all.shape)
    corr_perm, betas_perm=compute_grid_search_regression(perm_sig_all,story_id_all, button_all, args)
    corr_perm_all.append(corr_perm)
    betas_perm_all.append(betas_perm)

def plot_correlations(corr_perm_all,corr_score,analyses_str, nfeat, args, suffix):
    f,ax=plt.subplots(1,1, figsize=(5,5))
    sns.violinplot(data=np.mean(corr_perm_all,axis=1), color='skyblue')
    plt.title('%s: Actual Correlation vs. Null, NFeature =%i' %(analyses_str, nfeat), fontsize=12)
    plt.scatter(0, np.mean(corr_score), marker='o', color='r', s=100)
    plt.savefig(results_dir + 'model_button_regression_%s_%i_%s_%s_%s.png' % (analyses_str, nfeat, args.model, args.stat, suffix) ,  bbox_inches="tight")

out_betas=results_dir + 'perm_regression_outputs_%s_%i_%s_%s' % (analyses_str, nfeat, args.model, suffix) 
np.savez(out_betas,betas_perm_all=betas_perm_all, betas=betas, corr_score=corr_score,corr_perm_all=corr_perm_all)
plot_correlations(corr_perm_all,corr_score,analyses_str, nfeat, args, suffix)
pval=stats.percentileofscore(np.mean(corr_perm_all,axis=1),np.mean(corr_score))

if do_zscore:
    log_file = results_dir + '%s_correlation_pval_nfeature_%s_%i_%s_%s.txt' % (analyses_str, suffix,nfeat, args.model, args.stat)
else:
    log_file = results_dir + '%s_correlation_pval_nfeature_%s_%i_%s_%s.txt' % (analyses_str, suffix, nfeat, args.model, args.stat)

f = open(log_file, 'w')
f.write('Analyses = %s, Nfeatures= %i,  %s Mean Corr = %.2f, p-val= %2.1f' % (analyses_str, nfeat,args.stat,np.mean(corr_score),pval))
print(pval)
f.close


def plot_betas(betas,betas_perm_all,analyses_str, nfeat, args, suffix):
    f,ax = plt.subplots(1,1, figsize=(10,5))
    plt.title('Betas: %s, NFeature =%i' %(analyses_str, nfeat), fontsize=12)
    bperm=np.array(betas_perm_all)
    bperm_mean=np.mean(bperm, axis=1)
    sns.violinplot(data=bperm_mean,color='skyblue', inner=None,  alpha=0.5,ax=ax)
    betas_mean=np.mean(betas,axis=0)
    sns.lineplot(data=betas_mean, color='yellow',ax = ax)
    plt.savefig(results_dir +'%s_betas_%i_%s_%s' % (analyses_str, nfeat, args.model, suffix), bbox_inches='tight')

plot_betas(betas,betas_perm_all,analyses_str, nfeat, args, suffix)

def plot_prem_corr(corr_score,corr_perm_all,analyses_str, nfeat, suffix):
   mean_perm=np.mean(corr_perm_all,axis=1)
   all_corr=np.insert(mean_perm, mean_perm.shape[0]//2,np.mean(corr_score)) 
   f,ax=plt.subplots(1,1, figsize=(10,5))
   plt.title('Correlation Across Circular Shift: %s, NFeature =%i' %(analyses_str, nfeat), fontsize=12)
   sns.lineplot(data=all_corr, ax=ax)
   #xlab = [i for i in range(-mean_perm.shape[0], mean_perm.shape[0]+1, 100)]
   #ticknum=2*(mean_perm.shape[0]) + 2
   #xtick_loc=np.arange(0,ticknum, 100)
   #ax.set_xticks(xtick_loc)
   #ax.set_xticklabels(xlab, rotation=90)
   ax.set_xlabel('Circular Shift Position')
   ax.set_ylabel('Correlation')
   plt.savefig(results_dir + '%s_corr_perm_%i_%s_%s_%s.png' % (analyses_str, nfeat, args.model, args.stat, suffix), bbox_inches='tight')

plot_prem_corr(corr_score,corr_perm_all,analyses_str, nfeat, suffix)
print('Processing Completed')
