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
from sklearn.metrics import mean_squared_error
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

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
#%matplotlib inline 
#%autosave 5


from context_helper import folders 
#%matplotlib inline 
#%autosave 5


from surprise_helper import  compute_autocorr
model='GPT2'

home_dir = folders['tiger']
save_plot='Y'
results_dir= home_dir + 'results/revision1/'
log_dir = home_dir + 'logs/'

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--analyses', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--stat', type=str)


args= parser.parse_args()
compute_aic_per_story= True

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
    df_entr_sur=pd.read_csv(surprise_file,  sep=',', header=(0))
    #if pod_name=='tunnel':
        #df_filtered=df_entr_sur.loc[125:]
        #df_entr_sur=df_filtered
    #print(df_entr_sur.shape,df_entr_sur.columns)
    if pod_name=='monkey':
        onset=datum_data['onset']/512
    else:
        onset=datum_data['onset']
    df_entr_sur['onsets']=onset

    df_entr_sur.drop(columns=['Unnamed: 0'], inplace=True)
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


def compute_grid_search_regression_aic(sig_mat_all,story_id_all, button_all, args):
    X, y=sig_mat_all, button_all
    ps = PredefinedSplit(story_id_all)
    corr_score=[]
    betas=[]
    mse_all=[]
    aic_all=[]
 
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
           
            #Compute AIC  
            mse=mean_squared_error(y_test,ypred)
            num_samples= len(button_all)
            aic = num_samples *np.log(mse) + 2*(nfeat + 1)
            

            mse_all.append(mse)
            aic_all.append(aic)
    

            result.append\
            (   {
            'grid': search,
            'classifier': search.best_estimator_,
            'best params': search.best_params_,
            'cv': search.cv,
            'corr':corr_score,
            'aic': aic_all,
            'mse': mse_all
            })

    return corr_score, result


df_results = pd.DataFrame(index = range(5))

df_results['MSE']=np.nan
df_results['Mean-MSE']=np.nan
df_results['Correlation']=np.nan

df_results['AIC-Value']=np.nan
df_results['Mean-AIC-Value']=np.nan
df_results['N_Features']=np.nan
df_results['MSE']=df_results['MSE'].astype(object)
df_results['AIC-Value']=df_results['AIC-Value'].astype(object)
counter=0

for nfeat in [20, 50, 100, 150, 200]:


    stories=['monkey','pieman', 'tunnel']
    sig_mat_all=[]
    button_all=[]
    story_id_all=[]
    analyses_str=args.analyses
    do_zscore=False # only for non-transient analysis
    if do_zscore:
        suffix='yesZscore'
    else:
        suffix='noZscore'
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
        #df_button= get_button_press_proportion(pod_name)
        # use the button press density computed via KDE 
        df_button=get_button_press_kde(pod_name)
        word_onset=get_onset_time(df)[num_nan+nfeat:]
        button_extract=df_button[word_onset,1]
        compute_autocorr(button_extract, pod_name)
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

    # Compute AIC
    if compute_aic_per_story is False:
        suffix='full_ic'
        classify = Lasso(alpha=1e-05,tol=1e-2,max_iter=5000)
        classify.fit(sig_mat_all, button_all)
        y_predict=classify.predict(sig_mat_all)
        mse=mean_squared_error(button_all,y_predict)
        num_samples= len(button_all)
        df_results['MSE'].loc[counter]=mse
        df_results['N_Features'].loc[counter]=int(nfeat)
        df_results['AIC-Value'].loc[counter] = num_samples *np.log(mse) + 2*(nfeat + 1)
        df_results['Correlation']=np.corrcoef(y_predict, button_all)[0,1]
       
    if compute_aic_per_story:
        suffix= suffix + 'mean_ic'
        corr_score, aic_results =compute_grid_search_regression_aic(sig_mat_all,story_id_all, button_all, args)
        df_results['MSE'].loc[counter]=[aic_results[0]['mse']]
        df_results['Mean-MSE'].loc[counter]=np.mean(aic_results[0]['mse'])

        df_results['N_Features'].loc[counter]=int(nfeat)
        df_results['AIC-Value'].loc[counter] = [aic_results[0]['aic']]
        df_results['Mean-AIC-Value'].loc[counter] = np.mean(aic_results[0]['aic'])
        df_results['Correlation'].loc[counter]=np.mean(corr_score)
        

    counter=counter+1

col_order=['N_Features','Mean-AIC-Value','Correlation','Mean-MSE']
fname = results_dir + '%s_model_aic_%s.tex' % (args.analyses, suffix)
fname_csv = results_dir + '%s_model_aic_%s.csv' % (args.analyses, suffix)
df_results[col_order].to_latex(fname,index=False)
df_results.to_csv(fname_csv)
