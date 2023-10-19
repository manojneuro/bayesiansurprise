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

import matplotlib.style as style 
sns.set(style = 'whitegrid', context='poster', rc={"lines.linewidth": 2.5})

import matplotlib
import matplotlib.patches as patches
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
from scipy.spatial.distance import cdist,mahalanobis
from scipy.stats import wasserstein_distance, pearsonr, entropy, ttest_ind, ks_2samp
from scipy import stats
from scipy.signal import correlate, correlation_lags

import pickle as pickle
from scipy.interpolate import UnivariateSpline
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.model_selection import TimeSeriesSplit

folders = { "desk": "/Users/manojkumar/Research/context-prediction/",
               "server": "/jukebox/norman/mkumar/context-prediction/",
               "tiger": "/scratch/gpfs/mk35/context-prediction/"}

elec_mni ={"661":"MPR_011918_coor_MNI_2018-03-02.txt",
            "662":"MPR_110113_coor_MNI_2018-03-02.txt",
            "717":"030519_coor_MNI_2019-03-11.txt",
            "723":"040519_coor_MNI_2019-04-08.txt",
            "741":"autoNY741_coor_MNI_2019-08-02.txt",
            "742":"autoNY742_coor_MNI_2019-08-07.txt",
            "743":"autoNY743_coor_MNI_2019-08-30.txt",
            "763":"autoNY763_coor_MNI_2020-02-21.txt"}
num_elec={"661": 104, 
          "662": 96,
          "717": 252,
          "723": 156,
          "741": 128,
          "742": 172,
          "743": 122,
          "763": 75,
          "798": 192}

bound_dict_gpt2= {"monkey":14, "pieman": 17, "tunnel":36}

brainiak_ver='brainiak_v11'
npad=200 # padd the surprise dataframe for smoothing, windowing operations.



def load_pickle(args):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(args, 'rb') as fh:
        datum = pickle.load(fh)
    return datum

def z_score(neural_signal):
    scaler = preprocessing.StandardScaler().fit(neural_signal.reshape(-1, 1))
    neural_signal_z=scaler.transform(neural_signal.reshape(-1, 1))
    neural_signal_z=np.ndarray.flatten(neural_signal_z)
    return neural_signal_z

def run_pca(data, n_components=50):

    pca = PCA(n_components, svd_solver='auto')

    df_emb = data

    pca_output = pca.fit_transform(df_emb)
    df_emb = pca_output

    return df_emb

def read_embedding(pod_name, model, get_PCA='Y'):
    """Reads embedding files, filters out rows that do not have embeddings,
    gets the PCA version if flag is true.
    """
    # get file name
    if model=='GPT2':
        model_file='gpt2-xl-c_1024'
    home_dir = folders['tiger'] # cluster name is the input

    if pod_name=='monkey':
        if get_PCA=='Y':
            embed_file=home_dir +'/code/podcast-extract-embeddings/results/gpt2-xl-c_1024/gpt2-xl-c_1024-layer_0_pca50d.csv'
        else:
            embed_file=home_dir +'/code/podcast-extract-embeddings/results/gpt2-xl-c_1024/gpt2-xl-c_1024-layer_0.csv'
        datum_file=home_dir +'/code/podcast-extract-embeddings/results/gpt2-xl-c_1024/datum.csv'
    else:
        if get_PCA=='Y':
            embed_file=home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) \
            +'%s%s-layer_0_pca50d.csv' % (pod_name, model_file)
        else:
            embed_file=home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) \
            +'%s%s-layer_0.csv' % (pod_name, model_file)
        datum_file=home_dir +'/code/podcast-extract-embeddings/results/%s/' % (pod_name) + 'datum.csv'
    
    df_embed=pd.read_csv(embed_file,  sep=',', header=None)
    
    #col_names =['token','onset','offset','prob','speaker','cloze','predicted_word']
    df_datum=pd.read_csv(datum_file,  sep=',', header=(0), usecols=range(4))
    if get_PCA=='Y':
        df_embed_data=df_embed.iloc[:,-50:]
    else:  
        if pod_name=='monkey':
            df_embed_data=df_embed[df_embed.iloc[:,1].notnull()].iloc[:,5:-1]
        else:
            df_embed_data=df_embed[df_embed.iloc[:,1].notnull()].iloc[:,3:-1]
    df_filtered=df_datum
    return df_embed_data, df_filtered
    
def compute_density(time_indx_bp, x_grid):
    #compute denisty on defined grid
    kde = FFTKDE(bw='ISJ', kernel='gaussian')
    y_kde = kde.fit(time_indx_bp).evaluate(x_grid)
    return y_kde

def compute_corr_part_random(part, Nperm=100):
    # Split the part into two halves
    m,n =part.shape
    num_sub=n
    random.seed(42)
    corr=[]
    for i in range(Nperm):
        items = list(range(num_sub))

    # Shuffle the list randomly
        random.shuffle(items)
        array1 = items[:num_sub//2]
        array2 = items[num_sub//2:]
 
        half1 = part[:,array1]
        half2 = part[:,array2]
        time_split1 = half1.nonzero()[0]
        time_split2 = half2.nonzero()[0]

        kde = FFTKDE(bw='ISJ', kernel='gaussian')




    # Compute the kernel density estimate for each half
        _,y_kde1 = kde.fit(time_split1/1000).evaluate()
        _, y_kde2 = kde.fit(time_split2/1000).evaluate()
        #plt.plot(y_kde1)
        #plt.plot(y_kde2)
        #print(np.corrcoef(y_kde1, y_kde2)[1][0])
    # Compute the correlation between the two halves using Pearson's r
        corr.append(np.corrcoef(y_kde1, y_kde2)[1][0])

    return corr
    
    #Use only the latter half for piemandouble
    if pod_name=='piemandouble':
        df_embed_data=df_embed_data[899:]
        df_filtered=df_filtered[899:]
    
    return df_embed_data, df_filtered



    match_ptile_score = np.array(match_percentile)
    xlab=np.arange(1,len(match_ptile_score))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(xlab,match_ptile_score[:,0],'kx-')

    plt.ylabel('Match Score')
    plt.xlabel('Time window %s boundary' % match, labelpad=10)
    plt.ylim([0, 1])
    plt.title('%s %s Event Boundary Matches Human Event Boundary' % (pod_name.title(),model), y=1.1)
    plt.xticks(xlab, rotation =90)
    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(xlab,match_ptile_score[:,1])
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('Percentile Score')
    plt.ylim([0, 100])
    plt.xticks(xlab, rotation =90)

def interp(x, value_in):
    spl_interp =  UnivariateSpline(x, value_in)
    sample_points=np.linspace(0,1, 50)
    value_interp=spl_interp(sample_points)
    return sample_points, value_interp

 #Compute autocorrelations
def compute_autocorr(data1):
 
    plot_acf(data1, lags=len(data1)/2 -1)
    
def compute_cross_corr(pod_name, data1, data2, smooth):
    home_dir = folders['tiger'] # cluster name is the input
    results_dir =os.path.join(home_dir,'results', brainiak_ver,pod_name)
    cross_corr=correlate(data1,data2)
    lags = correlation_lags(len(data1), len(data2))
    cross_corr /= np.max(cross_corr)
    plt.figure(figsize=(5,5))
    ax=sns.lineplot(x=lags, y=cross_corr)
    ax.set_xlim([-500,500])
    
    plt.savefig(results_dir + '/%s_surprise_entropy_cross_corr_sm%i.png' % (pod_name,smooth), bbox_inches="tight")
    return cross_corr


    home_dir = folders['tiger'] # cluster name is the input
    results_dir =os.path.join(home_dir,'results', brainiak_ver,pod_name)
    tssplit = TimeSeriesSplit(n_splits=5)
    
    # get the word index before the button press:
    indx_word=np.empty(len(human_bounds))
    for j in range(len(human_bounds)):
        indx_word[j]=np.argmax(df['onsets']> human_bounds[j]) -1
        
    for metric in ['Surprise', 'Entropy']:
        f, axes = plt.subplots(5,1, figsize=(30, 10))
        plt.subplots_adjust(top = 1.5, bottom=0.01, wspace=0.1)
        title_text = '%s %s: %s Over Time: Zoomed In' % (pod_name.title(), model,metric)
        plt.suptitle(title_text, y=1.6, fontsize=20)
        input=df[metric]
        ymax=np.max(input) +2
        i=0
        for split1, split2 in tssplit.split(input):
            #print(split1, split2)
            signal1=input[split1]
            signal2=input[split2]
            if i==0:
                sns.lineplot(data=signal1, ax=axes[i])
                boundary_markers= indx_word[np.where(np.logical_and(indx_word>=np.min(split1), indx_word<=np.max(split1)))]
                axes[i].set_ylim(0,ymax)
                for xc in boundary_markers:
                    axes[i].axvline(x=xc, color='orange')
            else:
                sns.lineplot(data=signal2, ax=axes[i])
                boundary_markers= indx_word[np.where(np.logical_and(indx_word>=np.min(split2), indx_word<=np.max(split2)))]
                axes[i].set_ylim(0,ymax)
                for xc in boundary_markers:
                    axes[i].axvline(x=xc, color='orange')
            i=i+1
        plt.savefig(results_dir + '%s_%s_%s_zoomed_in_sm0.png' % (pod_name,model, metric),  bbox_inches="tight")
        
def pad_df(df, analyses):
    df_entr_sur_padd=[]
    npad=200
    surprise_pad_start = df['Surprise'].iloc[0]*np.ones(npad)
    entropy_pad_start = df['Entropy'].iloc[0]*np.ones(npad)
    analyses_pad_start=df[analyses].iloc[0]*np.ones(npad)
    #print(surprise_pad_start, entropy_pad_start)
    onset_pad_start = np.zeros(npad)
    df_pad_start=pd.DataFrame(surprise_pad_start)
    df_pad_start.columns=['Surprise']
    df_pad_start['Entropy']=entropy_pad_start
    df_pad_start['onsets']=onset_pad_start
    df_pad_start[analyses]=analyses_pad_start

    surprise_pad_end = df['Surprise'].iloc[-1]*np.ones(npad)
    entropy_pad_end = df['Entropy'].iloc[-1]*np.ones(npad)
    df_pad_end=pd.DataFrame(surprise_pad_end)
    df_pad_end.columns=['Surprise']
    df_pad_end['Entropy']=entropy_pad_end
    onset_pad_end = df['onsets'].iloc[-1]*2*np.ones(npad) # dumy pad value
    df_pad_end['onsets']=onset_pad_end
    merge_frames=[df_pad_start, df, df_pad_end ]
    df_entr_sur_padd=pd.concat(merge_frames)
    return df_entr_sur_padd