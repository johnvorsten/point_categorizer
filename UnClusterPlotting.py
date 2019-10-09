# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:44:19 2019

@author: z003vrzk
"""

from JVWork_UnsupervisedCluster import JVClusterTools
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from gap_statistic import OptimalK
from JVWork_WholeDBPipeline import JVDBPipe
from sklearn.manifold import MDS
import matplotlib.pyplot as plt






#%%

myDBPipe = JVDBPipe()
myClustering = JVClusterTools()

_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
_master_pts_db_clean = r"D:\Z - Saved SQL Databases\master_pts_db_clean.csv"
_ptname_hot_path = r"D:\Z - Saved SQL Databases\master_pts_db_name_onehot.csv"
_sys_hot_path = r"D:\Z - Saved SQL Databases\master_pts_db_system_onehot.csv"
_desc_hot_path = r"D:\Z - Saved SQL Databases\master_pts_db_description_onehot.csv"

my_iter = myClustering.read_database_set(_master_pts_db)
ind, df = next(my_iter)

df_clean = myDBPipe.cleaning_pipeline(df, remove_dupe=False, 
                                      replace_numbers=False, remove_virtual=True)

df_text = myDBPipe.text_pipeline(df_clean, vocab_size='all')

_vocabulary = df_text.columns.tolist()

X = df_text.values

#Visualize some data
_point_names = myClustering.get_word_name(X, _vocabulary)
print(df_clean.loc[0, 'DBPath'])
for _i in range(0, min(5, len(_point_names))):
    print(_point_names[_i])

#%% Correct k
#Test it out : should be
_correct_k_dict_path = r"D:\Z - Saved SQL Databases\correct_k_dict.csv"
_correct_k_dict = pd.read_csv(_correct_k_dict_path, index_col=0, header='infer')

#%% Scaling and predicting number of classes per dataset
"""Find optimal K"""

mds = MDS(n_components = 2)
X_reduced_mds = mds.fit_transform(X)
_max_clusters = int(np.ceil(X.shape[0]/2))

optimalkMDS = OptimalK(parallel_backend='multiprocessing')
num_k_gap1_MDS = optimalkMDS(X_reduced_mds, cluster_array=np.arange(1,_max_clusters,1))
gapdf1_MDS = optimalkMDS.gap_df
optimalkX = OptimalK(parallel_backend='multiprocessing')
num_k_gap1_X = optimalkX(X.astype(np.float32), cluster_array=np.arange(1,_max_clusters,1))
gapdf1_X = optimalkX.gap_df

num_k_gap2_X, gapdf2_X = myClustering.optimalK2(X, nrefs=5, maxClusters=_max_clusters)
num_k_gap2_MDS, gapdf2_MDS = myClustering.optimalK2(X_reduced_mds, nrefs=5, maxClusters=_max_clusters)

try:
    _col = np.where(df_clean.columns == 'DBPath')[0][0]
    _row = np.where(_correct_k_dict['DBPath'] == df_clean.iloc[0, _col])[0][0]
    _correct_k = _correct_k_dict.iloc[_row, 1]
except:
    _excel_path = ".\\user_input.csv"
    df_clean.sort_values(by=['NAME']).to_csv(_excel_path)
    os.startfile(_excel_path)
    
    
    num_k_user = int(input('User Defined Number of Clusters : '))
    new_pts = pd.DataFrame({'DBPath':[df_clean.iloc[0, _col]],
                            'Correct Number Clusters':[num_k_user]})
    
    with open(_correct_k_dict_path, 'a') as f:
        new_pts.to_csv(f, header=False)
        
    _correct_k_dict = _correct_k_dict.append(new_pts)
    _correct_k = _correct_k_dict[df_clean.iloc[0, _col]]



#%%
#Plotting
def plt_MDS(x, y, classes, artist):
    """parameters
    -------
    x : array (1D) of values
    y : array (1D) of values
    classes : array of classes for each (x,y)"""
#    plt.figure(fig_)

    uniques = list(set(classes))
    colors = [np.array(plt.cm.viridis(i/float(len(uniques)))).reshape(1,-1) for i in range(len(uniques)+1)]
    for idx, uniq in enumerate(uniques):
        xi = [x[j] for j in range(len(x)) if classes[j] == uniq]
        yi = [y[j] for j in range(len(x)) if classes[j] == uniq]
        artist.scatter(xi, yi, c=colors[idx], label=str(uniq))
        
    artist.set_title('MDS Reduction')
    artist.legend()
    artist.set_xlabel('$z_1$')
    artist.set_ylabel('$z_2$')
    artist.grid(True)

mds_fig = plt.figure(1)
ax = mds_fig.subplots(1,1)
plt_MDS(X_reduced_mds[:,0], X_reduced_mds[:,1], np.zeros(X_reduced_mds.shape[0]), ax)

#Gap
def plt_gap(k_vals, gap_vals, optimal_k, artist, label='Gap1', correct_k=None):

    artist.plot(k_vals, gap_vals, linewidth=2, label=label)
    artist.scatter(optimal_k, gap_vals[optimal_k - 1], s=200, c='r')
    if correct_k:
        artist.axvline(x=correct_k, ymin=0.05, ymax=0.95, c='g', label='Correct k', linestyle='--')
    
    artist.grid(True)
    artist.set_xlabel('Cluster Count')
    artist.set_ylabel('Gap Values (mean(log(refDisps)) - log(origDisp))')
    artist.set_title('Gap Value v. Cluster Count')
    artist.legend()

gap_fig = plt.figure(2)
ax = gap_fig.subplots(1,1)
plt_gap(gapdf1_MDS['n_clusters'], gapdf1_MDS['gap_value'], num_k_gap1_MDS, ax, label='Gap1_MDS', correct_k=_correct_k)
plt_gap(gapdf1_X['n_clusters'], gapdf1_X['gap_value'], num_k_gap1_X, ax, label='Gap1_X')
plt_gap(gapdf2_MDS['n_clusters'], gapdf2_MDS['gap_value'], num_k_gap2_MDS, ax, label='Gap2_MDS')
plt_gap(gapdf2_X['n_clusters'], gapdf2_X['gap_value'], num_k_gap2_X, ax, label='Gap2_X')

gap_fig = plt.figure(3)
ax = gap_fig.subplots(1,1)
plt_gap(gapdf1_MDS['n_clusters'], gapdf1_MDS['gap*'], num_k_gap1_MDS, ax, label='Gap*1_MDS', correct_k=_correct_k)
plt_gap(gapdf1_X['n_clusters'], gapdf1_X['gap*'], num_k_gap1_X, ax, label='Gap*1_X')
plt_gap(gapdf2_MDS['n_clusters'], gapdf2_MDS['gap*'], num_k_gap2_MDS, ax, label='Gap*2_MDS')
plt_gap(gapdf2_X['n_clusters'], gapdf2_X['gap*'], num_k_gap2_X, ax, label='Gap*2_X')

#Dispersion
def plt_dispersion(k_vals, dispersion, label, artist):
    a = artist.plot(k_vals, dispersion, linewidth=1, label=label)
    artist.grid(True)
    artist.set_xlabel('cluster (k)')
    artist.set_ylabel('Dispersion metric')
    artist.set_title('Dispersion v. cluster count')
#    artist.axvline(x=correct_k, ymin=0.05, ymax=0.95, c='g', linestyle='--')
    artist.legend()
    return a

disp_fig = plt.figure(4)
ax = disp_fig.subplots(2,1, sharex=True)

plt_dispersion(gapdf2_MDS['n_clusters'], gapdf2_MDS['ref_dispersion'], 'Reference MDS', ax[0])
plt_dispersion(gapdf2_MDS['n_clusters'], gapdf2_MDS['obs_dispersion'], 'Observed MDS', ax[0])
plt_dispersion(gapdf2_X['n_clusters'], gapdf2_X['ref_dispersion'], 'Reference X', ax[1])
plt_dispersion(gapdf2_X['n_clusters'], gapdf2_X['obs_dispersion'], 'Observed X', ax[1])


#Find derivatives
_gradient_mds_orig = np.gradient(gapdf2_MDS['obs_dispersion'])
_gradient_mds_ref = np.gradient(gapdf2_MDS['ref_dispersion'])
_gradient_x_orig = np.gradient(gapdf2_X['obs_dispersion'])
_gradient_x_ref = np.gradient(gapdf2_X['ref_dispersion'])

disp_fig2 = plt.figure(5)
ax = disp_fig2.subplots(2,2, sharex=True)
plt_dispersion(gapdf2_MDS['n_clusters'], gapdf2_MDS['ref_dispersion'], 'Reference MDS', ax[0,0])
plt_dispersion(gapdf2_MDS['n_clusters'], gapdf2_MDS['obs_dispersion'], 'Observed MDS', ax[0,0])
plt_dispersion(gapdf2_X['n_clusters'], gapdf2_X['ref_dispersion'], 'Reference X', ax[0,1])
plt_dispersion(gapdf2_X['n_clusters'], gapdf2_X['obs_dispersion'], 'Observed X', ax[0,1])
plt_dispersion(gapdf2_MDS['n_clusters'], _gradient_mds_ref, 'Reference MDS gradient', ax[1,0])
plt_dispersion(gapdf2_MDS['n_clusters'], _gradient_mds_orig, 'Observed MDS gradient', ax[1,0])
plt_dispersion(gapdf2_X['n_clusters'], _gradient_x_ref, 'Reference X gradient', ax[1,1])
plt_dispersion(gapdf2_X['n_clusters'], _gradient_x_orig, 'Observed X gradient', ax[1,1])
ax[0,1].set_ylabel(None)
ax[1,1].set_ylabel(None)
ax[1,0].set_ylabel('Gradient')
ax[1,0].set_title(None)
ax[1,1].set_title(None)
ax[0,0].set_xlabel(None)
ax[0,1].set_xlabel(None)

del ax

#%% Predicting class labels
num_k_user = int(input('User Defined Number of Classes : '))

"""K-Means"""
kmeans = KMeans(n_clusters=num_k_user).fit(X)
_classes_means = kmeans.predict(X)
kmeans_mds = KMeans(n_clusters=num_k_user).fit(X_reduced_mds)
_classes_means_MDS = kmeans_mds.predict(X_reduced_mds)
#kmeans.labels_

"""Agglomerative"""
agglo_cluster = AgglomerativeClustering(n_clusters=num_k_user,affinity='euclidean',
                                        linkage='ward')
agglo_cluster.fit(X)
_classes_agglo = agglo_cluster.labels_
"""euclidean, l1, l2, manhattan, cosine, precomputed : possible affinity"""
"""linkage : ward, complete, average, single"""


"""Save"""
df_clean = df_clean.join(pd.Series(data=_classes_means, name='KMeans', index=df_clean.index))
df_clean = df_clean.join(pd.Series(data=_classes_agglo, name='Agglo', index=df_clean.index))
df_clean = df_clean.join(pd.Series(data=_classes_means_MDS, name='KMeans_MDS', index=df_clean.index))

df_clean = df_clean.sort_values(by=['NAME'])
df_clean.to_csv(r"D:\Z - Saved SQL Databases\test.csv")


#%% Testing

#Vector norms ; 
#In general, the distance bewteen two vectors can be found by finding the 
#norm of the difference between two vectors
#for example let u, v occupy R^n. the distance bewteen u & v = d(u,v) = norm(u-v) = norm(v-u):
#= sqrt((u1-v1)**2 + (u2-v2)**2 + .... (un - vn)**2)

distances = np.linalg.norm((X - X[0]), axis=1)