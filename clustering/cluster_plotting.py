# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:44:19 2019

@author: z003vrzk
"""

# Python imports
import os
import sys

# Third party imports
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
from gap_statistic import OptimalK
import sqlalchemy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# Local imports
# from JVWork_UnsupervisedCluster import JVClusterTools
# from JVWork_WholeDBPipeline import JVDBPipe


if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter, Labeling)
from unsupervised_cluster import UnsupervisedClusterPoints, OptimalKCluster
from transform import transform_pipeline

Insert = extract.Insert(server_name='.\DT_SQLEXPR2008',
                        driver_name='SQL Server Native Client 10.0',
                        database_name='Clustering')
UnsupervisedCluster = UnsupervisedClusterPoints()


#%%

# Instantiate local classes
Transform = transform_pipeline.Transform()
# Create 'clean' data processing pipeline
clean_pipe = Transform.cleaning_pipeline(remove_dupe=False,
                                      replace_numbers=False,
                                      remove_virtual=True)

# Create pipeline specifically for clustering text features
text_pipe = Transform.text_pipeline(vocab_size='all',
                                   attributes='NAME',
                                   seperator='.',
                                   heirarchial_weight_word_pattern=True)

# Set up connection to SQL
Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                        driver_name='SQL Server Native Client 10.0',
                        database_name='Clustering')

# Get a points dataframe
customer_id = 15
sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
with Insert.engine.begin() as connection:
    # res = connection.execute(sel).fetchone()
    database = pd.read_sql(sel, connection)

df_clean = clean_pipe.fit_transform(database)
X = text_pipe.fit_transform(df_clean).toarray()
#_word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
#df_text = pd.DataFrame(X, columns=_word_vocab)

# Get number of clusters
sel = sqlalchemy.select([Customers])\
    .where(Customers.id.__eq__(customer_id))
with Insert.engine.begin() as connection:
    res = connection.execute(sel).fetchone()
    correct_k = res.correct_k

#%%

# Dimensionality reduction
X_reduced_mds = UnsupervisedCluster._dimensionality_reduction(X, method='MDS', n_components=2)
_max_clusters = UnsupervisedCluster._get_max_nc(X_reduced_mds)

# Optimalk clustering
optimalK = OptimalK(parallel_backend='multiprocessing')
optimalk_result_MDS = optimalK(X_reduced_mds, cluster_array=np.arange(1,_max_clusters,1))
optimalk_gap_values_MDS = optimalK.gap_df
optimalk_result_X = optimalK(X.astype(np.float32), cluster_array=np.arange(1,_max_clusters,1))
optimalk_gap_values_X = optimalK.gap_df

# Optimalk2 clustering
optimalK2 = OptimalKCluster()
optimalk_result_MDS2, optimalk_gap_values_MDS2 = optimalK2.optimalK(X_reduced_mds, nrefs=5, max_clusters=_max_clusters)
optimalk_result_X2, optimalk_gap_values_X2 = optimalK2.optimalK(X, nrefs=5, max_clusters=_max_clusters)

#%%

# Plot dimensionality reduced dataset
def plt_MDS(x, y, classes, artist):
    """
    inputs
    -------
    x : (array | list) one-dimensional of values
    y : (array | list) one-dimensional of values
    classes : array of classes for each (x,y)"""

    uniques = list(set(classes))
    colors = [np.array(plt.cm.viridis(i/float(len(uniques)))).reshape(1,-1)
              for i in range(len(uniques)+1)]
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

# Plot gap values
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
plt_gap(optimalk_gap_values_MDS['n_clusters'], optimalk_gap_values_MDS['gap_value'], optimalk_result_MDS, ax, label='Gap1_MDS', correct_k=correct_k)
plt_gap(optimalk_gap_values_X['n_clusters'], optimalk_gap_values_X['gap_value'], optimalk_result_X, ax, label='Gap1_X')
plt_gap(optimalk_gap_values_MDS2['n_clusters'], optimalk_gap_values_MDS2['gap_value'], optimalk_result_MDS2, ax, label='Gap2_MDS')
plt_gap(optimalk_gap_values_X2['n_clusters'], optimalk_gap_values_X2['gap_value'], optimalk_result_X2, ax, label='Gap2_X')

gap_fig = plt.figure(3)
ax = gap_fig.subplots(1,1)
plt_gap(optimalk_gap_values_MDS['n_clusters'], optimalk_gap_values_MDS['gap*'], optimalk_result_MDS, ax, label='Gap*1_MDS', correct_k=correct_k)
plt_gap(optimalk_gap_values_X['n_clusters'], optimalk_gap_values_X['gap*'], optimalk_result_X, ax, label='Gap*1_X')
plt_gap(optimalk_gap_values_MDS2['n_clusters'], optimalk_gap_values_MDS2['gap*'], optimalk_result_MDS2, ax, label='Gap*2_MDS')
plt_gap(optimalk_gap_values_X2['n_clusters'], optimalk_gap_values_X2['gap*'], optimalk_result_X2, ax, label='Gap*2_X')

# Plot Dispersion
"""Dispersion is the dreviative of gap values, and it is a measure of how
quickly the variance between clusters is reduced by adding more clusters
to the optimalk simulation

Dispersion is a rate of change, and the optimalk occurs in the 'elbow' of
dispersion"""
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

plt_dispersion(optimalk_gap_values_MDS2['n_clusters'], optimalk_gap_values_MDS2['ref_dispersion'], 'Reference MDS', ax[0])
plt_dispersion(optimalk_gap_values_MDS2['n_clusters'], optimalk_gap_values_MDS2['obs_dispersion'], 'Observed MDS', ax[0])
plt_dispersion(optimalk_gap_values_X2['n_clusters'], optimalk_gap_values_X2['ref_dispersion'], 'Reference X', ax[1])
plt_dispersion(optimalk_gap_values_X2['n_clusters'], optimalk_gap_values_X2['obs_dispersion'], 'Observed X', ax[1])


#Find derivatives
_gradient_mds_orig = np.gradient(optimalk_gap_values_MDS2['obs_dispersion'])
_gradient_mds_ref = np.gradient(optimalk_gap_values_MDS2['ref_dispersion'])
_gradient_x_orig = np.gradient(optimalk_gap_values_X2['obs_dispersion'])
_gradient_x_ref = np.gradient(optimalk_gap_values_X2['ref_dispersion'])

disp_fig2 = plt.figure(5)
ax = disp_fig2.subplots(2,2, sharex=True)
plt_dispersion(optimalk_gap_values_MDS2['n_clusters'], optimalk_gap_values_MDS2['ref_dispersion'], 'Reference MDS', ax[0,0])
plt_dispersion(optimalk_gap_values_MDS2['n_clusters'], optimalk_gap_values_MDS2['obs_dispersion'], 'Observed MDS', ax[0,0])
plt_dispersion(optimalk_gap_values_X2['n_clusters'], optimalk_gap_values_X2['ref_dispersion'], 'Reference X', ax[0,1])
plt_dispersion(optimalk_gap_values_X2['n_clusters'], optimalk_gap_values_X2['obs_dispersion'], 'Observed X', ax[0,1])
plt_dispersion(optimalk_gap_values_MDS2['n_clusters'], _gradient_mds_ref, 'Reference MDS gradient', ax[1,0])
plt_dispersion(optimalk_gap_values_MDS2['n_clusters'], _gradient_mds_orig, 'Observed MDS gradient', ax[1,0])
plt_dispersion(optimalk_gap_values_X2['n_clusters'], _gradient_x_ref, 'Reference X gradient', ax[1,1])
plt_dispersion(optimalk_gap_values_X2['n_clusters'], _gradient_x_orig, 'Observed X gradient', ax[1,1])
ax[0,1].set_ylabel(None)
ax[1,1].set_ylabel(None)
ax[1,0].set_ylabel('Gradient')
ax[1,0].set_title(None)
ax[1,1].set_title(None)
ax[0,0].set_xlabel(None)
ax[0,1].set_xlabel(None)

del ax
