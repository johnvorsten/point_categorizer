# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:43:14 2019

@author: z003vrzk
"""

"""This class will use known or predicted number of clusters for a database
to partition the database into their partitions.
For example a database with point names :
BANC.RTU0202.CCT
BANC.EF0101.PRF
BANC.EF0101.RMT
BANC.EF0101.SS
BANC.EF0201.PRF
BANC.EF0201.SS
should be separated into BANC.RTU0202*, BANC.EF101*, BANC.EF0201* groups"""

# Python imports
import os
import sys

# Third party imports
from pymongo import MongoClient
import pymongo
import sqlalchemy
from kmodes.kmodes import KModes
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

    from transform.transform_pipeline import JVDBPipe
    from transform import transform_pipeline
    from extract import extract
    from extract.SQLAlchemyDataDefinition import (Customers, Points, Netdev,
                                                  ClusteringHyperparameter, Clustering,
                                                  TypesCorrection)
    from clustering import unsupervised_cluster
else:
    from ..pipeline.transform_pipeline import JVDBPipe
    from transform import transform_pipeline


#%%

def test_unsupervised_cluster():

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
    _word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
    df_text = pd.DataFrame(X, columns=_word_vocab)

    # Get number of clusters
    sel = sqlalchemy.select([Customers])\
        .where(Customers.id.__eq__(customer_id))
    with Insert.engine.begin() as connection:
        res = connection.execute(sel).fetchone()
        correct_k = res.correct_k

    if X.shape[0] <= 3 or correct_k == 1:
        # Dont cluster - just pass 1 cluster total
        prediction_agglo = np.ones((X.shape[0]))

    else:
        # Cluster
        agglomerative = AgglomerativeClustering(n_clusters=correct_k,
                                                affinity='euclidean',
                                                linkage='ward')
        prediction_agglo = agglomerative.fit_predict(X)

    return df_clean, prediction_agglo



def test_cluster_with_hyperparameters():
    """Test clustering with hyperparameters"""

    # Instantiate local classes
    Transform = transform_pipeline.Transform()
    UnsupervisedCluster = unsupervised_cluster.UnsupervisedClusterPoints()
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
    customer_id = 13
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    with Insert.engine.begin() as connection:
        # res = connection.execute(sel).fetchone()
        database = pd.read_sql(sel, connection)

    df_clean = clean_pipe.fit_transform(database)
    X = text_pipe.fit_transform(df_clean).toarray()
    #_word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
    #df_text = pd.DataFrame(X, columns=_word_vocab)

    hyperparameters = {'by_size': False,
      'distance': 'euclidean',
      'clusterer': 'ward.D',
      'n_components': 8,
      'reduce': 'MDS',
      'index': 'Ratkowsky'}

    result = UnsupervisedCluster.cluster_with_hyperparameters(hyperparameters, X)

    best_nc_df = result.best_nc_dataframe

    sel = sqlalchemy.select([Customers])\
        .where(Customers.id.__eq__(customer_id))
    with Insert.engine.begin() as connection:
        res = connection.execute(sel).fetchone()
        correct_k = res.correct_k

    return result


def test_cluster_with_hyperparameters2():

    # Instantiate local classes
    Transform = transform_pipeline.Transform()
    UnsupervisedCluster = unsupervised_cluster.UnsupervisedClusterPoints()
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
    customer_id = 13
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    with Insert.engine.begin() as connection:
        # res = connection.execute(sel).fetchone()
        database = pd.read_sql(sel, connection)

    df_clean = clean_pipe.fit_transform(database)
    X = text_pipe.fit_transform(df_clean).toarray()
    #_word_vocab = text_pipe.named_steps['WordDictToS

    hyperparameters = {'by_size': False,
                        'distance': 'euclidean',
                        'clusterer': 'ward.D',
                        'n_components': 8,
                        'reduce': 'MDS',
                        'index': 'Ratkowsky'}
    # Clean hyperparameters
    hyperparams = UnsupervisedCluster._parse_hyperparameter_dictionary(hyperparameters)

    # Perform dimensionality reduction on data
    X_dim_reduced = UnsupervisedCluster._dimensionality_reduction(X,
                                                   method=hyperparams['reduce'],
                                                   n_components=hyperparams['n_components'])

    # Conditionally call nbclust package or optimalk package
    # based on input clustering hyperparameters
    if hyperparams['index'] in UnsupervisedCluster.nbclust_indicies:
        # Cluster with nbclust and clustering algorithm
        min_nc = 3 # Static
        max_nc = UnsupervisedCluster._get_max_nc(X) # Based on actual data

        best_nc_df = UnsupervisedCluster._nbclust_calc(X_dim_reduced,
                                   index=hyperparams['index'],
                                   clusterer=hyperparams['clusterer'],
                                   distance=hyperparams['distance'],
                                   min_nc=min_nc,
                                   max_nc=max_nc)
    # Get number of clusters
    sel = sqlalchemy.select([Customers])\
        .where(Customers.id.__eq__(customer_id))
    with Insert.engine.begin() as connection:
        res = connection.execute(sel).fetchone()
        correct_k = res.correct_k
    print(correct_k)

    pass

