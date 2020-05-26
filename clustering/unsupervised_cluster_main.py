# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:43:14 2019

@author: z003vrzk
"""

# Python imports
import os
import sys

# Third party imports
from pymongo import MongoClient
import pymongo
from kmodes.kmodes import KModes
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import sqlalchemy

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

    from transform import transform_pipeline
    from extract import extract
    from extract.SQLAlchemyDataDefinition import (Customers, Points,
                                      Clustering, ClusteringHyperparameter)
    from clustering import unsupervised_cluster
else:
    # The module is not a top-level module
    from ..pipeline import transform_pipeline
    from ..extract import extract


#%%

"""Cluster with a pre-defined correct number of clusters
The clusterer chosen is agglomerative with euclidena affinity and ward linkage
"""

def cluster_agglomerative(X, correct_n_clusters, affinity='euclidean', linkage='ward'):
    """Cluster a given set of points with agglomerative method
    inputs
    ------
    X : (np.array) array of points to cluster. Shape (m,n) where m is the number
        of instances/points and n is the feature space
    correct_n_clusters : (int) Number of clusters to split instances/points on
    affinity : (str) distance measurement to split points on. See
    AgglomerativeClustering
    linkage : (str)
    outputs
    ------
    prediction : (np.array)"""
    if correct_n_clusters == 1 or X.shape[0] <= 3:
        # Dont cluster if there is only one system
        # Dont cluster - just pass 1 cluster total
        prediction_agglo = np.ones((X.shape[0]))
    else:
        # Cluster
        agglomerative = AgglomerativeClustering(n_clusters=correct_n_clusters,
                                                affinity='euclidean',
                                                linkage='ward')
        prediction_agglo = agglomerative.fit_predict(X)

    return prediction_agglo


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

result = cluster_agglomerative(X, correct_k,
                               affinity='euclidean', linkage='ward')


#%%
"""Cluster poitns databases based on given hyperparameters
Save the results to SQL"""

def main():

    # Hyperparameters
    hyperparams = {
        'by_size':False,
        'n_components':8,
        'reduce':'MDS',
        'clusterer':'ward.D',
        'distance':'euclidean',
        'index':'all'}

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

    # Clustering class
    UnsupervisedCluster = unsupervised_cluster.UnsupervisedClusterPoints()

    # Save hyperparameters to SQL
    # See if its already inserted
    sel = sqlalchemy.select([ClusteringHyperparameter]).where(
        sqlalchemy.sql.and_(ClusteringHyperparameter.by_size == hyperparams['by_size'],
                            ClusteringHyperparameter.clusterer == hyperparams['clusterer'],
                            ClusteringHyperparameter.distance == hyperparams['distance'],
                            ClusteringHyperparameter.reduce == hyperparams['reduce'],
                            ClusteringHyperparameter.n_components == hyperparams['n_components']))
    with Insert.engine.connect() as connection:
        res = connection.execute(sel).fetchall()

    if res.__len__():
        # Get hyperparameters id of existing hyperparameter set
        hyperparameter_id = res[0].id
    else:
        # Insert new object
        res = Insert.core_insert_instance(ClusteringHyperparameter, hyperparams)
        hyperparameter_id = res.inserted_primary_key[0]

    # Get customer list from SQL
    sel = sqlalchemy.select([Customers])
    customers = Insert.core_select_execute(sel)

    # Iterate through customers and cluster
    for customer in customers:

        # Get points from SQL
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer.id))
        database = Insert.pandas_select_execute(sel)
        if database.shape[0] == 0:
            print('Customer ID {} Skipped, points shape {}'.format(customer.id, database.shape[0]))
            continue
        else:
            df_clean = clean_pipe.fit_transform(database)
            X = text_pipe.fit_transform(df_clean).toarray()
            #_word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
            #df_text = pd.DataFrame(X, columns=_word_vocab)

        # NbClust clustering
        print('Customer ID {}\nDB Size : {}'.format(customer_id, X_reduced.shape))
        try:
            print('Starting NbClust')
            # Perform clustering with NbClust package
            result = UnsupervisedCluster.cluster_with_hyperparameters(hyperparams, X)
            best_nc_df = result.best_nc_dataframe
        except RRuntimeError as e:
            if str(e).__contains__('computationally singular'):
                # The eigenvalue matrix is singular. Reduce the number of dimensions
                _hyperparams = hyperparams
                _hyperparams['n_components'] = int(_hyperparams['n_components'] / 2)
                result = UnsupervisedCluster.cluster_with_hyperparameters(hyperparams, X)
                best_nc_df = result.best_nc_dataframe
            else:
                print(e)
                continue

        # Build dictionary for SQL
        sel = sqlalchemy.select([Customers]).where(Customers.id.__eq__(customer.id))
        with Insert.engine.connect() as connection:
            res = connection.execute(sel).fetchone()
            correct_k = res.correct_k
        values = best_nc_df.loc['Number_clusters'].to_dict()
        values['correct_k'] = correct_k
        values['customer_id'] = customer.id
        values['hyperparameter_id'] = hyperparameter_id
        n_lens = Clustering.get_n_len_features(X)
        for key, val in n_lens.items():
            values[key] = int(val)

        # Save results to SQL
        res = Insert.core_insert_instance(Clustering, values)
        print("Inserted {}".format(res.inserted_primary_key))

    pass







