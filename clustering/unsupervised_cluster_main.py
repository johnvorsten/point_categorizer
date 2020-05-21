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
else:
    # The module is not a top-level module
    from ..pipeline import transform_pipeline
    from ..extract import extract


#%% New

# Instantiate local classes
Transform = transform_pipeline.Transform()
# Create 'clean' data processing pipeline
cleaning_pipeline = Transform.cleaning_pipeline(remove_dupe=False,
                                      replace_numbers=False,
                                      remove_virtual=True)
# Create pipeline specifically for clustering text features
text_pipe = Transform.text_pipeline(vocab_size='all',
                                   attributes='NAME',
                                   seperator='.',
                                   heirarchial_weight_word_pattern=True)
# Retrieve information from SQL
server_name = '.\DT_SQLEXPR2008'
driver_name = 'SQL Server Native Client 10.0'
database_name = 'Clustering'
Insert = extract.Insert(server_name=server_name,
                        driver_name=driver_name,
                        database_name=database_name)

# Get a specific customer
customer_name = r'D:\Z - Saved SQL Databases\44OP-148387_BU_FORMAN_LAB\JobDB.mdf'
sel = sqlalchemy.select([Customers]).where(Customer.name.__eq__(customer_name))
with Insert.engine.connect() as connection:
    customer_result = connection.execute(sel).fetchall()

if customer_result.__len__() == 0:
    # No result found
    raise ValueError("No customer named {} exists in database".format(customer_name))
else:
    customer_row = customer_result[0]

# Get related points into dataframe
sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_row.id))
df = pd.read_sql(sel, Insert.engine)

# Run dataframe through pipeline
df_clean = cleaning_pipeline.fit_transform(df)
X = text_pipe.fit_transform(df_clean).toarray()
# word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
# df_text = pd.DataFrame(X, columns=word_vocab)

# Perform clustering
sel = sqlalchemy.select([Clustering]).where(Clustering.customer_id.__eq__(customer_row.id))
with Insert.engine.connect() as connection:
    res = connection.execute(sel).fetchall()
correct_n_clusters = res.correct_k


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


#%%

# Instantiate local classes
JVDBPipe = transform_pipeline.JVDBPipe()
# Create 'clean' data processing pipeline
cleaning_pipeline = JVDBPipe.cleaning_pipeline(remove_dupe=False,
                                      replace_numbers=False,
                                      remove_virtual=True)
# Create pipeline specifically for clustering text features
text_pipe = JVDBPipe.text_pipeline(vocab_size='all',
                                   attributes='NAME',
                                   seperator='.',
                                   heirarchial_weight_word_pattern=True)
# Retrieve information from Mongo
client = MongoClient('localhost', 27017)
db = client['master_points']
collection_raw = db['raw_databases']
collection_clustered = db['clustered_points']

_cursor = collection_raw.find()
for document in _cursor:

    # pass data through cleaning and text pipeline
    database = pd.DataFrame.from_dict(document['points'],
                                      orient='columns')

    # Test if document already exists
    exists_cursor = collection_clustered.find({'database_tag':document['database_tag']}, {'_id':1})
    if exists_cursor.alive:
        # Dont recalculate
        continue

    df_clean = cleaning_pipeline.fit_transform(database)
    X = text_pipe.fit_transform(df_clean).toarray()
    #_word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
    #df_text = pd.DataFrame(X, columns=_word_vocab)


    # perform clustering
    actual_n_clusters = document['correct_k']
    if actual_n_clusters == 1:
        # Dont cluster - why would you?
        prediction_agglo = np.ones((X.shape[0]))

    if X.shape[0] <= 3:
        # Dont cluster - just pass 1 cluster total
        prediction_agglo = np.ones((X.shape[0]))

    else:
        # Cluster
        agglomerative = AgglomerativeClustering(n_clusters=actual_n_clusters,
                                                affinity='euclidean',
                                                linkage='ward')
        prediction_agglo = agglomerative.fit_predict(X)

    # Format of clustered_points_dict is {setid:}
    clustered_points_list = []
    for setid in set(prediction_agglo):
        cluster_dict = {}
        indicies = (prediction_agglo==setid)
        cluster_names = df_clean.iloc[indicies]
        points_dict = cluster_names.to_dict(orient='list')
        cluster_dict['points'] = points_dict

        # Add entries into clustered_points_dict
        clustered_points_list.append(cluster_dict)

    try:
        database_tag = document['database_tag']
        collection_clustered.insert_one({'clustered_points':clustered_points_list,
                                         'database_tag':database_tag,
                                         'correct_k':actual_n_clusters})

    except pymongo.errors.WriteError:
        continue

#%% TestUnsupervisedClusterPoints

from UnsupervisedCluster import UnsupervisedClusterPoints

# Create list of 5 best hyperparam dicts
best_hyperparam_list = [{'by_size': False,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Ratkowsky'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Cindex'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'CCC'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Silhouette'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Hartigan'}]



client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

_cursor = collection.find()

#for document in _cursor:
document = next(_cursor)

myClusterer = UnsupervisedClusterPoints()
database_iterator = myClusterer.split_database_on_panel(document)
#for sub_database in database_iterator:
sub_database = next(database_iterator)

my_pipeline = myClusterer.make_cleaning_pipe(remove_dupe=False,
                 replace_numbers=False,
                 remove_virtual=True,
                 vocab_size='all',
                 attributes='NAME',
                 seperator='.',
                 heirarchial_weight_word_pattern=True)
database, df_clean, X = my_pipeline(sub_database,
                                    input_type='DataFrame')




test_list = [{'by_size': False,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'all'}]

result = myClusterer.cluster_with_hyperparameter_list(test_list,
                                                      X)




