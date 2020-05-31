# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:23:50 2019

This module does a few things :

1)
Collects dataset features (context featuers) and saves them to mongo
2)
Extracts labels for all dataset (relevance labels per example)
3)
Extracts per_item features (features of clustering hyperparameters) from
datasets and saves them to mongo. This includes by_size, clusterer, index,
and reduce
4)
Encode and save clusterer & index labels.  Clusterer and index labels will
be encoded into multi-label features and used as per-item features
in ranking.
For example, a instance may be clusterered with the k-means algorithm, and
its optimal-k value infered with the Cindex metric. This is then encoded
into an array of shape [1, (n_clusterers + n_index)]. In my case, n_clusterers
is 4 (average, kmeans, ward.D, ward.D2), and n_index is 33 (there are
33 possible cluster metrics). The resulting encoded array is of shape
(n_examples, 37) where n_examples is the number of unique hyperparameter
combinations I calculated the dataset to be




@author: z003vrzk
"""
# Python imports
import pickle
import os
import sys

#Third party imports
from pymongo import MongoClient
import sqlalchemy
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

#Local Imports
# from Labeling import ExtractLabels
# from UnsupervisedCluster import JVClusterTools
# from JVWork_WholeDBPipeline import JVDBPipe
# from JVWork_AccuracyVisual import import_error_dfs
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from ranking import Labeling, get_unique_labels, save_unique_labels
from transform import transform_pipeline
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter, Labeling)
from clustering.accuracy_visualize import Record, get_records

# Local declarations
ExtractLabels = Labeling.ExtractLabels()

#%%

def test_get_database_features():

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

    full_pipeline = Pipeline([('clean_pipe', clean_pipe),
                              ('text_pipe',text_pipe),
                              ])
    # Set up connection to SQL
    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    # Get a points dataframe
    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    database = Insert.pandas_select_execute(sel)
    sel = sqlalchemy.select([Customers.name]).where(Customers.id.__eq__(customer_id))
    customer_name = Insert.core_select_execute(sel)[0].name

    database_features = ExtractLabels.get_database_features(database,
                                                            full_pipeline,
                                                            instance_name=customer_name)
    return database_features

#%%

def test_get_database_labels():

    # Set up connection to SQL
    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    # Get all records relating to one customer
    customer_id = 15
    sel = sqlalchemy.select([Clustering.id, Clustering.correct_k])\
        .where(Clustering.customer_id.__eq__(customer_id))
    res = Insert.core_select_execute(sel)
    primary_keys = [x.id for x in res]
    correct_k = res[0].correct_k

    sel = sqlalchemy.select([Customers.name]).where(Customers.id.__eq__(customer_id))
    customer_name = Insert.core_select_execute(sel)[0].name

    # Calculate ranking of all records
    records = get_records(primary_keys)
    best_labels = ExtractLabels.calc_labels(records, correct_k, error_scale=0.8, var_scale=0.2)

    return best_labels


def test_get_unique_labels():

    unique_labels = get_unique_labels()

    return unique_labels

#%% Save encoded labels
pass

from Labeling import get_unique_labels

from pymongo import MongoClient
from bson.binary import Binary
import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np

unique_labels = get_unique_labels()
client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

# TODO index, clusterer

clust_uniq = sorted(unique_labels['clusterer'])
ind_uniq = sorted(unique_labels['index'])

# Separate clusterer and indicies
one_hot = OneHotEncoder(categories=[clust_uniq, ind_uniq])


for document in collection.find():
    # Extract saved labels
    clust_labels = list(document['best_hyper']['clusterer']) # Maintains order
    idx_labels = list(document['best_hyper']['index'])

    # shape = [n_examples, 2]
    labels_array = np.array([clust_labels, idx_labels]).transpose()

    encoded = one_hot.fit_transform(labels_array).toarray()
    categories = one_hot.categories_
    encoded_pickle = Binary(pickle.dumps(encoded, protocol=2), subtype=128)
    cat_pickle = Binary(pickle.dumps(categories, protocol=2), subtype=128)

    collection.update_one({'_id':document['_id']},
                           {'$set':{'encoded_hyper.clust_index.cat':cat_pickle,
                                    'encoded_hyper.clust_index.val':encoded_pickle}})

#%% Save a vocabulary of all hyperparameter tags

def test_save_unique_labels():

    save_unique_labels()

    return None


#%% Save a list of clustering hyperparameters for later use in model serving
"""The ranking model imputs a tensor of context features and per-item features
The per-item features are clusterering hyperparameters turned to indicator
columns.
In order to do prediction on a new database, I must input the per-item
clustering hyperparameters into the model.
In training, I have been doing this with actual recorded hyperparameters
For prediction I must generate the clustering hyperparameters - the must
be known before
This module will generate an array of clustering hyperparameters like :
[['False', 'kmeans', '8', 'TSNE', 'optk_TSNE_gap*_max'],
 ['True', 'ward.D', '8', 'MDS', 'SDbw'],
 [...]]
This can be fed to tf.feature_columns or TFRecords in order to generate
inputs to a ranking model for prediction
"""

document = collection.find_one()

hyper_labels = document['hyper_labels']
_file_name = r'data/JV_default_serving_peritem_features'
peritem_features = []

for key, subdict in hyper_labels.items():
    peritem_dict = {}

    by_size = str(subdict['by_size'])
    clusterer = str(subdict['clusterer'])
    index = str(subdict['index'])
    n_components = str(subdict['n_components'])
    reduce = str(subdict['reduce'])

    # Add peritem features to dictionary
    peritem_dict['by_size'] = by_size
    peritem_dict['n_components'] = n_components
    peritem_dict['reduce'] = reduce
    peritem_dict['clusterer'] = clusterer
    peritem_dict['index'] = index

    peritem_features.append(peritem_dict)

# Write pickled list
with open(_file_name, 'wb') as f:
    pickle.dump(peritem_features, f)

# Test pickled list
with open(_file_name, 'rb') as f:
    test_list = pickle.load(f)

test_list == peritem_features # True



