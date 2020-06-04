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
import pickle
    from collections import Counter

#Third party imports
from pymongo import MongoClient
import sqlalchemy
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from bson.binary import Binary
# import tensorflow as tf

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

from ranking import Labeling
from transform import transform_pipeline
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter)
from extract.SQLAlchemyDataDefinition import Labeling as SQLTableLabeling
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

    unique_labels = Labeling.get_unique_labels()

    return unique_labels


def test_get_best_hyperparameter_for_dataset():

    hyperparameter_name = 'by_size'

    return None


def test_save_unique_labels():
    """Save a vocabulary of all hyperparameter tags"""

    unique_labels = test_get_unique_labels()
    Labeling.save_unique_labels(unique_labels)

    return None


def test_get_hyperparameters_serving():

    hyperparameters_serving = get_hyperparameters_serving()

    # Optionally save hyperparameters_serving
    # save_hyperparameters_serving(hyperparameters_serving)

    # Optinoally read hyperparameters from file
    # hyperparameters_serving = open_hyperparameters_serving()

    return hyperparameters_serving
