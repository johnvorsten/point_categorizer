# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:20:55 2020

@author: z003vrzk
"""

# Python imports
import sys
import os
from collections import Counter
import traceback
import copy

# Third party imports
import tensorflow as tf
import numpy as np
import sqlalchemy
from sqlalchemy.sql import text as sqltext
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import ShuffleSplit

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract import extract
from transform import transform_pipeline
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev,
                          Customers, ClusteringHyperparameter, Labeling)
from MILCategorization import mil_load

#%%

# Load dataset
LoadMIL = mil_load.LoadMIL()
file_name = r'../data/MIL_dataset.dat'
dataset = LoadMIL.load_mil_dataset()
bags = dataset['dataset']
bag_labels = dataset['bag_labels']

# Split dataset
rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
train_index, test_index = next(rs.split(bags, bag_labels))
train_bags, train_bag_labels = bags[train_index], bag_labels[train_index]
test_bags, test_bag_labels = bags[test_index], bag_labels[test_index]

def split_bags_single_instance_generator(bags, bag_labels):
    """Convert a n x (m x p) array of bag instances into a k x p array of
    instances
    Lables are expanded for each instance in each bag"""

    for bag, label in zip(bags, bag_labels):
        # Unpack bag into instances

        instances = bag.toarray()
        labels = np.array([label].__mul__(instances.shape[0]))

        yield instances, labels

# Unpack bags into single instances for training and testing
test_iterator = split_bags_single_instance_generator(test_bags, test_bag_labels)
train_iterator = split_bags_single_instance_generator(train_bags, train_bag_labels)

# Initialize datasets
train_instances, train_labels = [], []
test_instances, test_labels = [], []

# Gather datasets
for _i in range(20):
    _test_instances, _test_labels = next(test_iterator)
    _train_instances, _train_labels = next(train_iterator)

    train_instances.append(_train_instances)
    train_labels.append(_train_labels)
    test_instances.append(_test_instances)
    test_labels.append(_test_labels)

train_instances = np.concatenate(train_instances)
train_labels = np.concatenate(train_labels)
test_instances = np.concatenate(test_instances)
test_labels = np.concatenate(test_labels)

# Tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_instances, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_instances, test_labels))

# Shuffle and batch dataset
BATCH_SIZE = 9999
SHUFFLE_BUFFER_SIZE = 100
train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
tds = train_dataset.__iter__()
_ins, _lab = next(tds)


#%%

"""Create a KNN Implementation in tensorflow"""

@tf.function
def l2_norm(x, y):
    """Find the L2 norm between x and y
    inputs
    ------
    x : (tf.Variable) of shape [1, n_features]
    y : (tf.Variable) of shape [n_training_instances, n_features]
    outputs
    -------
    l2_norm : (tf.Variable) of shape [n_training_instances, 1]"""
    l2_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.math.subtract(x, y)), axis=1))
    return l2_norm

@tf.function
def best_k_label(distances, labels, k):
    """Find the best label from 'labels' given an a tensor of distances

    inputs
    -------
    distances : (tf.Variable) of shape [n_training_instances]
        A tensor of distances between a test instance and training instances
    labels : (tf.Variable) of shape [n_training_instances]
        A tensor of labels describing the training features in distances
    k : (tf.Constant) Number of training instances to use in the kNN decision"""

    top_values, top_indicies = tf.math.top_k(tf.math.negative(distances), k=k)

    # From indicies find top k instance labels
    top_labels = tf.gather(train_labels, top_indicies, axis=0)
    values, indicies, counts = tf.unique_with_counts(top_labels)
    # Choose the most frequest value
    max_count_index = tf.math.argmax(counts)
    best_label = tf.gather(values, max_count_index)

    return best_label

tf.config.experimental_run_functions_eagerly(True)

#%% Test

N_FEATURES = train_bags[0].shape[1]

instance = tf.Variable(initial_value=tf.zeros(shape=(N_FEATURES)),
                       dtype=tf.float32, name='instance')
knn_set = tf.constant(value=train_instances[:100],
                      dtype=tf.float32, name='knn_set')
knn_labels = tf.constant(value=train_labels[:100],
                         dtype=tf.string, name='knn_labels')

instance.assign(test_instances[0])

distances_ = tf.math.sqrt(
    tf.math.reduce_sum(
        tf.math.square(
            tf.math.subtract(instance, knn_set)), axis=1))
distances = l2_norm(instance, knn_set) # These are equal :)

k = 10

top_values, top_indicies = tf.math.top_k(tf.math.negative(distances), k=k)

# From indicies find top k instance labels
top_labels = tf.gather(train_labels, top_indicies, axis=0)
values, indicies, counts = tf.unique_with_counts(top_labels)
# Choose the most frequest value
max_count_index = tf.math.argmax(counts)
best_label_ = tf.gather(values, max_count_index)
best_label = best_k_label(distances, knn_labels, k) # 'ahu'

# Evaluate accuracy
test_labels[0] # 'alarm' :(

#%% SKLearn














