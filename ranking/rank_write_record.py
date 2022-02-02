# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:41:39 2019

Introduction

Use this module for converting data in mongodb to tensorflow TFRecord
proto buffer objects.
The general workflow of this module will be
a) Create connection to mongodb
b) instantiate parameters for saving TFRecord objects (see rank_write_record_test)
c) iterate through chosen document_ids
d) for each document extract per-item and context features
e) serialize per-item and context features
f) Serialize per-item and context features into EIE format described
by tensorflow-ranking (see tensorflow-ranking git or rank_write_record_test)
g) Save TFRecord objects

Other considerations / enhancements

1.
Create a dummy dataset that I can use to see if the model is training correctly
The dummy dataset will have obvious relationships bewteen the clust_index
label and a feature of the dataset
(Example - if the dataset has mostly 1-length points then it will map to
kmeans + gap*max)

How do I do this?
a) Create a probabilistic mapping from the features to a label. The mapping
should contain some randomness, so sometimes features are not mapped
to their most probabilistic feature.
Gaussian distribution over input space? Initialize the probabilistic distribution
randomly, save it, then use it to predict dummy data?

! THE GAUSSIAN MIXTURE MODEL IDEA DOES NOT WORK - USUALLY ONLY A COUPLE
! CLASSES ARE PREDICTED

2.
What if I reduce the number of labels from continuous to divided into 5 categories
Maybe it will be easier for the model to learn?
This is saved on file ./data/JV_test_binned & ./data/JV_train_binned
See n_bins and reciprocal in relevance_scorer() below

3.
Have the option to save text per-item featuers (clusterer and index) or
encode them into one-hot vectors.  See example_text parameter in
serialize_examples() below

@author: z003vrzk
"""
# Python imports
import os
import sys
import configparser
from typing import Dict, Union

# Thrid party imports
import tensorflow as tf
import pickle
import numpy as np
import sqlalchemy

#Local Imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract import extract
from extract.SQLAlchemyDataDefinition import (Customers)

# Declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

#%% functions & methods

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def tf_feature_mapper(value, dtype=None):
    # Float -> tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    # int -> tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    # Create a dictionary mapper to map the function to data type
    dict_mapper = {int:_int64_feature,
                   float:_float_feature,
                   np.float64:_float_feature,
                   bytes:_bytes_feature,
                   np.bytes_:_bytes_feature,
                   bool:_int64_feature}

    # Make sure all list items are same dtype
    if hasattr(value, '__iter__'):
        if dtype:
            expected_type = dtype
        else:
            expected_type = type(value[0])
        all_same = all(isinstance(item, expected_type) for item in value)

        if not all_same: # Cast as the first type in iterable

            try:
                value = [expected_type(item) for item in value]
            except ValueError:
                raise ValueError('Cannot cast all items in passed list to \
                                 same type')

        return dict_mapper[expected_type](value)

    # Handle single values (not list)
    else:
        if dtype:
            expected_type = dtype
        else:
            expected_type = type(value)

        return dict_mapper[expected_type]([value])

    return

def relevance_scorer(relevance, n_bins=None, reciprocal=None):
    """Scale labels to a range (min,10] and invert the list to satisfy
    ranking label slope (h(x)|index is monotonically decreasing with
    increasing rank
    As a result, the label matrix is reversed so relevant items
    have a rank of 10, and irrelevant labels have a rank of min
    inputs
    -------
    n_bins: (int) number of divisions between ranked examples.
        For example, if n_bins = 3, exampels will be labeled [1,1,1,2,2,2,3,3,3]
    reciprocal: (bool) label examples based on inverse of
        loss metric (see Labeling.py)
    relevance: (list | iterable) of relevance measurements
        (loss metric see Labeling.py)
    output
    -------
    relevance: scaled or ranked labels that are suitable intputs into
    tensorflow_ranking"""

    assert (bool(n_bins) ^ bool(reciprocal)), 'n_bins and reciprocal must not \
                                                both be defined'

    relevance = np.array(relevance)

    if reciprocal:
        # Replace small with ones (small goes to inf or larger number)
        ge_zero = relevance >= 0
        l_one = relevance < 1
        is_small = np.logical_and(ge_zero, l_one)

        ones = np.ones(is_small.shape)

        # Reciprocal to change label relevance (max relevance -> min rel.)
        col_reciprocal = np.reciprocal(relevance)

        # Replace inf (is_small) and padded values
        col_reciprocal = np.where(is_small, ones, col_reciprocal)

        # Scale to max of 1
        col_max = np.amax(col_reciprocal)
        col_reciprocal = col_reciprocal / col_max

        return list(col_reciprocal)

    # Rank relevance scores into binned categores ranging from 1:n_bins
    # returned ranks are reversed for decreasing labels for increasing
    # Ranks (less relevant examples get lower scores)
    elif n_bins:

        ranked_by_bin = []
        array_len = len(relevance)

        for n in range(1, n_bins+1):

            lower = int((n-1) * array_len / n_bins)
            upper = int((n) * array_len / n_bins)
            # relevance label should be float - not int
            ranks = [float(n)] * (upper - lower)
            ranked_by_bin.extend(ranks)

        # Ranking should be decreasing - highest value ranks at the beginning
        return list(reversed(ranked_by_bin))

    raise(ValueError('Invalid function arguments'))

def serialize_context(document: Dict[str, dict]):
    """Create a serialized tf.train.Example proto from a document stored in
    mongo-db
    input
    -------
    document: (dict) must have db_features key {'db_features': dict}
    output
    -------
    context_proto_str: (bytes) serialized context features in tf.train.Example
    proto

    the context_feature spec is of the form
    {'n_instance': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_features': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'len_var': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'uniq_ratio': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_len1': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_len2': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_len3': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_len4': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_len5': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_len6': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,)),
     'n_len7': FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=(0.0,))
     }"""

    # Create dictionary of context features
    context_dict = {}
    enforce_order = ['n_instance', 'n_features', 'len_var', 'uniq_ratio',
                        'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
                        'n_len6', 'n_len7']
    for key in enforce_order:
        # Serialize context features
        val = document['db_features'].get(key, 0.0) # Enforce float
        context_dict[key] = tf_feature_mapper(val, dtype=float)

    # Create serialized context tf.train.Example
    context_proto = tf.train.Example(
            features=tf.train.Features(feature=context_dict))
    context_proto_str = context_proto.SerializeToString() # Serialized

    return context_proto_str

def serialize_examples_v1(document,
                          example_features,
                          reciprocal=False,
                          n_bins=5,
                          text_features=False):
    """Create a list of serialized tf.train.Example proto from a document
      stored in mongo-db
      input
      -------
      document: (dict) must have ['encoded_hyper']['clust_index']['val']
          and ['hyper_labels'] fields
      example_features: (list) name of example feautres to include in per-item
          feature list. Must be one of 'clust_index' or 'all'. If 'clust_index' is
          chosen, only (clusterer, index) pairs are included. If 'all' then
          ['by_size','clusterer','n_components','reduce','index'] are all included
          as indicator columns.
      reciprocal: (bool) the reciprocal of scores will be used
      n_bins: (int) number of bins to place relevance in if reciprocal=False
      example_text: (bool) Process clust_index as text (TRUE) or load
          from encoded (True)
      output
      -------
      peritem_list: (list) a list of serialized per-item (exmpale) featuers
      in the form of tf.train.Example protos

      The per_item feature spec is of the form
      {'encoded_clust_index': FixedLenFeature(shape=(37,),
            dtype=tf.float32,
            default_value=np.array.zeros((1,37))),
       'relevance': FixedLenFeature(shape=(1,),
           dtype=tf.float32,
           default_value=(-1,))
       }

    """

    if text_features: # TEXT FEATURES
        # Create dictionary of peritem features (encoded cat_index)
        # Extract saved labels (text -> bytes)

        clusterer_list = document['best_hyper']['clusterer'] # list of bytes
        index_list = document['best_hyper']['index'] # list of bytes

        peritem_features = []
        for clusterer, index in zip(clusterer_list, index_list):
            peritem_features.append([clusterer.encode(), index.encode()])

    else: # ENCODED clust_index
        # Create dictionary of peritem features (encoded cat_index)
        # Labels are clust_index encoding - not example labels (relevance)
        peritem_features = pickle.loads(document['encoded_hyper']['clust_index']['val'])

    # Extract labels and relevance for iteration
    relevances = [subdict['loss']
        for key, subdict in document['hyper_labels'].items()]

    # Scale or transform relevance
    relevances = relevance_scorer(relevances,
                                  reciprocal=reciprocal,
                                  n_bins=n_bins)

    # Create a list of serialized per-item features
    peritem_list = []

    for text_feature, relevance_label in zip(peritem_features, relevances):
        peritem_dict = {}

        # Add peritem features to dictionary
        peritem_dict['relevance'] = tf_feature_mapper(relevance_label)
        peritem_dict['encoded_clust_index'] = tf_feature_mapper(text_feature)

        # Create EIE and append to serialized peritem_list
        peritem_proto = tf.train.Example(
                features=tf.train.Features(feature=peritem_dict))
        peritem_proto_str = peritem_proto.SerializeToString() # Serialized

        # List of serialized tf.train.Example
        peritem_list.append(peritem_proto_str)

    return peritem_list

def serialize_examples_v2(document,
                          reciprocal=False,
                          n_bins=5,
                          shuffle_peritem=True):
    """Create a list of serialized tf.train.Example proto from a document
      stored in mongo-db
      input
      -------
      document: (dict) must have ['encoded_hyper']['clust_index']['val']
          and ['hyper_labels'] fields
      reciprocal: (bool) the reciprocal of scores will be used. If True,
          n_bins must be False
      n_bins: (int) number of bins to place relevance in.
      shuffle_peritem: (bool) set to True if you want to shuffle peritem examples
          this will make peritem features in a non-default order
      output
      -------
      peritem_list: (list) a list of serialized per-item (exmpale) featuers
      in the form of tf.train.Example protos

      The per_item feature spec is of the form
      {'peritem_features': FixedLenFeature(shape=(37,),
            dtype=tf.float32,
            default_value=np.array.zeros((1,37))),
       'relevance': FixedLenFeature(shape=(1,),
           dtype=tf.float32,
           default_value=(-1,))
       }

    """

    assert bool(n_bins) ^ bool(reciprocal), 'n_bins and reciprocal arguments\
        must not both be defined. One must be False if the other is defined'

    # Create dictionary of peritem features based on function input
    # Extract saved labels (text -> bytes)
    # ALl per-item features are saved as text and encoded or transformed later
    hyper_labels = document['hyper_labels'] # Dictionary, keys are order

    peritem_features = [] # For iterating through and saving
    for key, subdict in hyper_labels.items():
        single_item_feature_dict = {} # Append to list

        # Append all features to peritem features
        by_size = str(subdict['by_size'])
        clusterer = str(subdict['clusterer'])
        index = str(subdict['index'])
        n_components = str(subdict['n_components'])
        reduce = str(subdict['reduce'])
        single_item_feature_dict['by_size'] = by_size.encode()
        single_item_feature_dict['n_components'] = n_components.encode()
        single_item_feature_dict['reduce'] = reduce.encode()
        single_item_feature_dict['clusterer'] = clusterer.encode()
        single_item_feature_dict['index'] = index.encode()
        peritem_features.append(single_item_feature_dict)

    # Extract labels and relevance for iteration
    relevances = [subdict['loss']
        for key, subdict in document['hyper_labels'].items()]

    # Scale or transform relevance
    relevances = relevance_scorer(relevances,
                                  reciprocal=reciprocal,
                                  n_bins=n_bins)

    if shuffle_peritem:
        # Shuffle per-item examples
        peritem_features_shuffled = np.random.permutation(peritem_features)
        relevances_shuffled = np.random.permutation(relevances)

        relevances = relevances_shuffled
        peritem_features = peritem_features_shuffled

    # Create a list of serialized per-item features
    peritem_list = []

    for single_item_feature_dict, relevance_label in zip(peritem_features,
                                                         relevances):
        peritem_dict = {}

        # Add peritem features to dictionary
        peritem_dict['relevance'] = tf_feature_mapper(
                relevance_label)
        peritem_dict['by_size'] = tf_feature_mapper(
                [single_item_feature_dict['by_size']])
        peritem_dict['n_components'] = tf_feature_mapper(
                [single_item_feature_dict['n_components']])
        peritem_dict['reduce'] = tf_feature_mapper(
                [single_item_feature_dict['reduce']])
        peritem_dict['clusterer'] = tf_feature_mapper(
                [single_item_feature_dict['clusterer']])
        peritem_dict['index'] = tf_feature_mapper(
                [single_item_feature_dict['index']])

        # Create EIE and append to serialized peritem_list
        peritem_proto = tf.train.Example(
                features=tf.train.Features(feature=peritem_dict))
        peritem_proto_str = peritem_proto.SerializeToString() # Serialized

        # List of serialized tf.train.Example
        peritem_list.append(peritem_proto_str)

    return peritem_list

def serialize_examples_from_dictionary(example_features,
                                       label_key,
                                       peritem_keys,
                                       reciprocal=False,
                                       n_bins=5,
                                       shuffle_peritem=True):
    """Create a list of serialized tf.train.Example proto from a document
      stored in mongo-db
      input
      -------
      example_features: (list) of dictionaries. Each dictionary contains keys
          of peritem features and relevance labels of each item.
          Dict values are converted to encoded and
          saved as serialized features. ALl per-item features are saved as
          text and encoded or transformed later.
          Example:
           [{'distance': 'euclidean',
          'by_size': False,
          'n_components': '0',
          'reduce': '0',
          'clusterer': 'kmeans',
          'index': 'gap_tib',
          'relevance': 8761.6},
         {'distance': 'euclidean',
          'by_size': False,
          'n_components': '8',
          'clusterer': 'average',
          'reduce': 'MDS',
          'index': 'Frey',
          'relevance': 8761.6}, ..., ]
     label_key: (str) key of relevance label. Should be something like
         'relevance' (see above)
     peritem_keys: (iter | list) of string keys of peritem features. Each
         key should be a key in example_features.
         Example peritem_keys = ['by_size','n_components',
                                 'clusterer','reduce','index']
     reciprocal: (bool) the reciprocal of scores will be used. If True,
          n_bins must be False
     n_bins: (int) number of bins to place relevance in.
     shuffle_peritem: (bool) set to True if you want to shuffle peritem examples
          this will make peritem features in a non-default order
      output
      -------
      peritem_list: (list) a list of serialized per-item (exmpale) featuers
          in the form of tf.train.Example protos

      The per_item feature spec is of the form
      {'peritem_features': FixedLenFeature(shape=(37,),
            dtype=tf.float32,
            default_value=np.array.zeros((1,37))),
       'relevance': FixedLenFeature(shape=(1,),
           dtype=tf.float32,
           default_value=(-1,))
       }

    """
    msg = "Example_features must be iterable, not type {}"
    assert hasattr(example_features, '__iter__'), msg.format(type(example_features))

    msg = ('n_bins and reciprocal arguments must not both be defined. ' +
          ' One must be False if the other is defined')
    assert bool(n_bins) ^ bool(reciprocal), msg

    # Make sure all example_feature dictionaries contain label_key and peritem_keys
    # Also make test if they are all contain the same keys
    keys = []
    for _example_dict in example_features:
        keys.append(frozenset(_example_dict.keys()))
    msg = "All dictionary keys in example_features must be the same. Found {}"
    assert set(keys).__len__() == 1, msg.format(set(keys))

    # Assert that peritem keys is a subset of all dictionary keys
    msg = "peritem_keys {} is not in keys in example_features keys {}"
    assert set(peritem_keys).issubset(set(keys[0])), msg.format(set(peritem_keys), set(keys[0]))

    # Assert that label_key is a subset of all dictionary keys
    msg = "peritem_keys {} is not in keys in example_features keys {}"
    assert set({label_key}).issubset(set(keys[0])), msg.format(set(label_key), set(keys[0]))

    # label_key and peritem_keys must be equal to keys in example_features
    msg = "peritem_keys and label_key union do not equal the set of passed feature keys"
    feature_set = set(peritem_keys)
    feature_set.add(label_key)
    assert (feature_set == set(keys[0])), msg

    peritem_features = [] # For iterating through and saving
    for peritem_feature_dict in example_features:
        single_item_feature_dict = {} # Append to list

        for feature_name in peritem_keys:
            # Append all features to peritem features
            feature_value = str(peritem_feature_dict[feature_name])
            single_item_feature_dict[feature_name] = feature_value.encode()

        peritem_features.append(single_item_feature_dict)

    # Extract labels and relevance for iteration
    relevances = []
    for peritem_feature_dict in example_features:
        relevance = peritem_feature_dict[label_key]
        relevances.append(relevance)

    # Scale or transform relevance
    relevances = relevance_scorer(relevances,
                                  reciprocal=reciprocal,
                                  n_bins=n_bins)

    if shuffle_peritem:
        # Shuffle per-item examples
        index = np.random.permutation(len(peritem_features))
        peritem_features_shuffled = [peritem_features[x] for x in index]
        relevances_shuffled = [relevances[x] for x in index]

        relevances = relevances_shuffled
        peritem_features = peritem_features_shuffled

    # Create a list of serialized per-item features
    peritem_list = []

    for single_item_feature_dict, relevance_label in zip(peritem_features,
                                                         relevances):
        peritem_dict = {}

        # Add peritem features to dictionary
        peritem_dict[label_key] = tf_feature_mapper(relevance_label)
        for key in peritem_keys:
            # TODO Add tf.train.Features as values instead of bytes feature
            peritem_dict[key] = tf_feature_mapper([single_item_feature_dict[key]])

        # peritem_dict.update(**single_item_feature_dict)

        # Create EIE and append to serialized peritem_list
        peritem_proto = tf.train.Example(
                features=tf.train.Features(feature=peritem_dict))
        peritem_proto_str = peritem_proto.SerializeToString() # Serialized

        # List of serialized tf.train.Example
        peritem_list.append(peritem_proto_str)

    return peritem_list

def get_train_test_id_mongo(collection, train_pct=0.8):
    """Returns Mongo-db _id's for training and testing instances. document _ids
    are shuffled using sklearn's model_selection.train_test_split
    inputs
    -------
    collection: mongo collection object
    train_pct: (float) percent of docuemnt _ids to be considered for training
    outputs_
    -------
    (train_ids, test_ids): list of training and testing _ids """

    ids = []
    _ids = collection.find({}, {'_id':1})
    for _id in _ids:
        ids.append(_id['_id'])

    permuted_sequence = np.random.permutation(ids)
    n_train = int(len(permuted_sequence) * train_pct)

    train_ids = permuted_sequence[:n_train]
    test_ids = permuted_sequence[n_train:]

    return train_ids, test_ids

def get_train_test_id_sql(train_pct=0.8):
    """Returns primary keys of all unique customers
    The complete set of customer_ids are split into training and testing
    sets
    inputs
    -------
    train_pct: (float) percent of docuemnt _ids to be considered for training
        outputs_
    -------
    (train_ids, test_ids): (list) of training and testing _ids """

    # Set up connection to SQL
    Insert = extract.Insert(server_name=server_name,
                            driver_name=driver_name,
                            database_name=database_name)

    # Query SQL for all customer primary keys
    sel = sqlalchemy.select([Customers.id])
    customer_ids = Insert.core_select_execute(sel)
    customer_ids = [x.id for x in customer_ids]

    # Permute all primary keys into training and testing sets
    index = np.arange(len(customer_ids))
    np.random.shuffle(index)
    n_train = int(len(customer_ids) * train_pct)
    train_index = index[n_train:]
    text_index = index[:n_train]

    train_ids = [customer_ids[idx] for idx in train_index]
    test_ids = [customer_ids[idx] for idx in text_index]

    return train_ids, test_ids

