# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:28:00 2022

Context features: Features related to the group of points being clustered. 
Includes information like the number of points being clustered, measures of 
dispersion, and relative size of each point within the group of points

Peritem features: Features related to our question (which is 'which set of clustering algorithms and indexes will perform best on this group of points'). 
See HYPERPARAMETER_LIST which lists a chosen set of clustering algorithms 
(called 'hyperparameters' below) which are candidates for performing clustering 
on a group of points

How to predict with tensorflow ranking

@author: jvorsten
"""

# Python imports
import configparser
import pickle
import os, sys
from typing import Dict, Union, Iterable

# Third party imports
import sqlalchemy
import tensorflow as tf

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
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev,
                                              Customers, 
                                              ClusteringHyperparameter, 
                                              Labeling)
from transform_ranking import Transform
from Labeling import ClusteringLabels
from rank_write_record import (
    tf_feature_mapper, 
    _bytes_feature)
from Labeling import HYPERPARAMETER_SERVING_FILEPATH, ClusteringLabels

# Declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

Insert = extract.Insert(server_name=server_name,
                        driver_name=driver_name,
                        database_name=database_name)

"""Peritem clusterer hyperparameters used for prediction
# Return these with the output prediction. User will be able 
# To rank clusterer hyperparameters based on prediction"""
_default_peritem_features_file = r'../data/JV_default_serving_peritem_features'
with open(_default_peritem_features_file, 'rb') as f:
    # Import from file
    HYPERPARAMETER_LIST = pickle.load(f)
    
_LIST_SIZE_MODEL4 = 200
_MODEL4_DIRECTORY = 'final_model\\Run_20191024002109model4\\1572051525'

#%%

def serialize_context_from_dictionary(context_features: Dict[str, Union[int,float]]):
    """Create a serialized tf.train.Example proto from a dictionary of context
        features
    input
    -------
    context_features: (dict) Must contain the keys
        ['n_instance', 'n_features', 'len_var', 'uniq_ratio',
         'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
         'n_len6', 'n_len7']
    output
    ------
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
    enforce_order = {'n_instance', 'n_features', 'len_var', 'uniq_ratio',
                     'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
                     'n_len6', 'n_len7'}

    # Ensure integrity of context_features
    if enforce_order != set(context_features.keys()):
        msg="Invalid key {} included in context_features"\
            .format(enforce_order.difference(database_features.keys()))
        raise KeyError(msg)

    # tf.train.Feature for each value passed in context_features
    for key in enforce_order:
        # Serialize context features
        val = context_features[key] # Enforce float
        context_dict[key] = tf_feature_mapper(float(val), dtype=float)

    # Create serialized context tf.train.Example
    context_proto = tf.train.Example(
            features=tf.train.Features(feature=context_dict))
    context_proto_str = context_proto.SerializeToString() # Serialized

    return context_proto_str

def serialize_examples_model4(hyperparameter_list: Iterable[Dict[str,str]], 
                              list_size: int):
    """
    Create a list of serialized tf.train.Example proto. 
    Use this for serving & predicting models. Not used for training; the 
    "relevance" label is not included. 
    Per-item example features are chosen as a fixed-set of clustering 
    hyperparameters based on list_size; they are not based on how each document 
    was actually clustered.
    peritem features are generated from a constant list of clusterer
    hyperparameters.
    
    input
    -------
    hyperparameter_list: (iterable of dictionaries) each dictionary defining a
    set of clustering 'hyperparameters'. Each dictionary of hyperparameters
    defines a group of inputs used to cluster a group of objects. Example
        [{'by_size': 'False',
          'n_components': '8',
          'reduce': 'MDS',
          'clusterer': 'kmeans',
          'index': 'optk_MDS_gap_Tib'},
         {'by_size': 'True',
          'n_components': '0',
          'reduce': 'False',
          'clusterer': 'kmeans',
          'index': 'optk_X_gap*_max'}, {...}]
    list_size: (int) Number of examples to include. This must be determined
        from the input size of the ranking model. Optionally, pass False
        if you do not want peritem examples to be padded
    
    output
    -------
    peritem_list: (list) a list of serialized per-item (exmpale) featuers
      in the form of tf.train.Example protos
    """
    
    peritem_features = [] # For iterating through and saving
    for hyperparameters in hyperparameter_list:
        single_item_feature_dict = {} # Append to list
        
        # Append all features to peritem features
        by_size = str(hyperparameters['by_size'])
        clusterer = str(hyperparameters['clusterer'])
        index = str(hyperparameters['index'])
        n_components = str(hyperparameters['n_components'])
        reduce = str(hyperparameters['reduce'])
        
        single_item_feature_dict['by_size'] = by_size.encode()
        single_item_feature_dict['n_components'] = n_components.encode()
        single_item_feature_dict['reduce'] = reduce.encode()
        single_item_feature_dict['clusterer'] = clusterer.encode()
        single_item_feature_dict['index'] = index.encode()
        peritem_features.append(single_item_feature_dict)
            
    def _pad(peritem_features, pad_value=b''):
        # Pad with empty string
        cur_size = len(peritem_features)
        new_size = list_size
        
        example_keys = list(peritem_features[0].keys())
        blank_exmaple_dict = {}
        for key in example_keys:
            blank_exmaple_dict[key] = pad_value
            
        for i in range(new_size-cur_size):
            peritem_features.append(blank_exmaple_dict)
            
        return peritem_features
    
    def _trim(peritem_features):
        # Return the first _list_size of peritem_features
        new_size = list_size
        trimmed_peritem_features = peritem_features[:new_size]
        return trimmed_peritem_features
    
    if bool(list_size): # Only pad if list_size is defined
        if (list_size > len(peritem_features)):
            peritem_features = _pad(peritem_features)
        elif (list_size < len(peritem_features)):
            peritem_features = _trim(peritem_features)
        
    # Create a list of serialized per-item features
    peritem_list = []
    
    for single_item_feature_dict in peritem_features:
        peritem_dict = {}
        
        # Add peritem features to dictionary
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

#%%

# Load an example from SQL database
customer_id = 15
sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
database = Insert.pandas_select_execute(sel)
sel = sqlalchemy.select([Customers.name]).where(Customers.id.__eq__(customer_id))
customer_name = Insert.core_select_execute(sel)[0].name

# Transformation pipeline
full_pipeline = Transform.get_ranking_pipeline()

# Dictionary with keys ['n_instance', 'n_features', 'len_var', 'uniq_ratio',
#                    'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
#                    'n_len6', 'n_len7']
database_features = ClusteringLabels.get_database_features(
    database,
    full_pipeline,
    instance_name=customer_name)
database_features.pop('instance')

#1. Context features (bytes object)
context_proto_str = serialize_context_from_dictionary(database_features)

#2. Peritem features (bytes object)
peritem_features = serialize_examples_model4(
    HYPERPARAMETER_LIST,
    list_size=_LIST_SIZE_MODEL4)

# Prepare serialized feature spec for EIE format
serialized_dict = {'serialized_context':_bytes_feature([context_proto_str]),
                   'serialized_examples':_bytes_feature(peritem_features)
                   }

# Convert to tf.train.Example object
serialized_proto = tf.train.Example(
        features=tf.train.Features(feature=serialized_dict))
serialized_example_in_example = serialized_proto.SerializeToString()

"""
#TODO
(done) Create pipeline for ranking features
(done) Create function serialize_context_from_dictionary, which assumes a dictionary 
of datbase features is passed
(done) Create function to serialize examples (without mongo document)
    see serialize_examples_model4
Load previously serialized model

Define dependencies
Isolate these models into a separate project
"""

# Load previously serialized model from V1 tensorflow
imported = tf.saved_model.load(_MODEL4_DIRECTORY)
pruned = imported.prune('Placeholder:0', 'groupwise_dnn_v2/accumulate_scores/div_no_nan:0')



#%% Worked using tensorflow 1.14.0

if __name__ == '__main__':
    
    # Placeholder:0 is the feature column name in serving_input_receiver_fn
    input_feed_dict = {'Placeholder:0':[serialized_example_in_example]}
    
    # Ensure the saved model is available
    tf.saved_model.contains_saved_model(_MODEL4_DIRECTORY) # True
    
    # Only 'serve' is a valid MetaGraphDef - dont use 'predict'
    graph_model4 = tf.Graph()
    with tf.compat.v1.Session(graph=graph_model4) as sess:
        import_model4 = tf.saved_model.load(
            sess, 
            ['serve'], 
            _MODEL4_DIRECTORY)
        outputs_model4 = sess.run(
            'groupwise_dnn_v2/accumulate_scores/div_no_nan:0', 
            feed_dict=input_feed_dict)
        
    print('outputs_v2: ', outputs_model4)