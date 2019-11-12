# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:11:35 2019

Write TFRecord objects for modeling. Each file written will be used for a 
different model - this module will be mostly used for writing TFRecord 
objects used to model hyperparameters : 
a)
by_size
b)
n_components
c)
reduce

@author: z003vrzk
"""

import tensorflow as tf
import pickle
import numpy as np



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


def serialize_features(document, label_names):
    """Create a serialized tf.train.Example proto from a document stored in
    mongo-db
    input
    -------
    document : (dict) must have db_features key
    label_names : (list) names of label to be included in example proto. 
    Should be a list including at least one of ['by_size','n_components','reduce']
    output
    -------
    feature_proto_str : (bytes) serialized context features in tf.train.Example
    proto
    
    the context_feature spec is of the form for context features
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
    features_dict = {}
    enforce_order = ['n_instance', 'n_features', 'len_var', 'uniq_ratio', 
                        'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
                        'n_len6', 'n_len7']
    # Add database features into TFRecord
    for key in enforce_order:
        # Serialize context features
        val = document['db_features'].get(key, 0.0) # Enforce float
        features_dict[key] = tf_feature_mapper(val, dtype=float)
        
    # Add label into TFRecord
    for key in label_names:
        label_val = document['best_hyper'].get(key, 0.0)
        
        # Enforce data types for features
        if key == 'by_size':
            label_val = bool(label_val)
            
        elif key == 'reduce':
            label_val = str(label_val)
            label_val = label_val.encode('utf-8') # Bytes
            # print(label_val, type(label_val), tf_feature_mapper([label_val]))
            
        elif key == 'n_components':
            label_val = int(label_val)
            
        else:
            raise ValueError('The passed key %d is not being handled by\
                             serialize_features. Please correct label_names\
                             or update the function')
        
        features_dict[key] = tf_feature_mapper([label_val], dtype=type(label_val))
    
    # Create serialized context tf.train.Example
    feature_proto = tf.train.Example(
            features=tf.train.Features(feature=features_dict))
    feature_proto_str = feature_proto.SerializeToString() # Serialized
    
    return feature_proto_str

def get_test_train_id(collection, train_pct=0.8):
    """Returns Mongo-db _id's for training and testing instances. document _ids
    are shuffled using sklearn's model_selection.train_test_split
    inputs
    -------
    collection : mongo collection object
    train_pct : (float) percent of docuemnt _ids to be considered for training
    outputs_
    -------
    (train_ids, test_ids) : list of training and testing _ids """
    
    ids = []
    _ids = collection.find({}, {'_id':1})
    for _id in _ids:
        ids.append(_id['_id'])
        
    permuted_sequence = np.random.permutation(ids)
    n_train = int(len(permuted_sequence) * train_pct)

    train_ids = permuted_sequence[:n_train]
    test_ids = permuted_sequence[n_train:]
    
    return train_ids, test_ids


def feature_columns():
    """Returns context feature names to column definitions.
    """
    n_instance = tf.feature_column.numeric_column(
        'n_instance', dtype=tf.float32, default_value=0.0)
    n_features = tf.feature_column.numeric_column(
        'n_features', dtype=tf.float32, default_value=0.0)
    len_var = tf.feature_column.numeric_column(
        'len_var', dtype=tf.float32, default_value=0.0)
    uniq_ratio = tf.feature_column.numeric_column(
        'uniq_ratio', dtype=tf.float32, default_value=0.0)
    n_len1 = tf.feature_column.numeric_column(
        'n_len1', dtype=tf.float32, default_value=0.0)
    n_len2 = tf.feature_column.numeric_column(
        'n_len2', dtype=tf.float32, default_value=0.0)
    n_len3 = tf.feature_column.numeric_column(
        'n_len3', dtype=tf.float32, default_value=0.0)
    n_len4 = tf.feature_column.numeric_column(
        'n_len4', dtype=tf.float32, default_value=0.0)
    n_len5 = tf.feature_column.numeric_column(
        'n_len5', dtype=tf.float32, default_value=0.0)
    n_len6 = tf.feature_column.numeric_column(
        'n_len6', dtype=tf.float32, default_value=0.0)
    n_len7 = tf.feature_column.numeric_column(
        'n_len7', dtype=tf.float32, default_value=0.0)
    
    # Labels
    by_size = tf.feature_column.numeric_column(
        'by_size', dtype=tf.int64)
    n_components = tf.feature_column.categorical_column_with_vocabulary_list(
        'n_components', [8, 0, 2], dtype=tf.int64)
    n_components_encoded = tf.feature_column.indicator_column(n_components)
    reduce = tf.feature_column.categorical_column_with_vocabulary_list(
        'reduce', ['MDS', 'TSNE', 'False'], dtype=tf.string)
    reduce_encoded = tf.feature_column.indicator_column(reduce)
    
    feature_cols = {
        'n_instance':n_instance,
        'n_features':n_features,
        'len_var':len_var,
        'uniq_ratio':uniq_ratio,
        'n_len1':n_len1,
        'n_len2':n_len2,
        'n_len3':n_len3,
        'n_len4':n_len4,
        'n_len5':n_len5,
        'n_len6':n_len6,
        'n_len7':n_len7,
        'by_size':by_size,
        'reduce':reduce_encoded,
        'n_components':n_components_encoded
        }
    
    return feature_cols
    

def tfrecord_reader(TFRecord_path):
    """Given a TFRecord serialized file, output a pandas dataframe. Useful
    for feature validation (data type and format)
    """
    # Create feature columns for parsing
    example_cols = list(feature_columns().values())
    
    # Whole TFRecord
    example_spec = tf.feature_column.make_parse_example_spec(
            example_cols)
    
    dataset = tf.data.TFRecordDataset(filenames=TFRecord_path)

    def _parse_fn_better(example_proto, example_spec):
        parsed_example = tf.io.parse_single_example(example_proto, example_spec)
        
        # pop label and return tuple of (features_dict, label_tensor)
#        label_tensor = parsed_example.pop(_LABEL_FEATURE)
        
        return parsed_example
    
    parsed_whole_dataset = dataset.map(lambda x: _parse_fn_better(x, example_spec))
    
    return parsed_whole_dataset


        












