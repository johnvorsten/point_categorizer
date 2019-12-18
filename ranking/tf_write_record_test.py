# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:16:15 2019

@author: z003vrzk
"""

# Thrid party imports
from pymongo import MongoClient
import tensorflow as tf
import numpy as np
import os

# Local imports
from tf_write_record import (serialize_features, 
                                get_test_train_id,
                                _bytes_feature,
                                feature_columns,
                                tfrecord_reader)

# Retrieve information from Mongo
client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

#%% Testing and writing

# Writing to a file
_test_file = r'data\JV_test_classify.tfrecords'
_train_file = r'data\JV_train_classify.tfrecords'
_train_pct = 0.8
if '_train_ids' not in locals():
    _train_ids, _test_ids = get_test_train_id(collection, 
                                              train_pct=_train_pct)
_train = False
_reciprocal = False
_text_features = False
_n_bins = 5
if _train:
    _savefile = _train_file
    _objectids = _train_ids
else:
    _savefile = _test_file
    _objectids = _test_ids

if os.path.isfile(_savefile):
    _confirm = input(f'{_savefile} already exists. Overwrite?')
    if _confirm not in ['Y','y','Yes','yes',True]:
        raise SystemExit('Script execution stopped to not overwrite file')

writer = tf.io.TFRecordWriter(_savefile)

for document in collection.find({'_id':{'$in':list(_objectids)}}):

    _hyperparameters = ['by_size','n_components','reduce']
    feature_proto_str = serialize_features(document, _hyperparameters)
    
    writer.write(feature_proto_str)

writer.close()


#%% Hand testing
    
tf.enable_eager_execution()
tf.executing_eagerly()
#tf.set_random_seed(1234)
tf.logging.set_verbosity(tf.logging.INFO)

_test_file = r'data\JV_test_classify.tfrecords'
_train_file = r'data\JV_train_classify.tfrecords'

example_cols = list(feature_columns().values())

# Whole TFRecord
example_spec = tf.feature_column.make_parse_example_spec(
        example_cols)  

# Reading from a file
dataset = tf.data.TFRecordDataset(filenames=_train_file)
string_record = next(iter(dataset)) # bytes object

# Dictionary of key:tensor
parsed_example = tf.io.parse_single_example(string_record, example_spec)

# Print combined dense input of parsed example
tf.keras.layers.DenseFeatures(example_cols)(parsed_example).numpy()

for string_record in iter(dataset):
    # Dictionary of key:tensor
    parsed_example = tf.io.parse_single_example(string_record, example_spec)
    
    # Print combined dense input of parsed example
    input_layer = tf.keras.layers.DenseFeatures(example_cols)
    dense_layer = input_layer(parsed_example).numpy()
    print(dense_layer)
    
# Print a specific column
_col_name = 'by_size'
_example_col = [_col for _col in example_cols 
                if _col.name.__contains__(_col_name)]

tf.keras.layers.DenseFeatures(_example_col)(parsed_example).numpy()



#%% Testing

tf.enable_eager_execution()
tf.executing_eagerly()
#tf.set_random_seed(1234)
tf.logging.set_verbosity(tf.logging.INFO)

_test_file = r'data\JV_test_classify.tfrecords'
_train_file = r'data\JV_train_classify.tfrecords'

parsed_dataset = tfrecord_reader(_train_file)

feature_dict = parsed_dataset.make_one_shot_iterator().get_next()

# This will give you the value of the indicator column - the value that the 
# indicator represetnts (not a one-hot encoded array)
tf.sparse.to_dense(feature_dict['n_components']).numpy() # Works with non-string
feature_dict['n_components'].values.numpy() # works with all
feature_dict['reduce'].values.numpy() # Works with string

# This will give you the actual one-hot encoded indicator column
tf.keras.layers.DenseFeatures(example_cols[12])(feature_dict).numpy()

# Whole shebang
for key, value in feature_dict.items():
    if hasattr(value, 'numpy'):
        print(key, value.numpy())
    else:
        _example_col = [_col for _col in example_cols 
                if _col.name.__contains__(key)]
        dense_col = tf.keras.layers.DenseFeatures(_example_col)
        print(key, dense_col(feature_dict).numpy(), value.values.numpy())





        
        
        