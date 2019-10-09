# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:34:41 2019

@author: z003vrzk
"""
# Thrid party imports
from pymongo import MongoClient
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np

# Local imports
from rank_write_record import (serialize_examples, 
                                serialize_context, 
                                get_test_train_id,
                                _bytes_feature)




#%% Save text clusterer_index per-item features

# Retrieve information from Mongo
client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

# Writing to a file
_test_file = r'data\JV_test_text.tfrecords'
_train_file = r'data\JV_train_text.tfrecords'
_train_pct = 0.8
_train_ids, _test_ids = get_test_train_id(collection, train_pct=_train_pct)
_train = False
_reciprocal = False
_text_features = True
_n_bins = 5
if _train:
    _savefile = _train_file
    _objectids = _train_ids
else:
    _savefile = _test_file
    _objectids = _test_ids

writer = tf.io.TFRecordWriter(_savefile)

for document in collection.find({'_id':{'$in':list(_objectids)}}):

    context_proto_str = serialize_context(document)
    
    peritem_list = serialize_examples(document, 
                                      reciprocal=_reciprocal,
                                      n_bins=_n_bins,
                                      example_text=_text_features)
    
    # Prepare serialized feature spec for EIE format
    serialized_dict = {'serialized_context':_bytes_feature([context_proto_str]),
                       'serialized_examples':_bytes_feature(peritem_list)
                       }
    
    # Convert to tf.train.Example object
    serialized_proto = tf.train.Example(
            features=tf.train.Features(feature=serialized_dict))
    serialized_str = serialized_proto.SerializeToString()
    
    writer.write(serialized_str)

writer.close()

#%% Serve TFRecord for estimaton

"""The goal of this functino is to import a document from mongodb and
serve it for estimation with model_serving.
"""

"""
USE THIS IF I DO HAVE TO INPUT FUNCTIONS DIFFERENTLY

from rank_write_record import tf_feature_mapper
import pickle

def encode_input_transmitter_fn(document):
    pass

example_text = False



document = collection.find_one()

# Get context features for final TFRecord

# Create dictionary of context features
context_dict = {}
enforce_order = ['n_instance', 'n_features', 'len_var', 'uniq_ratio', 
                    'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
                    'n_len6', 'n_len7']
for key in enforce_order:
    # Serialize context features
    val = document['db_features'].get(key, 0.0) # Enforce float
    context_dict[key] = tf_feature_mapper(val, dtype=float)
    

# Get per-item features for final TFRecord

if example_text: # TEXT FEATURES
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
    
# Create a list of serialized per-item features
#peritem_list = []

#for text_feature, relevance_label in zip(peritem_features):
#    peritem_dict = {}
#    
#    # Add peritem features to dictionary
#    peritem_dict['encoded_clust_index'] = tf_feature_mapper(text_feature)
pass

# I need to flatten per-item features so they can be fed into the model
# as a dense layer
# What is the input size of the model?
# Input size depends on 
# a) encoded dimension (length of features)
# b) number of examples (this changes from example to example)



# The maximum number of documents per instance in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 180
_DIMENSION_SIZE = peritem_features.shape[1] # One hot encoded feature dimension
_ENCODING_DIMENSION = 5 # For text -> Encoded features
# TODO pass the correct dimension based on hyperparameters
if example_text:
    dimension = _ENCODING_DIMENSION
else:
    dimension = _DIMENSION_SIZE

def pad(peritem_features, n_documents, feature_dimension):
    # pad peritem_features with padding arrays
    flat_peritem = np.flatten(peritem_features)
    missing_docs = n_documents - peritem_features.shape[0]
    if example_text:
        padding_vals = np.array(([''] * missing_docs * feature_dimension))
    else:
        padding_vals = np.zeros((missing_docs, feature_dimension))
        padding_vals = np.flatten(padding_vals)
    return padding_vals

def truncate(peritem_features, n_documents):
    # only include top n documents
    pass

if peritem_features.shape[0] > _LIST_SIZE:
    flat_peritem = truncate(peritem_features, n_documents=_LIST_SIZE)
elif peritem_features.shape[0] < _LIST_SIZE:
    flat_peritem = pad(peritem_features, n_documents=_LIST_SIZE, dimension)
else:
    flat_peritem = np.flatten(peritem_features) # Row order

"""
#%% Retrieving written objects

_Encoded_labels_dimension = 37

def context_feature_columns():
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
    context_feature_cols = {
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
        'n_len7':n_len7
        }
    
    return context_feature_cols

def example_feature_columns():
    """Returns the example feature columns.
    """
    
    encoded_clust_index = tf.feature_column.numeric_column(
        'encoded_clust_index', 
        dtype=tf.float32, 
        shape=[_Encoded_labels_dimension],
        default_value=np.zeros((_Encoded_labels_dimension)))
    
    peritem_feature_cols = {
        'encoded_clust_index':encoded_clust_index
        }
    
    return peritem_feature_cols

context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())
  
  # tf.feature_column.NumericColumn
label_column = tf.feature_column.numeric_column(
    'relevance', dtype=tf.float32, default_value=-1)
  
example_feature_spec = tf.feature_column.make_parse_example_spec(
          list(example_feature_columns().values()) + [label_column])


#%% tf.python_io testing

"""
How can I decompose my TFRecord EIE object into its original form?
a) Load the example proto-buffer from memory
b) Parse exampe with tf.io.parse_example with the EIE feature spec
EIE Feature spec format : 
feature_spec = {
    "serialized_context": tf.io.FixedLenFeature([1], tf.string),
    "serialized_examples": tf.io.VarLenFeature(tf.string),
    }
c) Parse context and per-item (examples) features using 
    tfr.pyton.data.parse_from_example_in_example. Be sure to create 
    example_feature_specs and context_feature_specs using 
    tf.feature_column.make_parse_example_spec([feature_columns])


tf.io.parse_example(
serialized,
features,
name=None,
example_names=None)
Parses serialized example proto's given serialized
parses serialized examples into a dictionary mapping eys to tensor objects"""

def tf_io_parse_test():
    
    _file = r'data\JVSerialized.tfrecords'
    
    record_iterator = tf.python_io.tf_record_iterator(path=_file)
    string_record = next(record_iterator)
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    feature_spec = {
        "serialized_context": tf.io.FixedLenFeature([1], tf.string),
        "serialized_examples": tf.io.VarLenFeature(tf.string),
        }
    features = tf.compat.v1.io.parse_example([string_record], feature_spec)
    
    parsed_features = tf.parse_single_example(string_record, feature_spec)
    
    # See https://stackoverflow.com/questions/49588382/how-to-convert-float-array-list-to-tfrecord
    features = tfr.python.data.parse_from_example_in_example(
              [string_record],
              context_feature_spec=context_feature_spec,
              example_feature_spec=example_feature_spec)



#%% Creating dummy data set
"""

THIS DOES NOT WORK - KEEP FOR RECORDS

from sklearn.mixture import GaussianMixture
from pymongo import MongoClient
from rank_inputfn import context_feature_columns, example_feature_columns
import tensorflow_ranking as tfr
import tensorflow as tf

_file_name = r'./data/JVSerialized.tfrecords'
_LABEL_FEATURE = 'relevance'
_PADDING_LABEL = 0

# Reading from a file
record_iterator = tf.python_io.tf_record_iterator(path=_file_name)


with open(r'./data/dummy_gaussian_fit.dat', 'rb') as f:
    features_all = pickle.load(f)

_n_components = 5
_cv_type = 'full'

gmm2 = GaussianMixture(n_components=_n_components,
                      covariance_type=_cv_type,
                      warm_start=True,
                      n_init=2,
                      max_iter=20)


string_record = next(record_iterator) # bytes object


context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())
  
  # tf.feature_column.NumericColumn
label_column = tf.feature_column.numeric_column(
    _LABEL_FEATURE, dtype=tf.float32, default_value=_PADDING_LABEL)
  
example_feature_spec = tf.feature_column.make_parse_example_spec(
          list(example_feature_columns().values()) + [label_column])

features = tfr.data.parse_from_example_in_example([string_record],
                                 list_size=None,
                                 context_feature_spec=context_feature_spec,
                                 example_feature_spec=example_feature_spec)


with tf.Session() as sess:
    relevance = features.pop('relevance')
    relevance = relevance.eval()
    
    # Flatten context_input into one dimension
    context_input = [
          tf.keras.layers.Flatten()(features[name])
          for name in sorted(context_feature_columns())
      ]
    context_input = tf.squeeze(context_input)
    
    clust_index = features.pop('encoded_clust_index')
    clust_index = tf.squeeze(clust_index)
    _n_examples = clust_index.shape[0].value
    context_input = tf.stack([context_input] * _n_examples)
    
    # Combine per-item and context features into array of shape
    # (n_examples, context_feat + per_item_feat)
    output = tf.concat([context_input, clust_index], axis=1)
    output = output.eval()
    
" Predicting "
gmm2.fit(features_all)
gmm2.predict(output)


"""