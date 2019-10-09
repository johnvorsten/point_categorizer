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

# Thrid party imports
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
                   np.bytes_:_bytes_feature}
    
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
    n_bins : number of divisions between ranked examples. For example, if
    n_bins = 3 exampels will be labeled [1,1,1,2,2,2,3,3,3]
    reciprocal : label examples based on inverse of loss metric (see Labeling.py)
    relevance : list of relevance measurements (loss metric see Labeling.py)
    output
    -------
    relevance : scaled or ranked labels that are suitable intputs into 
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
        
        return list(reversed(ranked_by_bin))
    
    raise(ValueError('Invalid function arguments'))



def serialize_context(document):
    """Create a serialized tf.train.Example proto from a document stored in
    mongo-db
    input
    -------
    document : (dict) must have db_features key
    output
    context_proto_str : (bytes) serialized context features in tf.train.Example
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


def serialize_examples(document, 
                        reciprocal=False, 
                        n_bins=5, 
                        example_text=False):
    """Create a list of serialized tf.train.Example proto from a document 
      stored in mongo-db
      input
      -------
      document : (dict) must have ['encoded_hyper']['clust_index']['val']
      and ['hyper_labels'] fields
      reciprocal : the reciprocal of scores will be used 
      n_bins : (int) number of bins to place relevance in. 
      example_text : (bool) Process clust_index as text (TRUE) or load 
      from encoded (True)
      output
      -------
      peritem_list : (list) a list of serialized per-item (exmpale) featuers
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




    













