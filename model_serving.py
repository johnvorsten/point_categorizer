# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:12:37 2019

@author: z003vrzk
"""
from rank_write_record import tf_feature_mapper
import pickle
from pymongo import MongoClient
import tensorflow as tf

# Local imports
from rank_write_record import (serialize_examples, 
                                serialize_context, 
                                get_test_train_id,
                                _bytes_feature)


client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

#%% 

_RECIPROCAL = False
_TEXT_FEATURES = False
_N_BINS = 5

def encode_input_transmitter_fn(document):
    
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
    serialized_example_in_example = serialized_proto.SerializeToString()
    
    return serialized_example_in_example

example_text = False



document = collection.find_one()









