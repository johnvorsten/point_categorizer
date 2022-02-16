# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 17:28:00 2022

Context features: Features related to the group of points being clustered. 
Includes information like the number of points being clustered, measures of 
dispersion, and relative size of each point within the group of points

Peritem features: Features related to our question, which is 'which set of 
clustering algorithms and indexes will perform best on this group of points'. 
See HYPERPARAMETER_LIST which lists a chosen set of clustering algorithms 
(called 'hyperparameters' below) which are candidates for performing clustering 
on a group of points

# Legacy information
_MODEL4_DIRECTORY = './final_model/Run_20191024002109model4/1572051525'
# I do not know what this file is for, or what it does
HYPERPARAMETER_SERVING_FILEPATH = r'../data/serving_hyperparameters.dat'

@author: jvorsten
"""

# Python imports
import configparser
import pickle
from typing import Dict, Union, Iterable, List
import base64

# Third party imports
import tensorflow as tf
import requests
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

# Local imports
from rank_write_record import (
    tf_feature_mapper, 
    _bytes_feature)

# Declarations
"""Peritem clusterer hyperparameters used for prediction
# Return these with the output prediction. User will be able 
# To rank clusterer hyperparameters based on prediction"""
DEFAULT_PERITEM_FEATURES_FILE = r'./default_serving_peritem_features'
with open(DEFAULT_PERITEM_FEATURES_FILE, 'rb') as f:
    # Import from file
    HYPERPARAMETER_LIST = pickle.load(f)
    
LIST_SIZE_MODEL4 = 200

# Serving configuration
config = configparser.ConfigParser()
config.read(r'./serving_config.ini')
RANKING_MODEL_URL = config['ranking']['RANKING_MODEL_URL']

# FastAPI application
app = FastAPI(title=config['FastAPI']['title'],
              description=config['FastAPI']['description'],
              version=config['FastAPI']['version'],
              )


#%%

class DatabaseFeatures:
    instance:str
    n_instance:int
    n_features:int
    len_var:float
    uniq_ratio:float
    n_len1:float
    n_len2:float
    n_len3:float 
    n_len4:float 
    n_len5:float 
    n_len6:float 
    n_len7:float
    

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
            .format(enforce_order.difference(context_features.keys()))
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

def serialize_example_in_example(database_features: Dict[str, float],
                                 hyperparameter_list: Iterable[Dict[str,str]],
                                 list_size: int):
    
    #1. Context features (bytes object)
    serialized_context = serialize_context_from_dictionary(database_features)

    #2. Peritem features (bytes object)
    serialized_peritem = serialize_examples_model4(
        hyperparameter_list,
        list_size=list_size)

    # Prepare serialized feature spec for EIE format
    serialized_dict = {'serialized_context':_bytes_feature([serialized_context]),
                       'serialized_examples':_bytes_feature(serialized_peritem)
                       }

    # Convert to tf.train.Example object
    serialized_proto = tf.train.Example(
            features=tf.train.Features(feature=serialized_dict))
    serialized_example_in_example = serialized_proto.SerializeToString()
    
    return serialized_example_in_example

def cached_serialize_example_in_example(database_features: Dict[str, float],
                                        serialized_peritem: bytes):
    """Unfortunately, the HYPERPARAMETER_LIST object is not hashable
    unless I convert each dictionary to a static dictionary, and the list
    to a vector (lru caching requires hashable objects)"""
    
    #1. Context features (bytes object)
    serialized_context = serialize_context_from_dictionary(database_features)

    #2. Peritem features (bytes object)
    # Should be constant - cache the results for faster execution

    # Prepare serialized feature spec for EIE format
    serialized_dict = {'serialized_context':_bytes_feature([serialized_context]),
                       'serialized_examples':_bytes_feature(serialized_peritem)
                       }

    # Convert to tf.train.Example object
    serialized_proto = tf.train.Example(
            features=tf.train.Features(feature=serialized_dict))
    serialized_example_in_example = serialized_proto.SerializeToString()
    
    return serialized_example_in_example

def get_top_n_hyperparameters(n: int,
                              hyperparameter_list: List[Dict[str,str]],
                              scores: List[float]):
    """Return the top 5 hyperparameter sets based on a list of scores
    inputs
    -------
    hyperparameter_list: (list of dictionary) each entry """
    idx = sorted(range(len(scores)), key=scores.__getitem__)
    
    top_n_hyperparameters = []
    worst_n_hyperparameters = []
    for i in range(n):
        # Top n predictions
        top_n_hyperparameters.append(HYPERPARAMETER_LIST[idx[-i]])
        # Worst n predictions
        worst_n_hyperparameters.append(HYPERPARAMETER_LIST[idx[i]])
    
    return top_n_hyperparameters, worst_n_hyperparameters

#%% FastAPI declaration

SERIALIZED_PERITEM = serialize_examples_model4(
    HYPERPARAMETER_LIST,
    list_size=LIST_SIZE_MODEL4)
    
class RawInputData(BaseModel):
    """Raw input data from HTTP Web form
    inputs description
    -------
    n_instance: number of instances within data
    n_features: number of features per instance
    len_var: variance of lenght of categorical text features
    uniq_ratio: ratio of (number of instances / number of features)
    n_len1: number features with a 1-length word after tokenization of its text feature
    n_len2: number features with a 2-length word after tokenization of its text feature
    n_len3: ...
    n_len4:
    n_len5:
    n_len6:
    n_len7:
    """
    n_instance:float
    n_features:float
    len_var:float
    uniq_ratio:float
    n_len1:float
    n_len2:float
    n_len3:float 
    n_len4:float 
    n_len5:float 
    n_len6:float 
    n_len7:float

class ClusteringHyperparameters(BaseModel):
    """Contents of HYPERPARAMETER_LIST
    """
    by_size:str
    n_components:str
    reduce:str
    clusterer:str
    index:str
    
    
class PredictorOutput(BaseModel):
    """We will choose to output the top 5, and worst clustering hyperparameters
    predicted based on your database context features
    each output dictionary contains the keys 
    {'by_size':str,'n_components':str,'reduce':str,'clusterer':str,'index':str}
        """
    top_n: List[ClusteringHyperparameters]
    worst_n: List[ClusteringHyperparameters]


@app.on_event("startup")
async def startup():
    """How could I use this to add objects to the application state?
    app.state.PredictionObject = load_all_of_my_predictors()
    Then use app.state.PredictionObject in all of the endpoints?
    Access the app object by importing Request from fastapi, and adding
    it as a parameter in each endpoint like
    async def endpoint(request: Request, data_model:DataModel):"""
    pass

@app.get("/")
async def root():
    msg=("See API endpoints at /clustering-ranking/model4predict")
    return {"message":msg}

@app.post("/clustering-ranking/model4predict/", response_model=PredictorOutput)
async def clustering_ranking_model4(database_features: RawInputData):
    """Serve predictions from the CompNBPredictor"""

    # Serialize input context features into an example-in-example
    serialized_example_in_example = cached_serialize_example_in_example(
        database_features,
        SERIALIZED_PERITEM)
    
    # Tensorflow serving requires 
    json_data = {
      "signature_name":"serving_default",
      "instances":[
          {"b64":base64.b64encode(serialized_example_in_example)\
                       .decode('utf-8')}
              ]
          }
    
    # Response is a list of predicitons {'predictions': [[float,float,...]]}
    resp = requests.post(RANKING_MODEL_URL, json=json_data)
    relevancies = resp.json()['predictions'][0]
    top_5_hyperparameters, worst_5_hyperparameter = \
        get_top_n_hyperparameters(5, HYPERPARAMETER_LIST, relevancies)
    
    return PredictorOutput(top_n=top_5_hyperparameters,
                           worst_n=worst_5_hyperparameter)

#%%
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=config['FastAPI']['SERVE_PORT']
        )