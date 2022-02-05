# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:12:37 2019

@author: z003vrzk
"""
# Python imports
import os, sys
import pickle
from collections import namedtuple

# Third party imports
import tensorflow as tf
import numpy as np

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
        
from rank_write_record import (
    tf_feature_mapper,
    serialize_context, 
    _bytes_feature)
from Labeling import HYPERPARAMETER_SERVING_FILEPATH, ClusteringLabels

# Declarations
"""Peritem clusterer hyperparameters used for prediction
# Return these with the output prediction. User will be able 
# To rank clusterer hyperparameters based on prediction
# See _serialize_examples_serving_v1 for hyper_list of version1 model2"""
_default_peritem_features_file = r'../data/JV_default_serving_peritem_features'
with open(_default_peritem_features_file, 'rb') as f:
    # Import from file
    HYPERPARAMETER_LIST = pickle.load(f)

#%% Module functions

class LoadSerializedAndServe:
    """Loads an existing serialized model and makes a prediction based on a 
    passed document from mongodb. The output is an array of ranked clustering
    hyperparameters that should yield the closest prediction on correct number
    of clusters
    
    inputs
    -------
    document: (dict) document from mongodb
    list_size: (int) number of examples to include in model for prediction. 
    This is based on the model's input layer when training. The correct
    values to pass are 180 for 'model2' and 200 for 'model4'
    model_name: (str) one of 'model2' or 'model4'. 'model2' is a DNN that
    was trained on only the clusterer algorithm and closen index. 'model4' is 
    a DNN that was trained on all clusterer hyperparameters [index, reduce, 
    clusterer, n_components, by_size]
    
    Example usage
    
    ServingClass = LoadSerializedAndServe()
    document = collection.find_one()
    
    predictions = ServingClass.load_serialized_and_serve_model2(document)
    """
    _list_size_model2 = 180
    _list_size_model4 = 200
    _directory_model2 = b'final_model\\Run_20191006011340model2\\1570661762'
    _directory_model4 = b'final_model\\Run_20191024002109model4\\1572051525'
    _graph_model2 = tf.Graph()
    _graph_model4 = tf.Graph()
    output_format = namedtuple('outputs', ['score','hyperparameter_dict'])

    @classmethod
    def load_serialized_and_serve_model2(cls, document, hyper_list_model2):
        
        input_eie = cls._encode_input_transmitter_fn_v1(
            document, 
            text_features=False,
            list_size=cls._list_size_model2)
        # Placeholder:0 is the feature column name in serving_input_receiver_fn
        input_feed_dict = {'Placeholder:0':[input_eie]}
        
        # Only 'serve' is a valid MetaGraphDef - dont use 'predict'
        with tf.Session(graph=cls._graph_model2) as sess:
            
            # MetaGraphDef - load MetaGraphDef to session graph
            cls._MetaGraphDef_model2 = tf.saved_model.load(sess, 
                                                ['serve'], 
                                                cls._directory_model2)
            
            outputs_model2 = sess.run('groupwise_dnn_v2/accumulate_scores/div_no_nan:0', 
                               feed_dict=input_feed_dict)
        
        # per-item features (clusterer) used for prediction
        clusterer_hyperparameters = hyper_list_model2[:cls._list_size_model2]
        
        output_tuple_list = []
        for score, hyperparameter in zip(outputs_model2.ravel(), clusterer_hyperparameters):
            output_tuple = cls.output_format(score, hyperparameter)
            output_tuple_list.append(output_tuple)
        
        return output_tuple_list
    
    @classmethod
    def load_serialized_and_serve_v1_2(cls,
                                       document, 
                                       list_size, 
                                       model_directory):
        """use a tensorflow-ranking estimator saved into a serialized form 
        to give a prediction on a model. (Version 2 using 
        tf.contrib.predictor.from_saved_model instead of tf.saved_model.load)
        
        inputs
        -------
        document: (dict) a document retrieved from mongo db in my own special 
        format
        text_features: (bool) will the model clust_index be passed as 
        text features and embedded (True) or one-hot encoded (False)
        list_size: (int) number of examples to prune or pad input EIE to
        model_directory: directory of serialized model. 
        
        Example Usage: 
        client = MongoClient('localhost', 27017)
        db = client['master_points']
        collection = db['raw_databases']
        
        # Retrieve document
        document = collection.find_one()
    
        model_directory = b'final_model\\Run_20191006011340model2\\1570661762'
        
        load_serialized_and_serve_2(document=document,
                                  text_features=False,
                                  list_size=180,
                                  model_directory=model_directory)
        """
    
        input_eie = cls._encode_input_transmitter_fn_v1(
            document, 
            text_features=False,
            list_size=list_size)
        input_feed_dict = {'inputs':[input_eie]}
        
        predict_fn = tf.contrib.predictor.from_saved_model(model_directory)
    
        outputs = predict_fn(input_feed_dict)
        
        return outputs
    
    @classmethod
    def load_serialized_and_serve_model4(cls, 
                                         document, 
                                         list_size, 
                                         peritem_source):
        """Create a serialized EIE from a mongo-db document. Feed the serialized
        TFProto to the serialized model and output the ranked prediction
        input
        -------
        document: (dict) document from mongodb
        list_size: (bool) list size of per-item features to be included. If 
        the default None is kept, 200 (default for model4) is used. If False 
        is passed then no padding or trimming of peritem lists will take place
        peritem_source: (str) 'default' or 'document'. The determines where
        peritem features come from. If 'default' a set list of clusterer
        hyperparameters is used. If 'document', the list of clusterer 
        hyperparameters associated with the document is used
        
        output
        -------
        output_tuple_list: (tuple) a named tuple of prediction scores and 
        hyperparameter dicts. Each score is presumable relevant to the 
        hyperparameter dict. Use output_tuple_list[0].score for the score and
        output_tuple_list[0].hyperparameter_dict for clustering hyperparams"""
        
        if list_size is None:
            list_size = cls._list_size_model4
            
        else:
            list_size = False
            
        if peritem_source == 'document':
            hyper_list = cls._get_hyper_list(document, 
                                              peritem_source=peritem_source)
        elif peritem_source == 'default':
            hyper_list = HYPERPARAMETER_LIST
        
        input_eie = cls._encode_input_transmitter_fn_v2(
            document,
            list_size=list_size,
            peritem_source=peritem_source)
        # Placeholder:0 is the feature column name in serving_input_receiver_fn
        input_feed_dict = {'Placeholder:0':[input_eie]}
        
        # Only 'serve' is a valid MetaGraphDef - dont use 'predict'
        with tf.Session(graph=cls._graph_model4) as sess:
            
            # MetaGraphDef - load MetaGraphDef to session graph
            cls._MetaGraphDef_model4 = tf.saved_model.load(sess, 
                                                    ['serve'], 
                                                    cls._directory_model4)
            
            outputs_model4 = sess.run('groupwise_dnn_v2/accumulate_scores/div_no_nan:0', 
                               feed_dict=input_feed_dict)
            
        # Default per-item features (clusterer) used for prediction
        clusterer_hyperparameters = hyper_list[:list_size]
        # TODO Change this to get hyper_list based on peritem_source
        
        output_tuple_list = []
        for score, hyperparameter in zip(outputs_model4.ravel(), 
                                         clusterer_hyperparameters):
            output_tuple = cls.output_format(score, hyperparameter)
            output_tuple_list.append(output_tuple)
            
        return output_tuple_list
     
    @classmethod
    def _serialize_examples_serving_v1(cls,
                                       document:dict,
                                       text_features:bool,
                                       list_size:int):
        """
        Create a list of serialized tf.train.Example proto from a document 
        stored in mongo-db. Use this for serving & predicting models (not
        used for training - the "relevance" label is not included)
        input
        -------
        document: (dict) must have ['encoded_hyper']['clust_index']['val']
        and ['hyper_labels'] fields
        text_features: (bool) Process clust_index as text (True) or load 
        from encoded (False)
        list_size: (int)
        output
        -------
        peritem_list: (list) a list of serialized per-item (exmpale) featuers
          in the form of tf.train.Example protos
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
            
            # Return for prediction
            hyper_list = pickle.loads(document['encoded_hyper']['clust_index']['cat'])
            hyper_list = np.concatenate((hyper_list[0], hyper_list[1]), axis=0)
            hyper_list_model2 = []
            for peritem_feature in peritem_features:
                clusterer, index = hyper_list[np.array(peritem_feature, dtype=np.bool_)]
                hyper_dict = {'clusterer':clusterer, 'index':index}
                hyper_list_model2.append(hyper_dict)
        
        def _pad(peritem_features):
            
            if text_features:
                # Pad with empty string
                cur_size = len(peritem_features)
                new_size = list_size
                example_shape = len(peritem_features[0])
                pad_value = [''] * example_shape
                for i in range(new_size-cur_size):
                    peritem_features.append(pad_value)
            
            else:
                # Pad with zeros array
                cur_size = len(peritem_features)
                new_size = list_size
                example_shape = len(peritem_features[0])
                pad_value = np.zeros((new_size-cur_size, example_shape))
                peritem_features = np.append(peritem_features, pad_value, axis=0)
                
            return peritem_features
        
        def _trim(peritem_features):
            # Return the first _list_size of peritem_features
            new_size = list_size
            trimmed_peritem_features = peritem_features[:new_size]
            return trimmed_peritem_features
        
        if list_size > len(peritem_features):
            peritem_features = _pad(peritem_features)
        elif list_size < len(peritem_features):
            peritem_features = _trim(peritem_features)
            
        # Create a list of serialized per-item features
        peritem_list = []
        
        for peritem_feature in peritem_features:
            peritem_dict = {}
            
            # Add peritem features to dictionary
            peritem_dict['encoded_clust_index'] = tf_feature_mapper(peritem_feature)
            
            # Create EIE and append to serialized peritem_list
            peritem_proto = tf.train.Example(
                    features=tf.train.Features(feature=peritem_dict))
            peritem_proto_str = peritem_proto.SerializeToString() # Serialized
        
            # List of serialized tf.train.Example
            peritem_list.append(peritem_proto_str)
            
        return peritem_list
    
    @classmethod
    def _encode_input_transmitter_fn_v1(cls, 
                                        document, 
                                        list_size,
                                        text_features=False):
        """Use this function to output a TFRecord from a mongo document.
        Useful for serving serialized tensorflow estimators/models
        inputs
        -------
        document: (dict) document retrieved from Mongo. Dictionary of field:value
        reciprocal: """
        
        # Same for serving and training
        context_proto_str = serialize_context(document)
        
        peritem_list = cls._serialize_examples_serving_v1(
            document, 
            text_features=text_features,
            list_size=list_size)
        
        # Prepare serialized feature spec for EIE format
        serialized_dict = {'serialized_context':_bytes_feature([context_proto_str]),
                           'serialized_examples':_bytes_feature(peritem_list)
                           }
        
        # Convert to tf.train.Example object
        serialized_proto = tf.train.Example(
                features=tf.train.Features(feature=serialized_dict))
        serialized_example_in_example = serialized_proto.SerializeToString()
        
        return serialized_example_in_example
    
    @classmethod
    def _get_hyper_list(cls,
                        document,
                        peritem_source):
        """Use this only when peritem_source is 'document'. Retrieves 
        peritem features from an existing document"""
        
        assert peritem_source == 'document', 'Dont use this function unless you\
        pass peritem_source=document'

        hyper_list = list(document['hyper_labels'].values())
            
        return hyper_list
    
    @classmethod
    def _serialize_examples_serving_v2(cls,
                                       document,
                                       list_size,
                                       peritem_source):
        """
        Create a list of serialized tf.train.Example proto from a document 
        stored in mongo-db. Use this for serving & predicting models (not
        used for training - the "relevance" label is not included). per-item
        example features are chosen as a fixed-set of clustering hyperparameters
        based on list_size (they are not based on how each document was actually
        clustered). 
        
        input
        -------
        document: (dict) must have ['encoded_hyper']['clust_index']['val']
          and ['hyper_labels'] fields
        list_size: (int) Number of examples to include. This must be determined
            from the input size of the ranking model. Optionally, pass False
            if you do not want peritem examples to be padded
        peritem_source: (str) 'default' or 'document'. The determines where
        peritem features come from. If 'default' a set list of clusterer
        hyperparameters is used. If 'document', the list of clusterer 
        hyperparameters associated with the document is used
        output
        -------
        peritem_list: (list) a list of serialized per-item (exmpale) featuers
          in the form of tf.train.Example protos
        """
           
        # Create dictionary of peritem features based on function input
        # Extract saved labels (text -> bytes)
        # ALl per-item features are saved as text and encoded or transformed later
        
        if peritem_source == 'default':
            hyper_list = HYPERPARAMETER_LIST
                
        elif peritem_source == 'document':
            hyper_list = list(document['hyper_labels'].values())
        
        peritem_features = [] # For iterating through and saving
        for subdict in hyper_list:
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
    
    @classmethod
    def _encode_input_transmitter_fn_v2(cls,
                                        document,
                                        list_size,
                                        peritem_source):
        """Use this function to output a TFRecord from a dictionary
        Useful for serving serialized tensorflow estimators/models
        inputs
        -------
        document: (dict) document retrieved from Mongo. Dictionary of field:value
        list_size: (int) number of examples to use in the model. Typ 160-200 for me
        """
        
        # Same for serving and training
        context_proto_str = serialize_context(document)
        
        peritem_list = cls._serialize_examples_serving_v2(
            document, 
            list_size=list_size,
            peritem_source=peritem_source)
        
        # Prepare serialized feature spec for EIE format
        serialized_dict = {'serialized_context':_bytes_feature([context_proto_str]),
                           'serialized_examples':_bytes_feature(peritem_list)
                           }
        
        # Convert to tf.train.Example object
        serialized_proto = tf.train.Example(
                features=tf.train.Features(feature=serialized_dict))
        serialized_example_in_example = serialized_proto.SerializeToString()
        
        return serialized_example_in_example


#%% This works
if __name__ == '__main__':
    
    _LIST_SIZE_V1 = 180
    _LIST_SIZE_V2 = 200
    model_directory_v1 = b'final_model\\Run_20191006011340model2\\1570661762'
    model_directory_v2 = b'final_model\\Run_20191024002109model4\\1572051525'
    
    document = collection.find_one()
    
    input_eie_v1 = LoadSerializedAndServe._encode_input_transmitter_fn_v1(
        document, 
        text_features=False,
        list_size=_LIST_SIZE_V1)
    input_eie_v2 = LoadSerializedAndServe._encode_input_transmitter_fn_v2(
        document,
        list_size=_LIST_SIZE_V2,
        example_features='all')
    
    # Placeholder:0 is the feature column name in serving_input_receiver_fn
    input_feed_dict_v1 = {'Placeholder:0':[input_eie_v1]}
    input_feed_dict_v2 = {'Placeholder:0':[input_eie_v2]}
    
    # Only 'serve' is a valid MetaGraphDef - dont use 'predict'
    graph_v1 = tf.Graph()
    with tf.Session(graph=graph_v1) as sess:
        model_import_v1 = tf.saved_model.load(
            sess,
            ['serve'],
            model_directory_v1)
        outputs_v1 = sess.run(
            'groupwise_dnn_v2/accumulate_scores/div_no_nan:0', 
            feed_dict=input_feed_dict_v1)
    
    graph_v2 = tf.Graph()
    with tf.Session(graph=graph_v2) as sess:
        model_import_v2 = tf.saved_model.load(
            sess, 
            ['serve'], 
            model_directory_v2)
        outputs_v2 = sess.run(
            'groupwise_dnn_v2/accumulate_scores/div_no_nan:0', 
            feed_dict=input_feed_dict_v2)
        
    print('ouputs_v1: ', outputs_v1, '\n\n')
    print('outputs_v2: ', outputs_v2)


#%% Depreciated, but WORKS
if __name__ == '__main__':
    
    model_directory = b'final_model\\Run_20191006011340model2\\1570661762'
    
    predict_fn = tf.contrib.predictor.from_saved_model(model_directory)
    
    input_eie = LoadSerializedAndServe._encode_input_transmitter_fn_v1(document, 
                                                text_features=False,
                                                list_size=180)
    
    # I need to name the input dictionary key correctly - use 'inputs' instead of 
    # EIE_input
    #prediction = predict_fn(
    #        {'EIE_input': [input_eie]})
    
    prediction = predict_fn(
            {'inputs': [input_eie]})


#%% Using saved_model_cli

import subprocess

def show_model_cli_tagset_signaturede():
    _directory = os.path.join(r'C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\final_model\Run_20191006011340model2', '1570661762/')
    _directory2 = os.path.join(r'C:\Users\z003vrzk\.spyder-py3\Scripts\ML\Point database categorizer\final_model\Run_20191024002109model4', '1572051525')
    
    ## This is how to show the cli
    #saved_model_cli show --dir 1570661762/ --tag_set serve \
    #--signature_def serving_default
    
    cmd = ['saved_model_cli', 'show', '--dir', _directory, '--tag_set', 'serve', 
           '--signature_def', 'serving_default']
    cmd2 = ['saved_model_cli', 'show', '--dir', _directory2, '--tag_set', 'serve', 
           '--signature_def', 'serving_default']
    
    with open(r'_stdout_data.txt', 'w') as f:
        subprocess.call(cmd, stdout=f) #.stdout
        
    
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, error = p.communicate() # out is bytes
    print(out.decode())
    
    return out.decode()

# Expected output form
"""
The given SavedModel SignatureDef contains the following input(s):
  inputs['inputs'] tensor_info:
      dtype: DT_STRING
      shape: (-1)
      name: Placeholder:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['outputs'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, -1)
      name: groupwise_dnn_v2/accumulate_scores/div_no_nan:0
Method name is: tensorflow/serving/regress
"""




    
#%% Depreciated

## Depreciated
#predict_fn = tf.contrib.predictor.from_saved_model(model_directory)
#
## Depreciated
#with tf.Session() as sess:
#    model_import = tf.saved_model.load(sess, ['serve'], model_directory)


#%% 

# Not depreciated - version 2 tensorflow
#model_directory_v2 = b'final_model\\Run_20191024002109model4\\1572051525'
#predict_fn = tf.saved_model.load(model_directory_v2)

#%% Doesnt work

#document = collection.find_one()
#
#model_directory = b'final_model\\Run_20191006011340model2\\1570661762'
#
#model_import = tf.saved_model.load_v2(model_directory)
#print(list(model_import.signatures.keys())) # ['serving_default']
#
#serving_fn = model_import.signatures['serving_default']
#predict_fn = model_import.signatures['predict']
#
#print(serving_fn.inputs) # Placeholder:0
#print(serving_fn.outputs) # 'groupwise_dnn_v2/accumulate_scores/div_no_nan:0'
#
#
#input_eie = _encode_input_transmitter_fn_v1(document, 
#                                        text_features=False,
#                                        list_size=180)
#
## Compatable w/ 1.14 and tf.saved_model.load
## Do not use the {'input_key':[serialized]} that is recommended other places
#prediction_serve = serving_fn(inputs=tf.constant([input_eie],
#                                                 dtype=tf.string))
#prediction_predict = predict_fn(tf.constant([input_eie], dtype=tf.string))
#
#with tf.Session() as sess: # Error
#    sess.run(prediction_serve)

#%% Depreciated, but WORKS

#predict_fn = tf.contrib.predictor.from_saved_model(model_directory)
#
## I need to name the input dictionary key correctly - use 'inputs' instead of 
## EIE_input
#prediction = predict_fn(
#        {'EIE_input': [input_eie]})
#
#prediction = predict_fn(
#        {'inputs': [input_eie]})


#%% Serve TFRecord for estimaton

"""
The goal of this function is to import a document from mongodb and
serve it for estimation with model_serving.

It is currently commented out becasue I think its the wrong form for serving
into the serialized model - see serving_input_receiver_fn() in ranking_model2
"""

"""
    # TESTING
    model_import_v1_test = tf.saved_model.load_v2(model_directory_v1, 
                                                   tags=['serve']) # Graph only
    dir(model_import_v1_test)
    dir(model_import_v1_test.initializer)
    
    # What do I do from here?
    dir(model_import_v1_test.signatures['serving_default'])
    
    model_import_v1_graph = model_import_v1_test.graph
    dir(model_import_v1_graph)
    
    # Try to import yourself
    from tensorflow.python.saved_model import loader_impl
    saved_model_proto = loader_impl.parse_saved_model(model_directory_v1)
    if (len(saved_model_proto.meta_graphs) == 1
        and saved_model_proto.meta_graphs[0].HasField("object_graph_def")):
        meta_graph_def = saved_model_proto.meta_graphs[0]
    
    my_meta_graph = saved_model_proto.meta_graphs[0]
    
    with tf.Session(graph=my_meta_graph) as sess:
#        sess.run(model_import_v1_test.initializer)
        outputs_v1 = sess.run('groupwise_dnn_v2/accumulate_scores/div_no_nan:0', 
                           feed_dict=input_feed_dict_v1)
"""






