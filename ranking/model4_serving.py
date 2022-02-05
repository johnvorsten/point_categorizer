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
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
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
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev,
                                              Customers, 
                                              ClusteringHyperparameter, 
                                              Labeling)
from transform_ranking import Transform
from rank_write_record import (
    tf_feature_mapper, 
    _bytes_feature)

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
HYPERPARAMETER_SERVING_FILEPATH = r'../data/serving_hyperparameters.dat'
with open(_default_peritem_features_file, 'rb') as f:
    # Import from file
    HYPERPARAMETER_LIST = pickle.load(f)
    
_LIST_SIZE_MODEL4 = 200
_MODEL4_DIRECTORY = 'final_model\\Run_20191024002109model4\\1572051525'
_file_name_bysize = '../data/vocab_bysize.txt'
_file_name_clusterer = '../data/vocab_clusterer.txt'
_file_name_index = '../data/vocab_index.txt'
_file_name_n_components = '../data/vocab_n_components.txt'
_file_name_reduce = '../data/vocab_reduce.txt'
    
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
    
class PipelineError(Exception):
    pass
    
def get_database_features(database, pipeline, instance_name) -> DatabaseFeatures:
    """Calculate the features of a dataset. These will be used to
    predict a good clustering algorithm.
    Inputs
    -------
    database: Your database. Its type not matter as long as the
        pipeline passed outputs a numpy array.
    pipeline: (sklearn.pipeline.Pipeline) Your finalized pipeline.
        See sklearn.Pipeline. The output of the pipelines .fit_transform()
        method should be your encoded array
        of instances and features of size (n,p) n=#instances, p=#features.
        Alternatively, you may pass a sequence of tuples containing names
        and pipeline objects: [('pipe1', myPipe1()), ('pipe2',myPipe2())]
    instance_name: (str | int) unique tag/name to identify each instance.
        The instance with feature vectors will be returned as a pandas
        dataframe with the tag on the index.
    Returns
    -------
    df: a pandas dataframe with the following features
        a. Number of points
		b. Correct number of clusters
		c. Typical cluster size (n_instances / n_clusters)
		d. Word length
		e. Variance of words lengths?
		f. Total number of unique words
        g. n_instances / Total number of unique words

    # Example Usage
    from JVWork_UnClusterAccuracy import AccuracyTest
    myTest = AccuracyTest()
    text_pipe = myDBPipe.text_pipeline(vocab_size='all', attributes='NAME',
                               seperator='.')
    clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False,
                                  replace_numbers=False,
                                  remove_virtual=True)
    mypipe = Pipeline([('clean_pipe', clean_pipe),
                       ('text_pipe',text_pipe)
                       ])
    correct_k = myTest.get_correct_k(db_name, df_clean, manual=True)

    data_transform = DataFeatures()
    db_feat = data_transform.calc_features(database, mypipe,
                                           tag=db_name, correct_k=correct_k)
    """

    if hasattr(pipeline, '__iter__'):
        pipeline = Pipeline(pipeline)
    else:
        pass

    try:
        data = pipeline.fit_transform(database)
    except:
        msg = "An error occured in the passed pipeline {}".format(pipeline.named_steps)
        raise(PipelineError(msg))

    if isinstance(data, csr_matrix):
        data = data.toarray()

    # Number of points
    n_points = data.shape[0]

    # Number of features
    n_features = data.shape[1]

    # Word lengths (as percentage of total number of instances)
    count_dict_pct = get_word_dictionary(data, percent=True)

    # Variance of lengths
    lengths = []
    count_dict_whole = get_word_dictionary(data, percent=False)
    for key, value in count_dict_whole.items():
        lengths.extend([int(key[-1])] * value)
    len_var = np.array(lengths).var(axis=0)

    features_dict = {
            'instance':instance_name,
            'n_instance':n_points,
            'n_features':n_features,
            'len_var':len_var,
            'uniq_ratio':n_points/n_features,
            }
    features_dict: DatabaseFeatures = {**features_dict, **count_dict_pct}
    # features_df = pd.DataFrame(features_dict, index=[instance_name])

    return features_dict

def get_word_dictionary(word_array, percent=True):

    count_dict = {}

    #Create a dictionary of word lengths
    for row in word_array:
        count = sum(row>0)
        try:
            count_dict[count] += 1
        except KeyError:
            count_dict[count] = 1

    #Return word lengths by percentage, or
    if percent:
        for key, label in count_dict.items():
            count_dict[key] = count_dict[key] / len(word_array) #Percentage
    else:
        pass #Dont change it, return number of each length

    max_key = 7 #Static for later supervised training
    old_keys = list(count_dict.keys())
    new_keys = ['n_len' + str(old_key) for old_key in old_keys]

    for old_key, new_key in zip(old_keys, new_keys):
        count_dict[new_key] = count_dict.pop(old_key)

    required_keys = ['n_len' + str(key) for key in range(1,max_key+1)]
    for key in required_keys:
        count_dict.setdefault(key, 0)

    return count_dict

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

def serving_input_receiver_fn():

    serialized_tfrecord = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='EIE_input')  # placeholder
    receiver_tensors = {'EIE_input':serialized_tfrecord}

    # Building the input reciever
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

    # Adding an outer layer (None) to shape would allow me to batch inputs
    # for efficiency. However that woud mean I have to add an outer list layer
    # aka [[]] to all my inputs, and combine features
    # It may be easier to pass a single batch at a time
    by_size = tf.feature_column.categorical_column_with_vocabulary_file(
        'by_size',
        _file_name_bysize,
        dtype=tf.string)
    clusterer = tf.feature_column.categorical_column_with_vocabulary_file(
        'clusterer',
        _file_name_clusterer,
        dtype=tf.string)
    index = tf.feature_column.categorical_column_with_vocabulary_file(
        'index',
        _file_name_index,
        dtype=tf.string)
    n_components = tf.feature_column.categorical_column_with_vocabulary_file(
        'n_components',
        _file_name_n_components,
        dtype=tf.string)
    reduce = tf.feature_column.categorical_column_with_vocabulary_file(
        'reduce',
        _file_name_reduce,
        dtype=tf.string)

    by_size_indicator = tf.feature_column.indicator_column(by_size)
    clusterer_indicator = tf.feature_column.indicator_column(clusterer)
    index_indicator = tf.feature_column.indicator_column(index)
    n_components_indicator = tf.feature_column.indicator_column(n_components)
    reduce_indicator = tf.feature_column.indicator_column(reduce)

    context_feature_spec = tf.feature_column.make_parse_example_spec(
            [n_instance,
            n_features,
            len_var,
            uniq_ratio,
            n_len1,
            n_len2,
            n_len3,
            n_len4,
            n_len5,
            n_len6,
            n_len7])

    example_feature_spec = tf.feature_column.make_parse_example_spec(
          [by_size_indicator,
           clusterer_indicator,
           index_indicator,
           n_components_indicator,
           reduce_indicator])

    # Parse receiver_tensors
    parsed_features = tfr.python.data.parse_from_example_in_example(
          serialized_tfrecord,
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec)

#    # Transform receiver_tensors - sparse must be transformed to dense
#    context_features, example_features = tfr.feature.encode_listwise_features(
#        features=parsed_features,
#        input_size=_LIST_SIZE,
#        context_feature_columns=context_feature_columns(),
#        example_feature_columns=example_feature_columns_v2(),
#        mode=tf.estimator.ModeKeys.PREDICT,
#        scope="transform_layer")

#    features = {**context_features, **example_features}

#    context_input = [
#          tf.compat.v1.layers.flatten(context_features[name])
#          for name in sorted(context_feature_columns())
#      ]
#    group_input = [
#          tf.compat.v1.layers.flatten(example_features[name])
#          for name in sorted(example_feature_columns_v2())
#      ]
#    input_layer = tf.concat(context_input + group_input, 1)

    return tf.estimator.export.ServingInputReceiver(parsed_features, receiver_tensors)

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
database_features = get_database_features(
    database,
    full_pipeline,
    instance_name=customer_name)
database_features.pop('instance')

#1. Context features (bytes object)
serialized_context = serialize_context_from_dictionary(database_features)

#2. Peritem features (bytes object)
serialized_peritem = serialize_examples_model4(
    HYPERPARAMETER_LIST,
    list_size=_LIST_SIZE_MODEL4)

# Prepare serialized feature spec for EIE format
serialized_dict = {'serialized_context':_bytes_feature([serialized_context]),
                   'serialized_examples':_bytes_feature(serialized_peritem)
                   }

# Convert to tf.train.Example object
serialized_proto = tf.train.Example(
        features=tf.train.Features(feature=serialized_dict))
serialized_example_in_example = serialized_proto.SerializeToString()

# Convert serialized example-in-example to tensor
context_features = tf.io.parse_example(
    serialized=[serialized_context],
    features=context_feature_spec)
peritem_features = tf.io.parse_example(
    serialized=serialized_peritem,
    features=example_feature_spec)

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
imported = tf.compat.v2.saved_model.load(_MODEL4_DIRECTORY)
pruned = imported.prune('Placeholder:0', 'groupwise_dnn_v2/accumulate_scores/div_no_nan:0')
pruned(serialized_example_in_example)
pruned(input_feed_dict)



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
        
    print('outputs_model4: ', outputs_model4)
    hyperparameter_list = HYPERPARAMETER_LIST[:_LIST_SIZE_MODEL4]
    _best_idx = np.argsort(outputs_model4, axis=1)
    print("Best 5 hyperparameter sets for clustering: \n")
    for i in range(1, 5):
        print("Score: ", outputs_model4[0, _best_idx[0, -i]])
        print(hyperparameter_list[_best_idx[0, -i]])
