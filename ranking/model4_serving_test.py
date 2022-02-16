# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 18:03:45 2022

@author: jvorsten
"""

# Python imports
import configparser
import pickle
import os, sys
import base64
from unittest import TestCase
import copy

# Third party imports
import sqlalchemy
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
import numpy as np
import requests
import pandas as pd

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
from extract.SQLAlchemyDataDefinition import (Points,
                                              Customers)
from transform_ranking import Transform
from rank_write_record import (_bytes_feature)
from model4_serving import (serialize_context_from_dictionary, 
                            serialize_examples_model4, 
                            serialize_example_in_example, 
                            DatabaseFeatures,
                            RANKING_MODEL_URL, 
                            RawInputData)

# Declarations
"""Peritem clusterer hyperparameters used for prediction
# Return these with the output prediction. User will be able 
# To rank clusterer hyperparameters based on prediction"""
_default_peritem_features_file = './default_serving_peritem_features.dat'
HYPERPARAMETER_SERVING_FILEPATH = '../data/serving_hyperparameters.dat'
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

# Store a static simple group of points representation
STATIC_DICT = {
    'id': {567:41996,569:41998,574:42003,576:42005,  642:42071,645:42074,  646:42075,647:42076,  648:42077,649:42078},
    'POINTID': {567:1068.0,  569:1069.0,  574:1073.0,576:1074.0,  642:1140.0,  645:1143.0,  646:1144.0,  647:1145.0,  648:1146.0,  649:1147.0},
    'DEVICEHI': {567:np.nan,  569:np.nan,  574:np.nan,  576:np.nan,  642: -12.5,  645:np.nan,  646:np.nan,  647:np.nan,  648:np.nan,  649:0.0},
    'DEVICELO': {567:np.nan,  569:np.nan,  574:np.nan,  576:np.nan,  642: -12.5,  645:np.nan,  646:np.nan,  647:np.nan,  648:np.nan,  649:0.0}, 
    'SIGNALHI': {567:np.nan,  569:np.nan,  574:np.nan,  576:np.nan,  642:0.0,  645:np.nan,  646:np.nan,  647:np.nan,  648:np.nan,  649:0.0},
    'SIGNALLO': {567:np.nan,  569:np.nan,  574:np.nan,  576:np.nan,  642:0.0,  645:np.nan,  646:np.nan,  647:np.nan,  648:np.nan,  649:0.0},
    'NUMBERWIRE': {567:np.nan,  569:np.nan,  574:np.nan,  576:np.nan,  642:0.0,  645:np.nan,  646:np.nan,  647:np.nan,  648:np.nan,  649:0.0},
    'WIRELENGTH': {567:np.nan,  569:np.nan,  574:np.nan,  576:np.nan,  642:0.0,  645:np.nan,  646:np.nan,  647:np.nan,  648:np.nan,  649:0.0},
    'WIRESIZE': {567:np.nan,  569:np.nan,  574:np.nan,  576:np.nan,  642:0.0,  645:np.nan,  646:np.nan,  647:np.nan,  648:np.nan,  649:0.0}, 
    'PROOFDELAY': {567:30.0,  569:30.0,  574:30.0,  576:30.0,  642:0.0,  645:0.0,  646:0.0,  647:0.0,  648:0.0,  649:0.0},
    'INVERTED': {567:False,  569:False,  574:False,  576:False,  642:False,  645:False,  646:False,  647:False,  648:False,  649:False}, 
    'PROOFPRSNT': {567:True,  569:True,  574:True,  576:True,  642:True,  645:True,  646:True,  647:True,  648:True,  649:True},
    'VIRTUAL': {567:False,  569:False,  574:False,  576:False,  642:False,  645:False,  646:False,  647:False,  648:False,  649:False}, 
    'TMEMBER': {567:False,  569:False,  574:False,  576:False,  642:False,  645:False,  646:False,  647:False,  648:False,  649:False}, 
    'ADDRESSEXT': {567:None,  569:None,  574:None,  576:None,  642:None,  645:None,  646:None,  647:None,  648:None,  649:None}, 
    'ALARMHIGH': {567:'0.0000',  569:'0.0000',  574:'0.0000',  576:'0.0000',  642:'0.0000',  645:'0.0000',  646:'0.0000',  647:'0.0000',  648:'0.0000',  649:'0.0000'},
    'ALARMLOW': {567:'0.0000',  569:'0.0000',  574:'0.0000',  576:'0.0000',  642:'0.0000',  645:'0.0000',  646:'0.0000',  647:'0.0000', 648:'0.0000',  649:'0.0000'}, 
    'ALARMTYPE': {567:'None',  569:'None',  574:'None',  576:'None',  642:'None',  645:'None',  646:'None',  647:'Standard',  648:'None',  649:'None'},
    'COMBOID': {567:None,  569:None,  574:None,  576:None,  642:None,  645:None,  646:None,  647:None,  648:None,  649:None}, 
    'CONTRLTYPE': {567:None,  569:None,  574:None,  576:None,  642:'PXCC',  645:None,  646:None,  647:None,  648:None,  649:'PXCM'},
    'CS': {567:None,  569:None,  574:None,  576:None,  642:None,  645:'RE',  646:None,  647:None,  648:None,  649:'TTE'},
    'CTSENSTYPE': {567:None,  569:None,  574:None,  576:None,  642:'1',  645:None,  646:None,  647:None,  648:None,  649:'8'},
    'CTSYSNAME': {567:'SOFT.HWP02.SS',  569:'SOFT.HWP01.SS',  574:'SOFT.CTF01.SS',  576:'SOFT.CTF02.SS',  642:'PRESSER.PCHW.DP',  645:'PLANT.CHL1.SS',  646:'PLANT.PCHWP2.STA',  647:'PLANT.BLR01.AL',  648:'PLANT.BLR01.PRF',  649:'PLANT.BLR01.RT'}, 
    'DESCRIPTOR': {567:'PUMP S/S',  569:'PUMP S/S', 574:'CT FAN S/S',  576:'CT FAN S/S',  642:'PLANT CHW DP',  645:'CHILLER S/S',  646:'PCHWP-2 STATUS',  647:'BOILER ALARM',  648:'BOILER STATUS',  649:'BOILER RET TEMP'},
    'DEVNUMBER': {567:None,  569:None,  574:None,  576:None,  642:None,  645:'     3',  646:None,  647:None,  648:None,  649:'     3'},
    'DEVUNITS': {567:None,  569:None,  574:None,  576:None,  642:'psi',  645:None,  646:None,  647:None,  648:None,  649:'DEG F'},
    'DROP': {567:'11',  569:'11',  574:'10',  576:'10',  642:'0',  645:'7',  646:'11',  647:'11',  648:'11',  649:'4'},
    'FUNCTION': {567:'Proof',  569:'Proof',  574:'Proof',  576:'Proof',  642:'Value',  645:'On/Off',  646:'Status',  647:'Status',  648:'Status',  649:'Value'},
    'INITVALUE': {567:'OFF',  569:'OFF',  574:'OFF',  576:'OFF',  642:'0.000000',  645:'OFF',  646:'0.000000',  647:'0.000000',  648:'0.000000',  649:'0.000000'},
    'INTERCEPT': {567:None,  569:None,  574:None,  576:None,  642:'-12.5000000000',  645:None,  646:None,  647:None,  648:None,  649:'0.0000000000'}, 
    'LAN': {567:'0',  569:'0',  574:'0',  576:'0',  642:'0',  645:'0',  646:'0',  647:'0',  648:'0',  649:'0'}, 
    'NAME': {567:'SOFT.HWP02.SS',  569:'SOFT.HWP01.SS',  574:'SOFT.CTF01.SS',  576:'SOFT.CTF02.SS',  642:'PRESSER.PCHW.DP',  645:'PLANT.CHL1.SS',  646:'PLANT.PCHWP2.STA',  647:'PLANT.BLR01.AL',  648:'PLANT.BLR01.PRF',  649:'PLANT.BLR01.RT'},
    'NETDEVID': {567:'SOFTCONTROLLER',  569:'SOFTCONTROLLER',  574:'SOFTCONTROLLER',  576:'SOFTCONTROLLER',  642:'PRESSER.PXCC01',  645:'PLANT.CHWSYS.PXCM01',  646:'PLANT.CHWSYS.PXCM01',  647:'PLANT.CHWSYS.PXCM01',  648:'PLANT.CHWSYS.PXCM01',  649:'PLANT.CHWSYS.PXCM01'}, 
    'POINT': {567:'2',  569:'1',  574:'13',  576:'14',  642:'1',  645:'2',646:'13',  647:'3',  648:'4',  649:'6'}, 
    'POINTACTUAL': {567:None,  569:None,  574:None,  576:None,  642:None,  645:None,  646:None,  647:None,  648:None,  649:None}, 
    'POWER': {567:None,  569:None,  574:None,  576:None,  642:None,  645:None,  646:None,  647:None,  648:None,  649:None},
    'S1000TYPE': {567:None,  569:None,  574:None,  576:None,  642:'?',  645:None,  646:None,  647:None,  648:None,  649:'?'}, 
    'SENSORTYPE': {567:None,  569:None,  574:None,  576:None,  642:'CURRENT',  645:None,  646:None,  647:None,  648:None,  649:'RTDNICKEL'}, 
    'SIGUNITS': {567:None,  569:None,  574:None,  576:None,  642:'mA',  645:None,  646:None,  647:None,  648:None,  649:'ohm'}, 
    'SLOPE': {567:None,  569:None,  574:None,  576:None,  642:'0.0020350000',  645:None,  646:None,  647:None,  648:None,  649:'1.0000000000'},
    'SYSTEM': {567:None,  569:None,  574:None,  576:None,  642:None,  645:'CHILLERS',  646:'SCHW',  647:'HWS-MAYBORN',  648:'HWS-MAYBORN',  649:'HWS-MAYBORN'}, 
    'TYPE': {567:'L2SL',  569:'L2SL',  574:'L2SL',  576:'L2SL',  642:'LAI',  645:'LDO',  646:'LDI',  647:'LDI',  648:'LDI',  649:'LAI'}, 
    'UNITSTYPE': {567:None,  569:None,  574:None,  576:None,  642:'US',  645:None,  646:None,  647:None,  648:None,  649:'US'}, 
    'customer_id': {567:15,  569:15,  574:15,  576:15,  642:15,  645:15,  646:15,  647:15,  648:15,  649:15},
    'group_id': {567:955.0,  569:954.0,  574:953.0,  576:952.0,  642:962.0,  645:967.0,  646:909.0,  647:914.0,  648:914.0,  649:970.0}
    }

STATIC_DATAFRAME = pd.DataFrame(STATIC_DICT)
DATABASE_FEATURES = {
    'instance': 'D:\\Z - Saved SQL Databases\\44OP-117216_UMHB_Stadium\\JobDB.mdf',
     'n_instance': 445,
     'n_features': 161,
     'len_var': 0.18328998863779825,
     'uniq_ratio': 2.7639751552795033,
     'n_len3': 0.7797752808988764,
     'n_len4': 0.2157303370786517,
     'n_len2': 0.0022471910112359553,
     'n_len1': 0.0022471910112359553,
     'n_len5': 0,
     'n_len6': 0,
     'n_len7': 0}



#%%

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

def get_feature_specification():
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
    
    return context_feature_spec, example_feature_spec

#%%

class Model4ServingTest(TestCase):
    
    def setUp(self):
        
        # Static database features
        self.database_features = copy.deepcopy(DATABASE_FEATURES)
        
        # Serialized context
        self.serialized_context = serialize_context_from_dictionary(self.database_features)
        
        # Serialized example in example
        self.serialized_peritem = serialize_examples_model4(
            HYPERPARAMETER_LIST,
            list_size=_LIST_SIZE_MODEL4)
        
        return None
    
    def test_serialize_context(self):
        
        #1. Context features (bytes object)
        serialized_context = serialize_context_from_dictionary(self.database_features)
        self.assertIsInstance(serialized_context, bytes)
        
        return None
    
    def test_serialize_example_in_example(self):
        
        #1. Context features (bytes object)
        serialized_context = serialize_context_from_dictionary(self.database_features)

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
        
        serialized_example_in_example2 = serialize_example_in_example(
            self.database_features,
            HYPERPARAMETER_LIST,
            _LIST_SIZE_MODEL4)
        self.assertEqual(serialized_example_in_example, serialized_example_in_example2)
        
        return None
    
    def test_parse_serialized_example_in_example(self):
        
        context_feature_spec, example_feature_spec = get_feature_specification()
        
        # Convert serialized example-in-example to tensor
        context_features = tf.io.parse_example(
            serialized=[self.serialized_context],
            features=context_feature_spec)
        peritem_features = tf.io.parse_example(
            serialized=self.serialized_peritem,
            features=example_feature_spec)
        
        return None
    
    def test_request_tensorflow_serving(self):
        """"""
        
        # Base 64 encode a byte string
        serialized_example_in_example_b64 = base64.b64encode(
            self.serialized_example_in_example)

        serialized_example_in_example_utf8 = base64.b64encode(
            self.serialized_example_in_example).decode("utf-8")

        model_server_url = "http://localhost:8501/v1/models/model4:predict"

        json_data = {
          "signature_name":"serving_default",
          "instances":[
              {"b64":base64.b64encode(self.serialized_example_in_example)\
                           .decode('utf-8')}
                  ]
              }
        # Do not actually send request - it might not be running
        # This is for record purposes only
        # resp = requests.post(model_server_url, json=json_data)
        # print(resp.json())
        
        return None
    

class Model4ServingOnlineTest():
    
    def setUp(self):
        
        # Static database features
        self.database_features = copy.deepcopy(DATABASE_FEATURES)
        
        # Serialized context
        self.serialized_context = serialize_context_from_dictionary(self.database_features)
        
        # Serialized example in example
        self.serialized_peritem = serialize_examples_model4(
            HYPERPARAMETER_LIST,
            list_size=_LIST_SIZE_MODEL4)
        
        return None
    
    def test_request_model4_serving(self):
        
        SERVING_MODEL_URL = 'http://localhost:8004/clustering-ranking/model4predict/'
        headers = {'accept': 'application/json',
                   'Content-Type': 'application/json'}
        json_data = RawInputData(**DATABASE_FEATURES).json()
        
        resp = requests.post(SERVING_MODEL_URL, json=json_data, headers=headers)
        print(resp.json())

        self.assertEqual(resp.status_code, 200)
            
        return None


class LegacyRecords:
    
    def test_serialize_example_in_example(self):
        
        # Requires MSSQL server, data in 'Clustering' database,
        # configuration file, and other stuff
        config = configparser.ConfigParser()
        config.read(r'../extract/sql_config.ini')
        server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
        driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
        database_name = config['sql_server']['DEFAULT_DATABASE_NAME']

        Insert = extract.Insert(server_name=server_name,
                                driver_name=driver_name,
                                database_name=database_name)
        
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
        
        return serialized_example_in_example
    
    def test_tf1_14_load_model_and_serve(self):
                            
        # Worked using tensorflow 1.14.0
        # Placeholder:0 is the feature column name in serving_input_receiver_fn
        input_feed_dict = {'Placeholder:0':[serialized_example_in_example]}
        input_feed_dict2 = {'EIE_input':[serialized_example_in_example]}
        
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
            
            # outputs2 = sess.run(
            #     'predict',
            #     feed_dict={"EIE_input":serialized_example_in_example})
            
        print('outputs_model4: ', outputs_model4)
        hyperparameter_list = HYPERPARAMETER_LIST[:_LIST_SIZE_MODEL4]
        _best_idx = np.argsort(outputs_model4, axis=1)
        print("Best 5 hyperparameter sets for clustering: \n")
        for i in range(1, 5):
            print("Score: ", outputs_model4[0, _best_idx[0, -i]])
            print(hyperparameter_list[_best_idx[0, -i]])
                
        return None
    
    def model_exploration(self):

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
        
        context_feature_spec, example_feature_spec = get_feature_specification()
        
        # Load previously serialized model from V1 tensorflow
        # When you 'prune' a graph, you extract functions for a new subgraph. This is 
        # Equivalent to naming feeds and fetches within a session
        _feed = 'Placeholder:0'
        _fetch = 'groupwise_dnn_v2/accumulate_scores/div_no_nan:0'
        _input_signature = example_feature_spec
        imported = tf.compat.v2.saved_model.load(_MODEL4_DIRECTORY)
        pruned = imported.prune(_feed, _fetch, input_signature=_input_signature)
        pruned.inputs
        pruned.name
        pruned.output_dtypes
        pruned.output_shapes
        pruned.outputs
        pruned.structured_input_signature
        pruned.structured_outputs
        pruned._captured_inputs
        
        pruned(serialized_example_in_example)
        pruned(tf.Tensor(serialized_example_in_example, 1, dtype=tf.string))
        pruned(context_features)
        pruned(tf.constant([[1.]]))
        pruned(tf.ones(5))
        pruned(serialized_context)
        pruned(input_feed_dict)
        
        
        pruned2 = imported.prune('predict', 'groupwise_dnn_v2/accumulate_scores/div_no_nan:0')
        f_serving_default = imported.signatures['serving_default']
        f_serving_default.inputs
        f_serving_default.captured_inputs
        f_serving_default.output_dtypes
        f_serving_default.output_shapes
        f_serving_default.outputs
        f_serving_default(serialized_example_in_example)
        f_serving_default(context_features)
        f_serving_default(tf.ones(5))
        
        f_predict= imported.signatures['predict']
        f_predict.inputs
        f_predict.captured_inputs
        f_predict.output_dtypes
        f_predict.output_shapes
        f_predict.outputs
        f_predict(serialized_example_in_example)
        
        return None