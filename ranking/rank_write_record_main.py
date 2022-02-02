# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:46:14 2020

@author: z003vrzk
"""

# Python imports
import sys
import os
import configparser

# Thrid party imports
from pymongo import MongoClient
import sqlalchemy
from sklearn.pipeline import Pipeline
import tensorflow as tf
import tensorflow_ranking as tfr
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


from ranking.rank_write_record import (serialize_examples_v2,
                                       serialize_context,
                                       serialize_context_from_dictionary,
                                       serialize_examples_from_dictionary,
                                       get_train_test_id_mongo,
                                       get_train_test_id_sql,
                                       _bytes_feature)
from ranking import Labeling
from transform import transform_pipeline
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter)
from extract.SQLAlchemyDataDefinition import Labeling as SQLTableLabeling
from clustering.accuracy_visualize import Record, get_records

# Declarations
ExtractLabels = Labeling.ExtractLabels()
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

#%%

def save_tfrecord_mongo():
    """Save TFRecord EIE format to files for ranking
    See rank_write_record.py for how Mongo database documents are
    converted to tf.train.Example objects
    Here the tf.train.Example objects are nested into the Example-in-example format
    recommended by tensorflow ranking library

    EIE Examples are of the form
     {'serialized_context':tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])),
      'serialized_examples': tf.train.Feature(bytes_list=tf.train.BytesList(value=value))}
     for 'serialized_context' value is a serialized tf.train.Example
     for 'serialized_examples' value is a list of serialized tf.train.Example
     objects that will be ranked according to their relevance to the context
     features"""

    # Set up connection to MongoDB
    client = MongoClient('localhost', 27017)
    db = client['master_points']
    collection = db['raw_databases']

    # location to save TFRecord files
    test_file = r'../data/JV_test_text_binned.tfrecords' # Testing
    train_file = r'../data\JV_train_text_binned.tfrecords' # Training

    # Get document IDs of training and testing documents
    _train_pct = 0.8
    if 'train_ids' not in locals():
        train_ids, test_ids = get_train_test_id_mongo(collection,
                                                  train_pct=_train_pct)

    """Set up how I want to assign labels to objects
    Reciprocal will cause labels to be the inverse of the loss metric
    Set to True if I do not want labels to be binned"""
    reciprocal = False # Reciprocal of relevance label - use if you dont bin labels
    n_bins = 5 # number of bins for relevance label

    if os.path.isfile(train_file):
        _confirm = input(f'{train_file} already exists. Overwrite?\n>>>')
        if _confirm not in ['Y','y','Yes','yes','True','true']:
            raise SystemExit('Script execution stopped to not overwrite file')

    train_writer = tf.io.TFRecordWriter(train_file)

    for document in collection.find({'_id':{'$in':list(train_ids)}}):

        # Serialize context featuers -> serialized_context
        # This is a serialized tf.train.Example object
        context_proto_str = serialize_context(document)

        # Serialize peritem features. AKA examples or instances that will be ranked
        # This is a list of serialized tf.train.Example objects
        peritem_list = serialize_examples_v2(document,
                                             reciprocal=reciprocal,
                                             n_bins=n_bins,
                                             shuffle_peritem=True)

        # Prepare serialized feature spec for EIE format
        serialized_dict = {'serialized_context':_bytes_feature([context_proto_str]),
                           'serialized_examples':_bytes_feature(peritem_list)
                           }

        # Convert dictionary to tf.train.Example object
        serialized_proto = tf.train.Example(
                features=tf.train.Features(feature=serialized_dict))
        serialized_str = serialized_proto.SerializeToString()

        train_writer.write(serialized_str)

    train_writer.close()



    if os.path.isfile(test_file):
        _confirm = input(f'{test_file} already exists. Overwrite?\n>>>')
        if _confirm not in ['Y','y','Yes','yes','True','true']:
            raise SystemExit('Script execution stopped to not overwrite file')

    test_writer = tf.io.TFRecordWriter(test_file)

    for document in collection.find({'_id':{'$in':list(test_ids)}}):

        # Serialize context featuers -> serialized_context
        # This is a serialized tf.train.Example object
        context_proto_str = serialize_context(document)

        # Serialize peritem features. AKA examples or instances that will be ranked
        # This is a list of serialized tf.train.Example objects
        peritem_list = serialize_examples_v2(document,
                                             reciprocal=reciprocal,
                                             n_bins=n_bins,
                                             shuffle_peritem=True)

        # Prepare serialized feature spec for EIE format
        serialized_dict = {'serialized_context':_bytes_feature([context_proto_str]),
                           'serialized_examples':_bytes_feature(peritem_list)
                           }

        # Convert dictionary to tf.train.Example object
        serialized_proto = tf.train.Example(
                features=tf.train.Features(feature=serialized_dict))
        serialized_str = serialized_proto.SerializeToString()

        test_writer.write(serialized_str)

    test_writer.close()

    return None

#%% Savee TFRecords from SQL

def save_tfrecord_sql(customer_ids,
                      peritem_keys,
                      label_key,
                      reciprocal,
                      n_bins,
                      tfrecord_writer):
    """Save TFRecord EIE format to files for ranking
    See rank_write_record.py for how Mongo database documents are
    converted to tf.train.Example objects
    Here the tf.train.Example objects are nested into the Example-in-example format
    recommended by tensorflow ranking library

    EIE Examples are of the form
     {'serialized_context':tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])),
      'serialized_examples': tf.train.Feature(bytes_list=tf.train.BytesList(value=value))}
     for 'serialized_context' value is a serialized tf.train.Example
     for 'serialized_examples' value is a list of serialized tf.train.Example
     objects that will be ranked according to their relevance to the context
     features

     Inputs
     -------
     customer_ids : (list) of customer_ids in SQL database to save
     peritem_keys : (list) of string keys that exist in peritem_features.
         Should be ['by_size','n_components','clusterer','reduce','index']
    reciprocal : (bool) Set up how I want to assign labels to objects
        Reciprocal will cause labels to be the inverse of the loss metric
        Set to True if I do not want labels to be binned
    n_bins : (int) number of bins for relevance label if reciprocal is False
    tfrecord_writer : (tf.io.TFRecordWriter) To serialized EIE TFRecord
     """

    """Create a pipeline for transforming points databases"""

    assert hasattr(customer_ids, '__iter__'), "customer_ids must be iterable"
    msg = "Each ID in customer_ids must be int type, not {}"
    for _id in customer_ids:
        assert isinstance(_id, int), msg.format(type(_id))


    Transform = transform_pipeline.Transform()
    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(remove_dupe=False,
                                          replace_numbers=False,
                                          remove_virtual=True)

    # Create pipeline specifically for clustering text features
    text_pipe = Transform.text_pipeline(vocab_size='all',
                                       attributes='NAME',
                                       seperator='.',
                                       heirarchial_weight_word_pattern=True)

    full_pipeline = Pipeline([('clean_pipe', clean_pipe),
                              ('text_pipe', text_pipe),
                              ])

    for customer_id in customer_ids:
        print("Saving TFRecord for Customer ID : {}".format(customer_id))

        """Serialize context featuers -> serialized_context
        This is a serialized tf.train.Example object"""
        # Get Points databases related to customer_id
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
        database = Insert.pandas_select_execute(sel)
        sel = sqlalchemy.select([Customers.name]).where(Customers.id.__eq__(customer_id))
        customer_name = Insert.core_select_execute(sel)[0].name
        if database.shape[0] == 0:
            print(database.shape)
            print(customer_name)
            # Null databases should be skipped
            continue
        # Extract database featuers from Points database
        try:
            database_features = ExtractLabels.get_database_features(database,
                                                                    full_pipeline,
                                                                    instance_name=customer_name)
        except Labeling.PipelineError:
            print("An error occured while getting database features")
            print("Customer name : {}".format(customer_name))
            print("Customer ID : {}".format(customer_id))
            print(database)
            continue
        context_features = database_features.to_dict(orient='records')[0]
        context_features.pop('instance')
        # Create serialized TFRecord proto
        context_proto_str = serialize_context_from_dictionary(context_features)


        """Serialize peritem features. AKA examples or instances that will be ranked
        This is a list of serialized tf.train.Example objects"""
        # Get a list of Clustering primary keys related to customer_id
        sel = sqlalchemy.select([Clustering.id, Clustering.correct_k])\
            .where(Clustering.customer_id.__eq__(customer_id))
        res = Insert.core_select_execute(sel)
        if len(res) == 0:
            # No clustering examples were found with the database
            print("Skipped {} No results".format(customer_name))
            continue
        primary_keys = [x.id for x in res]
        correct_k = res[0].correct_k
        # From primary keys create records. Records are used to find
        # Example features and labels
        records = get_records(primary_keys)
        # best_labels.hyperparameter_dict values are the peritem_features
        # The loss metric related to each hyperparameter_dict are labels to
        # each example
        best_labels = ExtractLabels.calc_labels(records, correct_k,
                                                error_scale=0.8, var_scale=0.2)
        example_features = []
        for label in best_labels:
            feature_dict = {}
            for key in peritem_keys:
                feature_dict[key] = label.hyperparameter_dict[key]
            feature_dict[label_key] = label.loss
            example_features.append(feature_dict)

        peritem_list = serialize_examples_from_dictionary(example_features,
                                       label_key=label_key,
                                       peritem_keys=peritem_keys,
                                       reciprocal=reciprocal,
                                       n_bins=n_bins,
                                       shuffle_peritem=shuffle_peritem)

        """Prepare serialized feature spec for EIE format"""
        serialized_dict = {'serialized_context':_bytes_feature([context_proto_str]),
                           'serialized_examples':_bytes_feature(peritem_list)
                           }

        # Convert dictionary to tf.train.Example object
        serialized_proto = tf.train.Example(
                features=tf.train.Features(feature=serialized_dict))
        serialized_str = serialized_proto.SerializeToString()

        tfrecord_writer.write(serialized_str)

    tfrecord_writer.close()

    return None

#%%
if __name__ == '__main__':
    # Set up connection to SQL
    Insert = extract.Insert(server_name,
                            driver_name,
                            database_name)

    # Peritem keys to serialize
    peritem_keys = ['by_size','n_components','clusterer','reduce','index']
    # Name of label in TFRecord
    label_key = 'relevance'

    # location to save TFRecord files
    test_file = r'../data/JV_test_text_binned.tfrecords' # Testing
    train_file = r'../data/JV_train_text_binned.tfrecords' # Training

    """Get primary keys of Customers and divide primary keys into training and
    Testing sets
    NOTE train_ids and test_ids are PRIMARY KEYS, and NOT indexes of any array"""
    _train_pct = 0.8
    train_ids, test_ids = get_train_test_id_sql(train_pct=_train_pct)

    """Set up how I want to assign labels to objects
    Reciprocal will cause labels to be the inverse of the loss metric
    Set to True if I do not want labels to be binned"""
    reciprocal = False # Reciprocal of relevance label - use if you dont bin labels
    n_bins = 5 # number of bins for relevance label
    shuffle_peritem = True

    if os.path.isfile(train_file):
        _confirm = input(f'{train_file} already exists. Overwrite?\n>>>')
        if _confirm not in ['Y','y','Yes','yes','True','true']:
            raise SystemExit('Script execution stopped to not overwrite file')

    train_writer = tf.io.TFRecordWriter(train_file)
    test_writer = tf.io.TFRecordWriter(test_file)


    save_tfrecord_sql(train_ids, peritem_keys,
                      label_key, reciprocal,
                      n_bins, train_writer)
    save_tfrecord_sql(test_ids, peritem_keys,
                      label_key, reciprocal,
                      n_bins, test_writer)







