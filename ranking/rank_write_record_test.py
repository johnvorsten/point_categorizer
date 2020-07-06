# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:34:41 2019

@author: z003vrzk
"""
# Python imports
import sys
import os

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
                                       get_test_train_id,
                                       _bytes_feature)
from ranking import Labeling
from transform import transform_pipeline
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter)
from extract.SQLAlchemyDataDefinition import Labeling as SQLTableLabeling
from clustering.accuracy_visualize import Record, get_records

ExtractLabels = Labeling.ExtractLabels()

#%% Save text clusterer_index per-item features


def test_mongo_serialize_context():
    """Serialize the database features (context features) from a known document
    A document is a python dictionary, or a record from the Mongo database"""

    client = MongoClient('localhost', 27017)
    db = client['master_points']
    collection = db['raw_databases']

    customer_name = 'D:\\Z - Saved SQL Databases\\44OP-117216_UMHB_Stadium\\JobDB.mdf'
    document = collection.find_one({'database_tag':customer_name})

    # Bytes tf.Record proto string
    context_proto_str = serialize_context(document)

    return context_proto_str


def test_mongo_serialize_examples():
    """Serialize the example features (peritem_list) from a known document in
    MongoDB
    A document is a python dictionary when returned, or a record from the mongo
    database"""


    """Set up how I want to assign labels to objects
    Reciprocal will cause labels to be the inverse of the loss metric
    Set to True if I do not want labels to be binned"""
    reciprocal = False # Reciprocal of relevance label - use if you dont bin labels
    n_bins = 5 # number of bins for relevance label

    client = MongoClient('localhost', 27017)
    db = client['master_points']
    collection = db['raw_databases']

    customer_name = 'D:\\Z - Saved SQL Databases\\44OP-117216_UMHB_Stadium\\JobDB.mdf'
    document = collection.find_one({'database_tag':customer_name})

    """peritem_list looks like this
    [b'\n\x83\x01\n\x15\n\trelevance\x12\x08\x12\x06\n\x04\x00\x00\x80@\n\x14\n\x07by_size\x12\t\n\x07\n\x05False\n\x15\n\x0cn_components\x12\x05\n\x03\n\x018\n\x11\n\x06reduce\x12\x07\n\x05\n\x03MDS\n\x17\n\tclusterer\x12\n\n\x08\n\x06Ward.D\n\x11\n\x05index\x12\x08\n\x06\n\x04Duda',
    b'\n\x84\x01\n\x15\n\trelevance\x12\x08\x12\x06\n\x04\x00\x00\x80@\n\x14\n\x07by_size\x12\t\n\x07\n\x05False\n\x15\n\x0cn_components\x12\x05\n\x03\n\x018\n\x11\n\x06reduce\x12\x07\n\x05\n\x03MDS\n\x18\n\tclusterer\x12\x0b\n\t\n\x07ward.D2\n\x11\n\x05index\x12\x08\n\x06\n\x04Ball']
    It is a list of serialized TFRecrod protos"""
    peritem_list = serialize_examples_v2(document,
                                         reciprocal=reciprocal,
                                         n_bins=n_bins,
                                         shuffle_peritem=True)

    return peritem_list

def test_mongo_serialize_example_in_example():
    """Example in example (EIE) is a nested TFRecord proto mandated by the
    Tensorflow ranking input function
    It is of the form {'serialized_context':tf.train.Feature(bytes_list=tf.train.BytesList(value=value)),
                       'serialized_examples': tf.train.Feature(bytes_list=tf.train.BytesList(value=value))}"""

    # Set up a connection to mongoDB
    client = MongoClient('localhost', 27017)
    db = client['master_points']
    collection = db['raw_databases']

    """Set up how I want to assign labels to objects
    Reciprocal will cause labels to be the inverse of the loss metric
    Set to True if I do not want labels to be binned"""
    reciprocal = False # Reciprocal of relevance label - use if you dont bin labels
    n_bins = 5 # number of bins for relevance label

    # Retrieve information from mongo
    customer_name = 'D:\\Z - Saved SQL Databases\\44OP-117216_UMHB_Stadium\\JobDB.mdf'
    document = collection.find_one({'database_tag':customer_name})

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

    return serialized_str





#%% Test serialize context features

def test_serialize_context_from_dictionary():

    # Instantiate local classes
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
                              ('text_pipe',text_pipe),
                              ])
    # Set up connection to SQL
    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    # Get a points dataframe
    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    database = Insert.pandas_select_execute(sel)
    sel = sqlalchemy.select([Customers.name]).where(Customers.id.__eq__(customer_id))
    customer_name = Insert.core_select_execute(sel)[0].name

    database_features = ExtractLabels.get_database_features(database,
                                                            full_pipeline,
                                                            instance_name=customer_name)

    context_features = database_features.to_dict(orient='records')[0]
    context_features.pop('instance')
    serialized_context = serialize_context_from_dictionary(context_features)

    return serialized_context


def test_serialize_examples_from_dictionary():
    """This module has (3) methods of serializing peritem """

    """Set up how I want to assign labels to objects
    Reciprocal will cause labels to be the inverse of the loss metric
    Set to True if I do not want labels to be binned"""
    reciprocal = False # Reciprocal of relevance label - use if you dont bin labels
    n_bins = 5 # number of bins for relevance label

    label_key = 'relevance'

    # These are peritem featuer columns names
    peritem_keys = ['by_size','n_components','clusterer','reduce','index']


    # Set up connection to SQL
    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    # Get all records relating to one customer
    customer_id = 15
    sel = sqlalchemy.select([Clustering.id, Clustering.correct_k])\
        .where(Clustering.customer_id.__eq__(customer_id))
    res = Insert.core_select_execute(sel)
    primary_keys = [x.id for x in res]
    correct_k = res[0].correct_k

    sel = sqlalchemy.select([Customers.name]).where(Customers.id.__eq__(customer_id))
    customer_name = Insert.core_select_execute(sel)[0].name

    # Calculate ranking of all records
    records = get_records(primary_keys)
    best_labels = ExtractLabels.calc_labels(records, correct_k,
                                            error_scale=0.8, var_scale=0.2)
    example_features = []
    for label in best_labels:
        feature_dict = {}
        for key in peritem_keys:
            feature_dict[key] = label.hyperparameter_dict[key]
        feature_dict[label_key] = label.loss
        example_features.append(feature_dict)

    serialized_example = serialize_examples_from_dictionary(example_features,
                                       label_key,
                                       peritem_keys,
                                       reciprocal=reciprocal,
                                       n_bins=n_bins,
                                       shuffle_peritem=True)

    return serialized_example


#%% Retrieving written objects

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
    """Returns the example feature columns. Use "./data/JV_train_binned.tfrecords"
    for this feature_column function
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

def example_feature_columns_v2():
    """Returns the example feature columns for version 2 of
    serialize_examples_v2
    """
    _file_name_bysize = r'./data/JV_vocab_bysize.txt'
    _file_name_clusterer = r'./data/JV_vocab_clusterer.txt'
    _file_name_index = r'./data/JV_vocab_index.txt'
    _file_name_n_components = r'./data/JV_vocab_n_components.txt'
    _file_name_reduce = r'./data/JV_vocab_reduce.txt'

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

    peritem_feature_cols = {
        'by_size':by_size_indicator,
        'clusterer':clusterer_indicator,
        'index':index_indicator,
        'n_components':n_components_indicator,
        'reduce':reduce_indicator
        }

    return peritem_feature_cols


def input_fn(path, num_epochs=None, shuffle=True):

    context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())

    label_column = tf.feature_column.numeric_column(
    _LABEL_FEATURE, dtype=tf.float32, default_value=_PADDING_LABEL)

    example_feature_spec = tf.feature_column.make_parse_example_spec(
          list(example_feature_columns_v2().values()) + [label_column])

    dataset = tfr.data.build_ranking_dataset(
        file_pattern=path,
        data_format=tfr.data.EIE,
        batch_size=_BATCH_SIZE,
        list_size=_LIST_SIZE,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=shuffle,
        num_epochs=num_epochs) # this suffles through the dataset forever

    #  iterator = tf.data.make_one_shot_iterator(dataset)
    features = tf.data.make_one_shot_iterator(dataset).get_next()

    label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
    label = tf.cast(label, tf.float32)

    return features, label



#%% Testing input_fn




tf.enable_eager_execution()
tf.executing_eagerly()
#tf.set_random_seed(1234)
tf.logging.set_verbosity(tf.logging.INFO)

_Encoded_labels_dimension = 37
# by_size, clusterer, n_components, reduce, index
_Encoded_labels_dimension_v2 = 5
_VOCABULARY_FILE = r'data/JV_vocab_all.txt'

_LABEL_FEATURE = 'relevance'
_BATCH_SIZE = 5
_LIST_SIZE = 200
_PADDING_LABEL = -1
_TEST_DATA_PATH = r'data\JV_test_text_binned.tfrecords'

features, label = input_fn(_TEST_DATA_PATH)

context_features, example_features = tfr.feature.encode_listwise_features(
    features=features,
    input_size=_LIST_SIZE,
    context_feature_columns=context_feature_columns(),
    example_feature_columns=example_feature_columns_v2(),
    mode=tf.estimator.ModeKeys.TRAIN,
    scope="transform_layer")



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


_TEST_DATA_PATH = r'data\JV_test_text_binned.tfrecords'

record_iterator = tf.python_io.tf_record_iterator(path=_TEST_DATA_PATH)
string_record = next(record_iterator)

example = tf.train.Example()
example.ParseFromString(string_record)

feature_spec = {
    "serialized_context": tf.io.FixedLenFeature([1], tf.string),
    "serialized_examples": tf.io.VarLenFeature(tf.string),
    }

parsed_eie = tf.compat.v1.io.parse_example([string_record], feature_spec)

# See https://stackoverflow.com/questions/49588382/how-to-convert-float-array-list-to-tfrecord
# Define feature_spec
context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())

label_column = tf.feature_column.numeric_column(
    'relevance', dtype=tf.float32, default_value=-1)

example_feature_spec = tf.feature_column.make_parse_example_spec(
          list(example_feature_columns().values()) + [label_column])

example_feature_spec_v2 = tf.feature_column.make_parse_example_spec(
          list(example_feature_columns_v2().values()) + [label_column])

# Parse features from string_record using feature_spec
parsed_features = tfr.python.data.parse_from_example_in_example(
          [string_record],
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec)

# Parse features from string_record using feature_spec
parsed_features_v2 = tfr.python.data.parse_from_example_in_example(
          [string_record],
          context_feature_spec=context_feature_spec,
          example_feature_spec=example_feature_spec_v2)

# tfr function transforms sparse to dense
context_features, example_features = tfr.feature.encode_listwise_features(
    features=features,
    input_size=_LIST_SIZE,
    context_feature_columns=context_feature_columns(),
    example_feature_columns=example_feature_columns_v2(),
    mode=tf.estimator.ModeKeys.PREDICT,
    scope="transform_layer")

# Get feature columns only for casting to dense tensor
peritem_feature_cols = list(example_feature_columns_v2().values())

# Print combined dense input of parsed example (doesnt work for this format)
tf.keras.layers.DenseFeatures(peritem_feature_cols[0])(parsed_features_v2).numpy()

# Whole shebang
for key, value in parsed_features_v2.items():
    if hasattr(value, 'numpy'):
        print(key, value.numpy())
    else:
        _example_col = [_col for _col in peritem_feature_cols
                if _col.name.__contains__(key)]
        dense_col = tf.keras.layers.DenseFeatures(_example_col)
        print(key, dense_col(parsed_features_v2).numpy(), value.values.numpy())















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