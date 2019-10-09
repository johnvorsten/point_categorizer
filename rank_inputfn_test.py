# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 14:57:25 2019

@author: z003vrzk
"""

import tensorflow as tf
import tensorflow_ranking as tfr
import six
from tensorflow_ranking.python import utils


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




val_flt = 123.3254
val_int = 5232
val_str = str.encode('foo')
feature = {
        'feature_val':_float_feature(val_flt),
        'feature_int':_int64_feature(val_int),
        'feature_bytes':_bytes_feature(val_str)
        }

# Create TFRecord object
example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

# Serialize TFRecord object for saving
serialized_example = example_proto.SerializeToString() # bytes object

# Writing to a file
file_name = r'data\test_file.tfrecords'
with tf.python_io.TFRecordWriter(file_name) as writer:
    writer.write(serialized_example) # bytes object

# Reading from a file
record_iterator = tf.python_io.tf_record_iterator(path=file_name)
string_record = next(record_iterator) # bytes object

# Parse bytes object into TFRecord obect (tf.train.Example)
example = tf.train.Example()
example.ParseFromString(string_record)

# OR
example2 = tf.train.Example.FromString(string_record)




#%% Example Reading an existing

import urllib.request
import os

_existing_file = r'data\train_ANTIQUE.tfrecords'

if not os.path.exists(_existing_file):
    _url = r'https://ciir.cs.umass.edu/downloads/Antique/tf-ranking/train.tfrecords'
    _url2 = r'https://ciir.cs.umass.edu/downloads/Antique/tf-ranking/vocab.txt'
    urllib.request.urlretrieve(_url, _existing_file)
    urllib.request.urlretrieve(_url2, r'data\vocab.txt')

#%% Reading

# Reading data with tf.data.TFRecordDataset
raw_dataset = tf.data.TFRecordDataset(_existing_file) # TFRecordDataset
test = raw_dataset.take(1)

# Reading data with python_io.tf_record_iterator
record_iterator = tf.python_io.tf_record_iterator(path=_existing_file)

string_record = next(record_iterator)

#%% First Example Layer
feature_spec = {
    "serialized_context": tf.io.FixedLenFeature([1], tf.string),
    "serialized_examples": tf.io.VarLenFeature(tf.string),
}
features = tf.compat.v1.io.parse_example([string_record], feature_spec)

#%% Second Example Layer
# Using tensorflow_ranking functions
context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_feature_columns().values())
  # tf.feature_column.NumericColumn
  # Key should be label (relevance)
  # default value is what to pad with
label_column = tf.feature_column.numeric_column(
        _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
  # feature_spec is {'example_feature':tf.VarLenFeature, 
  #                  'relevance':tf.FixedLenFeature}
  # EmbeddingColumn + NumericColumn
example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()) + [label_column])

parsed_features = tfr.data.parse_from_example_in_example(
        [string_record],
        list_size=50,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec)

parsed_features['query_tokens']._values





#%% Examining from [...]/tensorflow_ranking/examples/handling_sparse_features.ipynb

# Third party imports
import six
import os
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

tf.enable_eager_execution()
tf.executing_eagerly()
tf.set_random_seed(1234)
tf.logging.set_verbosity(tf.logging.INFO)

# Store the paths to files containing training and test instances.
_TRAIN_DATA_PATH = r"data\train.tfrecords"
#_TEST_DATA_PATH = "/tmp/test.tfrecords"

# Store the vocabulary path for query and document tokens.
_VOCAB_PATH = r"data\vocab.txt"

# The maximum number of documents per query in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 50

# The document relevance label.
_LABEL_FEATURE = "relevance"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1

# Learning rate for optimizer.
_LEARNING_RATE = 0.05

# Parameters to the scoring function.
_BATCH_SIZE = 32
_HIDDEN_LAYER_DIMS = ["64", "32", "16"]
_DROPOUT_RATE = 0.8
_GROUP_SIZE = 1  # Pointwise scoring.

# Location of model directory and number of training steps.
_MODEL_DIR = "/tmp/ranking_model_dir"
_NUM_TRAIN_STEPS = 15 * 1000


_EMBEDDING_DIMENSION = 20

def context_feature_columns():
  """Returns context feature names to column definitions."""
  sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
      key="query_tokens",
      vocabulary_file=_VOCAB_PATH)
  query_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  return {"query_tokens": query_embedding_column}

def example_feature_columns():
  """Returns the example feature columns."""
  sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
      key="document_tokens",
      vocabulary_file=_VOCAB_PATH)
  document_embedding_column = tf.feature_column.embedding_column(
      sparse_column, _EMBEDDING_DIMENSION)
  return {"document_tokens": document_embedding_column}

def input_fn(path, num_epochs=None):
  # context_feature_spec is {'feature':tf.VarLenFeature}
  context_feature_spec = tf.feature_column.make_parse_example_spec(
        context_feature_columns().values())
  # tf.feature_column.NumericColumn
  # Key should be label (relevance)
  # default value is what to pad with
  label_column = tf.feature_column.numeric_column(
        _LABEL_FEATURE, dtype=tf.int64, default_value=_PADDING_LABEL)
  # feature_spec is {'example_feature':tf.VarLenFeature, 
  #                  'relevance':tf.FixedLenFeature}
  # EmbeddingColumn + NumericColumn
  example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()) + [label_column])
  
  dataset = tfr.data.build_ranking_dataset(
        file_pattern=path,
        data_format=tfr.data.EIE,
        batch_size=_BATCH_SIZE,
        list_size=_LIST_SIZE,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=False,
        num_epochs=num_epochs)
  features = tf.data.make_one_shot_iterator(dataset).get_next()
  label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
  label = tf.cast(label, tf.float32)
  return features, label

features, label = input_fn(_TRAIN_DATA_PATH)


for key, val in features.items():
#    print(key, type(val))
    print(key, val.shape)



