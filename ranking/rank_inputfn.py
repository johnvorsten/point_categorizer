# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:33:57 2019

Read example-in-example saved data into feature dicts for training and eval


@author: z003vrzk
"""

# Thrid party imports
import tensorflow as tf
import numpy as np
import tensorflow_ranking as tfr

# Declarations
# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1

_Encoded_labels_dimension = 37 # Number of label features possible

# Parameters to the scoring function.
_BATCH_SIZE = 50
_LIST_SIZE = 237

# The document relevance label.
_LABEL_FEATURE = "relevance"

#%%
def context_feature_columns():
    """Returns context feature names to column definitions.
    DEPRECITATED : 
    context_feature_spec = {
        'n_instance': tf.io.FixedLenFeature([], tf.float32),
        'n_features': tf.io.FixedLenFeature([], tf.float32),
        'len_var': tf.io.FixedLenFeature([], tf.float32),
        'uniq_ratio': tf.io.FixedLenFeature([], tf.float32),
        'n_len1': tf.io.FixedLenFeature([], tf.float32),
        'n_len2': tf.io.FixedLenFeature([], tf.float32),
        'n_len3': tf.io.FixedLenFeature([], tf.float32),
        'n_len4': tf.io.FixedLenFeature([], tf.float32),
        'n_len5': tf.io.FixedLenFeature([], tf.float32),
        'n_len6': tf.io.FixedLenFeature([], tf.float32),
        'n_len7': tf.io.FixedLenFeature([], tf.float32)
        }
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
    DEPRECIATED : 
     Default value is -1 to ignore its importance
    peritem_feature_spec = {
        'relevance':tf.io.FixedLenFeature([], tf.float32, default_value=[-1]),
        'encoded_clust_index':tf.VarLenFeature(tf.float32)
        }
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


def input_fn(path, num_epochs=None):
  # {'key': tf.io.FixedLenFeature}
  #  context_feature_spec = context_feature_columns()
  context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())
  
  # tf.feature_column.NumericColumn
  label_column = tf.feature_column.numeric_column(
    _LABEL_FEATURE, dtype=tf.float32, default_value=_PADDING_LABEL)
  
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
    shuffle=False, # TODO Should this be true?
    num_epochs=None) # this suffles through the dataset forever
  # Should this be an interator, not a dict?
#  iterator = tf.data.make_one_shot_iterator(dataset)
  features = tf.data.make_one_shot_iterator(dataset).get_next()
  
  label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
  label = tf.cast(label, tf.float32)
  
  return features, label

if __name__ == "__main__":
    _file = r".\data\JV_train_binned.tfrecords"
    features, label = input_fn(_file)
