# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:48:46 2019

@author: z003vrzk
"""

# Third party imports
import tensorflow as tf
import tensorflow_ranking as tfr


tf.enable_eager_execution()
tf.executing_eagerly()
#tf.set_random_seed(1234)
tf.logging.set_verbosity(tf.logging.INFO)


#%% using tensorflow

# Paths to files for storing example-in-example information
_TRAIN_DATA_PATH = r'./data/JV_train_text.tfrecords'
_TEST_DATA_PATH = r'./data/JV_test_text.tfrecords'

# The maximum number of documents per query in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 150
_INPUT_SIZE = _LIST_SIZE

# Location of all clust_index vocabulary (per-item features)
# Hyperparameters related to per-item features
_VOCABULARY_FILE = r'./data/JV_vocab.txt'
_VOCABULARY_SIZE = 37 # All should be this long
_N_CATEGORIES = 37
_SUGGESTED_EMB_DIM = _N_CATEGORIES ** (1/4) # 2.4
_EMBEDDING_DIMENSION = 5

# Labels for scoring
_PADDING_LABEL = 0
_LABEL_FEATURE = 'relevance'

# Batch size for feeding into network - it affects the third dimension of tensor
_BATCH_SIZE = 15


#%% Input Function and feature specs

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
    sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
            'clust_index',
            _VOCABULARY_FILE,
            vocabulary_size=_VOCABULARY_SIZE
            )
    per_item_embedding_column = tf.feature_column.embedding_column(
            sparse_column,
            _EMBEDDING_DIMENSION)
    
    peritem_feature_cols = {
        'encoded_clust_index':per_item_embedding_column
        }
    
    return peritem_feature_cols


def input_fn(path, num_epochs=None):
  
  #  context_feature_spec = context_feature_columns()
  context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())
  
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
    shuffle=False, # Should shuffle be True?
    num_epochs=None) # num_empoch shuffles throug dataset forever
  
  features = tf.data.make_one_shot_iterator(dataset).get_next()
  
  label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
  label = tf.cast(label, tf.float32)
  
  return features, label

features_decoded, label = input_fn(_TRAIN_DATA_PATH)

#%% Examine what the serialized example_list looks like
# Here is what the example set looks like
"""
parsed_features['query_tokens']._values
Out[131]: 
<tf.Tensor: id=782, shape=(13,), dtype=string, numpy=
array([b'why', b'do', b'human', b'bee', b'##ing', b'h', b'##v', b'to',
       b'bel', b'##ive', b'in', b'god', b'?'], dtype=object)>
       
MY RESULT LOOKS LIKE THIS : 
parsed_features['clust_index']._values
Out[148]: <tf.Tensor: id=998, shape=(0,), dtype=string, 
numpy=array([], dtype=object)>

Why is it an empty array?
"""

_existing_file = r'data\JV_train_text.tfrecords'

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

label_column = tf.feature_column.numeric_column(
        _LABEL_FEATURE, dtype=tf.float32, default_value=_PADDING_LABEL)

example_feature_spec = tf.feature_column.make_parse_example_spec(
        list(example_feature_columns().values()) + [label_column])

parsed_features = tfr.data.parse_from_example_in_example(
        [string_record],
        list_size=50,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec)



#%% Try flattening features in transform_fn (aka encode_listwise_features)

# Name encoded_clust_index
example_feature_cols = list(example_feature_columns().values())

context_features, example_features = tfr.feature.encode_listwise_features(
        features=parsed_features,
        context_feature_columns=context_feature_columns(),
        example_feature_columns=example_feature_columns(),
        mode=tf.estimator.ModeKeys.TRAIN,
        input_size=_INPUT_SIZE
        )

# Encoded feature column
example_features['encoded_clust_index'].numpy()



#%% Try to convert features to dense values to read


from tensorflow.python.feature_column import feature_column_lib

dense_layer = feature_column_lib.DenseFeatures(
        feature_columns=list(example_feature_columns().values()),
        name='encoding_layer',
        trainable=True)

dense_layer(features)



parsed_features['clust_index']._values
parsed_features['clust_index']._dense_shape
parsed_features['clust_index'].dense_shape
parsed_features['clust_index'].values




#%% Did I save my features (strings) incorrectly?

# For text or sequence problems, the embeddign layer takes
# a 2D tensor of integers of shape (sampes, sequence_length)
# where each entry is a sequence of integers
# It can embed sequeces of variable length

# In the example I'm looking at the reviews are integer-encoded

# Createa a categorical feature column
sparse_column = tf.feature_column.categorical_column_with_vocabulary_file(
        'clust_index',
        _VOCABULARY_FILE,
        vocabulary_size=_VOCABULARY_SIZE
        )

# Pass categorical feature column to embedding column
per_item_embedding_column = tf.feature_column.embedding_column(
        sparse_column,
        dimension = _EMBEDDING_DIMENSION
        )

# Create a feature spec
peritem_feature_cols = {
    'encoded_clust_index':per_item_embedding_column
    }

col_list = list(peritem_feature_cols.values())

# This should return flattened dense tensors for training 
tfr.feature.encode_features(parsed_features,
                           col_list, 
                           mode=tf.estimator.ModeKeys.TRAIN
                           )


#%% 
# Try creating my own column data
text = [b'ward.D',b'kmeans',b'cindex']
text_tensor = tf.Variable(text)
text_batch = {'clust_index':text_tensor}

# Create a feature clumn -> Pass this to a embedding or indicator column
text_feature = tf.feature_column.categorical_column_with_vocabulary_file(
        'clust_index',
        _VOCABULARY_FILE,
        vocabulary_size=_VOCABULARY_SIZE)

# Create indicator or embeddign column -> Pass this to a layer
text_one_hot = tf.feature_column.indicator_column(text_feature)
text_embed = tf.feature_column.embedding_column(
        text_feature,
        embedding_dimension=5
        )

# Create a layer based on a feature column
# Frist, instantiate the layer by passin the feature_column
# Remember, the feature column does not contain data yet
# Fill in layer data with the batch data input
def demo(feature_column):
  feature_layer = tf.keras.layers.DenseFeatures(feature_column)
  print(feature_layer(text_batch).numpy())

demo(text_one_hot) # Indicator column
demo(text_embed) # embedding column







