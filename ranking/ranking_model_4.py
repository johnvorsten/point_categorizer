# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:41:03 2019

Ranking function for project (use this file to train and evaluate)

model_1 is my initial shot at making a good ranking model.
model_2 (THIS FILE) will try training binned labels instead of continuous labels

First impressions :


Some things to try :
1)
(NOT IMPLEMNTED HERE) Embed label space - see tensorflow documentation
for encoded feature space tip
2)
Reduce example list size (first runs were 237 - this leads to a huge input
layer of (237 * 37) + 11 = 8780 neurons input)
3)
Try binning input data into a non-continuous spectrum


@author: z003vrzk
"""
# Python imports
import sys
import os
from datetime import datetime

# Third party imports
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


tf.enable_eager_execution()
tf.executing_eagerly()
#tf.set_random_seed(1234)
tf.logging.set_verbosity(tf.logging.INFO)

#%% Tensorflow ranking setup

"""These functions return feature_spec dictionaries to retrieve data.
They also perform feature (text) embedding.
I will not use feature embedding, but I will document what is happening so I
understand it.

tf.feature_column.categorical_column_with_vocabulary_file
Inputs are in string or integer format. Vocabulary file maps value to an
integer ID.
tf.feature_column.embedding_column
Used when you have sparse inputs and want to convert to dense column.
Inputs must be a categorical_column (see above)
tf.feature_column.make_parse_example_spec

"""

# Store the paths to files containing training and test instances.
_TRAIN_DATA_PATH = r'../data/JV_train_text_binned.tfrecords'
_TEST_DATA_PATH = r'../data/JV_test_text_binned.tfrecords'

# The maximum number of documents per query in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 125

# The document relevance label.
_LABEL_FEATURE = "relevance"
_SHUFFLE_PERITEM = False

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1 # For non-reciprocal ranking

# Learning rate for optimizer.
_LEARNING_RATE = 0.1

# Parameters to the scoring function.
_BATCH_SIZE = 5
_HIDDEN_LAYER_DIMS = ["500", "250", "125", "125"]
_DROPOUT_RATE = 0.3
_GROUP_SIZE = _LIST_SIZE # Should this be set to _LIST_SIZE?

# Location of model directory and number of training steps.
if __name__ == '__main__':
    _confirm = input('Are you sure you want to create a new directory?\
                     (True/False)\n>>> ')
    if _confirm in ['y','Y','True','true','yes','Yes']:
        _now = datetime.now().strftime('%Y%m%d%H%M%S')
        _name = 'Run_' + _now + 'model4'
        _MODEL_DIR = os.path.join(r".\TF_Logs\ranking_model_dir", _name)

_NUM_TRAIN_STEPS = 20000
NUM_PARALLEL_EXEC_UNITS = 6


#%%

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

def example_feature_columns_v2():
    """Returns the example feature columns for version 2 of
    serialize_examples_v2
    """
    _file_name_bysize = r'../data/vocab_bysize.txt'
    _file_name_clusterer = r'../data/vocab_clusterer.txt'
    _file_name_index = r'../data/vocab_index.txt'
    _file_name_n_components = r'../data/vocab_n_components.txt'
    _file_name_reduce = r'../data/vocab_reduce.txt'

    by_size = tf.feature_column.categorical_column_with_vocabulary_file(
        'by_size',
        _file_name_bysize,
        dtype=tf.string,
        vocabulary_size=2)
    clusterer = tf.feature_column.categorical_column_with_vocabulary_file(
        'clusterer',
        _file_name_clusterer,
        dtype=tf.string,
        vocabulary_size=4)
    index = tf.feature_column.categorical_column_with_vocabulary_file(
        'index',
        _file_name_index,
        dtype=tf.string,
        vocabulary_size=29)
    n_components = tf.feature_column.categorical_column_with_vocabulary_file(
        'n_components',
        _file_name_n_components,
        dtype=tf.string,
        vocabulary_size=3)
    reduce = tf.feature_column.categorical_column_with_vocabulary_file(
        'reduce',
        _file_name_reduce,
        dtype=tf.string,
        vocabulary_size=1)

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
  """path : (str) path of tfrecord EIE file
  num_epochs : (int) or (None) how many times to iterate through dataset
  shuffle : (bool) shuffle dataset or not"""

  context_feature_spec = tf.feature_column.make_parse_example_spec(
          context_feature_columns().values())

  # tf.feature_column.NumericColumn
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
    num_epochs=num_epochs)

  features = tf.data.make_one_shot_iterator(dataset).get_next()

  label = tf.squeeze(features.pop(_LABEL_FEATURE), axis=2)
  label = tf.cast(label, tf.float32)

  return features, label



def make_transform_fn():

  def _transform_fn(features, mode):
    """Defines transform_fn."""
    context_features, example_features = tfr.feature.encode_listwise_features(
        features=features,
        input_size=_LIST_SIZE,
        context_feature_columns=context_feature_columns(),
        example_feature_columns=example_feature_columns_v2(),
        mode=mode,
        scope="transform_layer")

    return context_features, example_features

  return _transform_fn



def make_score_fn():
  """Returns a scoring function to build `EstimatorSpec`."""

  def _score_fn(context_features, group_features, mode, params, config):
    """Defines the network to score a group of documents."""

    with tf.compat.v1.name_scope("input_layer"):
      context_input = [
          tf.compat.v1.layers.flatten(context_features[name])
          for name in sorted(context_feature_columns())
      ]
      group_input = [
          tf.compat.v1.layers.flatten(group_features[name])
          for name in sorted(example_feature_columns_v2())
      ]
      input_layer = tf.concat(context_input + group_input, 1)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    cur_layer = input_layer
    cur_layer = tf.compat.v1.layers.batch_normalization(
      cur_layer,
      training=is_training,
      momentum=0.99)

    for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
      cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
      cur_layer = tf.compat.v1.layers.batch_normalization(
        cur_layer,
        training=is_training,
        momentum=0.99)
      cur_layer = tf.nn.relu(cur_layer)
      cur_layer = tf.compat.v1.layers.dropout(
          inputs=cur_layer, rate=_DROPOUT_RATE, training=is_training)

    logits = tf.compat.v1.layers.dense(cur_layer, units=_GROUP_SIZE)

    return logits

  return _score_fn



def eval_metric_fns():
  """Returns a dict from name to metric functions.

  This can be customized as follows. Care must be taken when handling padded
  lists.

  def _auc(labels, predictions, features):
    is_label_valid = tf_reshape(tf.greater_equal(labels, 0.), [-1, 1])
    clean_labels = tf.boolean_mask(tf.reshape(labels, [-1, 1], is_label_valid)
    clean_pred = tf.boolean_maks(tf.reshape(predictions, [-1, 1], is_label_valid)
    return tf.metrics.auc(clean_labels, tf.sigmoid(clean_pred), ...)
  metric_fns["auc"] = _auc

  Returns:
    A dict mapping from metric name to a metric function with above signature.
  """
  metric_fns = {}
  metric_fns.update({
      "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
          tfr.metrics.RankingMetricKey.NDCG,
          topn=topn,
          name='metrics')
      for topn in [5, 10, 25]
  })

  return metric_fns



def make_train_op_fn(optimizer):

  def _train_op_fn(loss):
    """Defines train op used in ranking head."""
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    minimize_op = optimizer.minimize(
        loss=loss, global_step=tf.compat.v1.train.get_global_step())

    train_op = tf.group([update_ops, minimize_op])

    return train_op

  return _train_op_fn



def train_and_eval_fn():
  """Train and eval function used by `tf.estimator.train_and_evaluate`."""
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=100,
      model_dir=_MODEL_DIR)

  ranker = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=_MODEL_DIR,
      config=run_config,
      params={'shuffle_peritem':True})

  train_input_fn = lambda: input_fn(_TRAIN_DATA_PATH)
  eval_input_fn = lambda: input_fn(_TEST_DATA_PATH, num_epochs=1)

  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=_NUM_TRAIN_STEPS)

  eval_spec =  tf.estimator.EvalSpec(
          name="eval",
          input_fn=eval_input_fn,
          throttle_secs=15,
          exporters=None)

  return (ranker, train_spec, eval_spec)

#%%

# Testing input_fn
features, label = input_fn(_TRAIN_DATA_PATH)

#%%
"""The steps for working with tensorflow-ranking

1) Choose loss functions, optimizers, and evaluation metrics
2) Create a ranking head. The ranking head takes care of metrics,
  losses, and optimizers
3) Create a model_function. The inputs to the model function are
  The ranking head
  A scoring function (creats logits)
  Feature transformation functions
4) Create the estimator object. The estimator will be used for training and
  evaluation. the Estimator inputs are a
  model function,
  runtime configurations,
  hyperparameters

5) Because training and evaluation specs are not defined in the model_fn,
  we define them after.  Training and evaluation specs are used to train
  and evaluate the model

6) Train and evaluate the model

"""

# Define a loss function. To find a complete list of available
# loss functions or to learn how to add your own custom function
# please refer to the tensorflow_ranking.losses module.

# APPROX_NDCG_LOSS
#_LOSS = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
#loss_fn = tfr.losses.make_loss_fn(_LOSS)
# Softmax loss
_LOSS = tfr.losses.RankingLossKey.SOFTMAX_LOSS
loss_fn = tfr.losses.make_loss_fn(_LOSS)


# Define an optimizer to pass to
optimizer = tf.compat.v1.train.AdagradOptimizer(
    learning_rate=_LEARNING_RATE)

ranking_head = tfr.head.create_ranking_head(
      loss_fn=loss_fn,
      eval_metric_fns=eval_metric_fns(),
      train_op_fn=make_train_op_fn(optimizer))

model_fn = tfr.model.make_groupwise_ranking_fn(
          group_score_fn=make_score_fn(),
          transform_fn=make_transform_fn(),
          group_size=_GROUP_SIZE,
          ranking_head=ranking_head)

run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=100,
    model_dir=_MODEL_DIR,
    session_config=tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
        inter_op_parallelism_threads=2,
        allow_soft_placement=True,
        device_count={'CPU':NUM_PARALLEL_EXEC_UNITS}))

ranker = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=_MODEL_DIR,
    config=run_config,
    params={'shuffle_peritem':_SHUFFLE_PERITEM})

# Creating input functions for training and evaluation
# Create training and evaluatino specs

train_input_fn = lambda: input_fn(_TRAIN_DATA_PATH)
eval_input_fn = lambda: input_fn(_TEST_DATA_PATH, num_epochs=1)

train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn, max_steps=_NUM_TRAIN_STEPS)

eval_spec =  tf.estimator.EvalSpec(
        name="eval",
        input_fn=eval_input_fn,
        throttle_secs=15,
        exporters=None)

tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)

#%% Tensorboard


def launch_TensorBoard(tracking_address=r'.\TF_Logs\ranking_model_dir'):
    from tensorboard import program
    tracking_address = r'TF_Logs'
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    return url


#%% Predictions


#def predict_input_fn(path):
#  context_feature_spec = tf.feature_column.make_parse_example_spec(
#        context_feature_columns().values())
#  example_feature_spec = tf.feature_column.make_parse_example_spec(
#        list(example_feature_columns().values()))
#  dataset = tfr.data.build_ranking_dataset(
#        file_pattern=path,
#        data_format=tfr.data.EIE,
#        batch_size=_BATCH_SIZE,
#        list_size=_LIST_SIZE,
#        context_feature_spec=context_feature_spec,
#        example_feature_spec=example_feature_spec,
#        reader=tf.data.TFRecordDataset,
#        shuffle=False,
#        num_epochs=1)
#  features = tf.data.make_one_shot_iterator(dataset).get_next()
#  return features
#
#predictions = ranker.predict(input_fn=lambda: predict_input_fn(_TEST_DATA_PATH))
#
#x = next(predictions)
#for _idx, x in enumerate(predictions):
#  print(_idx)



#%% Saving a model



"""
!! OLD !!
# Creating an input function
feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

# This serving_input_receiver_fn expects tf.Examples -> So i need to serialize
# My prediction data before I pass is to this model
# Other option is to build a custom serving_input_receiver_fn
# that outputs a tf.estimator.export.ServingInputReceiver(features, receiver_tensros)
input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)
"""

# Create an export directory
export_dir = os.path.join('final_model\Run_20191024002109model4')

def serving_input_receiver_fn():

    _file_name_bysize = r'./data/JV_vocab_bysize.txt'
    _file_name_clusterer = r'./data/JV_vocab_clusterer.txt'
    _file_name_index = r'./data/JV_vocab_index.txt'
    _file_name_n_components = r'./data/JV_vocab_n_components.txt'
    _file_name_reduce = r'./data/JV_vocab_reduce.txt'

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


model_dir = ranker.export_saved_model(
    export_dir,
    serving_input_receiver_fn,
    as_text=False
    )











