# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:56:40 2019

@author: z003vrzk
"""

import tensorflow as tf
import numpy as np
import os
from datetime import datetime

tf.enable_eager_execution()
tf.executing_eagerly()
#tf.set_random_seed(1234)
tf.logging.set_verbosity(tf.logging.INFO)


#%% Hyperparameters

# Store the paths to files containing training and test instances.
_TRAIN_DATA_PATH = "./data/JV_train_classify.tfrecords"
_TEST_DATA_PATH = "./data/JV_test_classify.tfrecords"

# The document label - one of ['by_size','n_components','reduce']
_LABEL_FEATURE = "by_size"

# Learning rate for optimizer.
_LEARNING_RATE = 0.1

# Parameters to the model
_BATCH_SIZE = 15
_HIDDEN_LAYER_DIMS = [64, 32, 16]
_DROPOUT_RATE = 0.2

if __name__ == '__main__':
    _confirm = input('Are you sure you want to create a new directory? (True/False)')
    if _confirm in ['y','Y','True']:
        _now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        _name = 'Run_' + _now + 'model1'
        _MODEL_DIR = os.path.join(r"TF_Logs\classifier_dir", _name)
_NUM_TRAIN_STEPS = 1500

#%% Input function

# Create an input function

def feature_columns():
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
    
    feature_cols = {
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
    
    return feature_cols

#%% 

"""Making an input function
# This function must return a tuple of (features, labels)
# Where features is a dictionary of string:tensor and labels is a tensor

I can independently parse the features and labels from the TFRecord, but
it is easier to simply parse the whole dataset in _parse_fn and pop the labels
Then return a tuple of (features, labels) from the parse_fn
# Features only
feature_spec = tf.feature_column.make_parse_example_spec(
        feature_cols)
# Label only
label_spec = tf.feature_column.make_parse_example_spec(
        label_cols)
    
# Old functions - now I'm parsing the whole TFRecord at once
def _parse_fn_features(example_proto, feature_spec):
    return tf.io.parse_single_example(example_proto, feature_spec)
def _parse_fn_labels(example_proto, label_spec):
    return tf.io.parse_single_example(example_proto, label_spec)
parsed_feature_dataset = dataset.map(lambda x: _parse_fn_features(x, feature_spec))
parsed_label_dataset = dataset.map(lambda x: _parse_fn_labels(x, label_spec))

# How to get features from a datset to a dictionary or other useful item?
# Method 1 - create an iterator, not suggested anymore
iter_dataset = dataset.make_one_shot_iterator().get_next()
features_dict = tf.io.parse_example([iter_dataset], feature_spec)
print(features_dict['len_var'])

# Method 2 - take item from parsed dataset
 Maybe not the best
for parsed_record in parsed_feature_dataset.take(1):
    print(repr(parsed_record))
    for key, value in parsed_record.items():
        print(key,value.numpy())
"""


    
def input_fn(path, num_epochs=None, shuffle=True):
    
    # tf.feature_column.NumericColumn
    by_size = tf.feature_column.numeric_column(
        'by_size', dtype=tf.int64)
    n_components = tf.feature_column.categorical_column_with_vocabulary_list(
        'n_components', [8, 0, 2], dtype=tf.int64)
    n_components_encoded = tf.feature_column.indicator_column(n_components)
    reduce = tf.feature_column.categorical_column_with_vocabulary_list(
        'reduce', ['MDS', 'TSNE', 'False'], dtype=tf.string)
    reduce_encoded = tf.feature_column.indicator_column(reduce)
    
    feature_cols = list(feature_columns().values())
    
    label_cols = []
    
    if _LABEL_FEATURE == 'by_size':
        label_cols.append(by_size)
    elif _LABEL_FEATURE == 'n_components':
        label_cols.append(n_components_encoded)
    elif _LABEL_FEATURE == 'reduce':
        label_cols.append(reduce_encoded)
        
    example_cols = feature_cols + label_cols
    
    # Whole TFRecord
    example_spec = tf.feature_column.make_parse_example_spec(
            example_cols)
    
    dataset = tf.data.TFRecordDataset(filenames=path)
    
    # Testing
    example_proto = next(iter(dataset))

    def _parse_fn_better(example_proto, example_spec):
        """Parsing function for TFRecord dataset"""
        parsed_example = tf.io.parse_single_example(example_proto, example_spec)
        
        # Convert all values to float32
        for key, value in parsed_example.items():
            parsed_example.update({key:tf.cast(value, tf.float32)})
        
        # pop label and return tuple of (features_dict, label_tensor)
        label_tensor = parsed_example.pop(_LABEL_FEATURE)
        
        return (parsed_example, label_tensor)
    
    parsed_whole_dataset = dataset.map(lambda x: _parse_fn_better(x, example_spec))
    
    # method 3 - extract (featuer_dict, labels) in parse_fn
    # Recommended method
    if shuffle: 
        parsed_whole_dataset = parsed_whole_dataset.shuffle(_BATCH_SIZE)
    parsed_whole_dataset = parsed_whole_dataset.repeat(num_epochs)
    parsed_whole_dataset = parsed_whole_dataset.batch(_BATCH_SIZE)
    
    return parsed_whole_dataset

training_dataset = input_fn(_TRAIN_DATA_PATH) # dataset(dictionary, tensor)

# Testing
#for features_batch, labels_batch in input_fn(_TRAIN_DATA_PATH).take(1):
#    print(features_batch)
#    print(labels_batch)


#%% Baseline classifier

classifier = tf.estimator.BaselineClassifier(n_classes=2)

classifier.train(input_fn=lambda : input_fn(_TRAIN_DATA_PATH),
                 steps=200)

evaluation_base = classifier.evaluate(input_fn=lambda : input_fn(_TEST_DATA_PATH, num_epochs=1),
                           steps=45)


#%% Premade estimators

# Define feature columns
feature_cols = list(feature_columns().values())

classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_cols,
        hidden_units=[64,32],
        n_classes=2,
        optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=_LEARNING_RATE,
                l1_regularization_strength=0.001),
        model_dir=_MODEL_DIR,
        activation_fn=tf.nn.relu,
        dropout=0.2)

classifier.train(input_fn=lambda : input_fn(_TRAIN_DATA_PATH),
                 steps=_NUM_TRAIN_STEPS)

evaluation = classifier.evaluate(input_fn=lambda : input_fn(_TEST_DATA_PATH, num_epochs=1))



#%%

def launch_TensorBoard(tracking_address=r'.\TF_Logs\classifier_dir'):
    from tensorboard import program
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    return url


#%% model function

def model_fn(features, labels, mode, params):
    
    # Create the model fuction
    
    # Output of model function is logits
    logits = TODO
    
    
    predict = (mode == tf.estimator.ModeKeys.PREDICT)
    train == (mode == tf.estimator.ModeKeys.TRAIN)
    if predict:
        _, top_5 = tf.nn.top_k(predictions, k=5)
        predictions = {
                'top_1':tf.argmax(logits,-1),
                'top_5':top_5,
                'probabilities':tf.nn.softmax(logits),
                'logits':logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    if train:
        # return tf.estimator.EstimatorSpec
        # Need an optimizer
        pass
    
    pass #TODO 



# Model function
# Should retur ops necessary to perform training, evaluation, prediction
    # features = dictionary ('key':tensor)
    # labels = Tensor
    # mode = tf.train.ModeKeys.TRAIN or PREDICT or EVAL
    # params = dictionary of hyperparameters
    # config = tf.estimator.RunConfig
    
# Estimator object
    # model_fn = see above. outputs necessary training ops
    # model_dir = string
    # config = tf.estimator.RunConfig # This is passed to model_fn
    # params = hyperparameters (dict) # This is passed to model_fn
    
# tf.estimator.RunConfig
# Information about execution environment
    # Passed to model_fn
    # 




#%% Simple keras model, custom estimator w/ keras

# Activation layer should be sigmoid for binary (2-class) logistic regression
# Activation layer should be softmax for multi-classs logistic regression

# Create the input layer to the model
# Define feature columns
feature_cols = list(feature_columns().values())

feature_layer = tf.keras.layers.DenseFeatures(feature_columns=feature_cols)
training_dataset = input_fn(_TRAIN_DATA_PATH)
feature_batch, label_batch = training_dataset.make_one_shot_iterator().get_next()
feature_layer(feature_batch)

# Create an input layer - try InputLayer
# This will not work - I need to pass a numpy array or tensor to the keras
# Model. It will not accept multiple named tensors like in a dictionary
# My workaround will be to only use tensorflow estimators...
model_input = tf.keras.layers.InputLayer(input_shape=, name=)


model = tf.keras.models.Sequential([
#        model_input,
#        tf.keras.layers.Flatten(input_shape=(None,11)),
        tf.keras.layers.Dense(64, 
                              activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l1(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, 
                              activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, 
                              activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, 
                              activation='sigmoid')])


model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
#              optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, nesterov=True),
              metrics=[tf.keras.metrics.CategoricalCrossentropy()])

#model.fit(input_fn(_TRAIN_DATA_PATH)(), 
#          steps_per_epoch=1, 
#          epochs=5,
#          validation_data= input_fn(_TEST_DATA_PATH)(),
#          validation_steps=1)

model.summary() # Error because input_shape is not yet defined & graph is not built

# Convert keras model to estimator
model_dir = 'TF_Logs/classifier_dir/Run12345'
keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=model_dir)


keras_estimator.train(input_fn= lambda : input_fn(_TRAIN_DATA_PATH), steps=25)





