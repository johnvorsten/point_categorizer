# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 09:02:37 2020

@author: z003vrzk
"""

# Python imports
import sys
import os
import configparser
from typing import Union

# Third party imports
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import (make_scorer, precision_score,
                             recall_score, accuracy_score, 
                             balanced_accuracy_score)
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from scipy.sparse import csr_matrix

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from mil_load import LoadMIL, load_mil_dataset
from bag_cross_validate import cross_validate_bag, BagScorer, bags_2_si

# Global declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

loadMIL = LoadMIL(server_name,
                  driver_name,
                  database_name)

#%%

# Load numeric dataset
_file = r'../data/MIL_dataset.dat'
_dataset = load_mil_dataset(_file)
_bags = _dataset['dataset']
_bag_labels = _dataset['bag_labels']

# Load categorical dataset
_cat_file = r'../data/MIL_cat_dataset.dat'
_cat_dataset = load_mil_dataset(_cat_file)
_cat_bags = _cat_dataset['dataset']
_cat_bag_labels = _cat_dataset['bag_labels']

# Split numeric dataset
rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
train_index, test_index = next(rs.split(_bags, _bag_labels))
train_bags, train_bag_labels = _bags[train_index], _bag_labels[train_index]
test_bags, test_bag_labels = _bags[test_index], _bag_labels[test_index]

# Split categorical dataset
train_index, test_index = next(rs.split(_cat_bags, _cat_bag_labels))
train_bags_cat, train_labels_cat = _cat_bags[train_index], _cat_bag_labels[train_index]
test_bags_cat, test_bag_labels_cat = _cat_bags[test_index], _cat_bag_labels[test_index]

# Unpack bags into single instances for training and testing
# Bags to single instances
si_X_train, si_y_train = bags_2_si(train_bags,
                                   train_bag_labels,
                                   sparse_input=True)
si_X_test, si_y_test = bags_2_si(test_bags,
                                 test_bag_labels,
                                 sparse_input=True)

# Unpack categorical
si_X_train_cat, si_y_train_cat = bags_2_si(train_bags_cat,
                                           train_labels_cat,
                                       sparse_input=True)
si_X_test_cat, si_y_test_cat = bags_2_si(train_bags_cat,
                                         train_labels_cat,
                                         sparse_input=True)

def _densify_bags(X : Union[np.ndarray, csr_matrix]) -> np.ndarray:
    """Convert a Numpy array of sparse bags into an array of dense bags
    inputs
    -------
    X: (scipy.sparse.csr_matrix) of shape (n) where n is the total number of 
        bags in the dataset. Each entry of X is of shape 
        (n_instances, n_features) where n_instances is the number of instances
        within a bag, and n_features is the features space of instances.
        n_instances can vary per bag
    outputs
    -------
    dense_bags: (np.ndaray, dtype='object') of shape (n) where n is the total 
        number of bags in the dataset. Each object is a dense numpy array
        of shape (n_instances, n_features). n_instances can vary per bag"""
        
    if not isinstance(X[0], csr_matrix):
        msg="Input must be of type scipy.sparse.csr_matrix. Got {}".format(type(X))
        raise ValueError(msg)
    if X.ndim != 1:
        msg="Input must have single outer dimension. Got {} dims".format(X.ndim)
        raise ValueError(msg)
    
    # Convert sparse bags to dense bags
    n_bags = X.shape[0]
    dense_bags = np.empty(n_bags, dtype='object')
    for n in range(n_bags):
        dense_bags[n] = X[n].toarray()
        
    return dense_bags


def _filter_bags_by_size(X:Union[np.ndarray, csr_matrix, list],
                        min_instances:int,
                        max_instances:int) -> np.ndarray:
    """Filter a set of bags by number of instances within the bag. If the 
    bag contains less than n_instances, then do not include that bag in the 
    returned index
    inputs
    -------
    X: (np.ndarray or iterable) of bags
    outputs
    --------
    index: (np.ndarray) index indicating where the number of instances per bag 
        is within the criteria.
    """
    
    # Store bags in linked list
    bags = []
    index = []
    
    # Iterate through bags and add to list
    n_bags = X.shape[0]
    for n in range(n_bags):
        if X[n].shape[0] < min_instances:
            continue
        elif X[n].shape[0] > max_instances:
            continue
        else:
            bags.append(X[n])
            index.append(n)
            
    return np.array(index, dtype=np.int16)
    



#%% Estimators

# K-NN
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform',
                           algorithm='ball_tree', n_jobs=4)

# Multinomial Native Bayes
multiNB = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

# CommplementNB - Like multinomial but for imbalanced datasets
compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

# Define scorers
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score, average='weighted'),
           'balanced_accuracy':'balanced_accuracy',
           }

# CRoss-validation using of instances within bags by broadcasting the bag label
# To all instances within a bag
res_knn_si = cross_validate(estimator=knn, 
                          X=si_X_train, 
                          y=si_y_train, 
                          cv=3, 
                          scoring=scoring,
                          )
res_multinomial_si = cross_validate(estimator=multiNB, 
                          X=si_X_train_cat, 
                          y=si_y_train_cat, 
                          cv=3, 
                          scoring=scoring,
                          )
res_comNB_si = cross_validate(estimator=compNB, 
                          X=si_X_train_cat, 
                          y=si_y_train_cat, 
                          cv=3, 
                          scoring=scoring,
                          )


#%% Predict on bags using most common label assigned to instances
# AKA Single-instance inference

# Initial Values
CV = 3
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
results = {}

# Define a scorer and Metrics
scorer = {'precision_weighted':make_scorer(precision_score, average='weighted'),
          'recall_weighted':make_scorer(recall_score, average='weighted'),
          'accuracy':make_scorer(accuracy_score),
          'accuracy_balanced':make_scorer(balanced_accuracy_score),
          }
accuracy = []
accuracy_balanced = []
precision = []
recall = []

# Define an estimator
ESTIMATOR = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

# Load raw datasets, Already loaded above
BAGS = train_bags_cat
BAG_LABELS = train_labels_cat

# Split bags into training and validation sets
rs = ShuffleSplit(n_splits=CV, test_size=TEST_SIZE, train_size=TRAIN_SIZE)
for train_index, test_index in rs.split(BAGS, BAG_LABELS):

    # Split bags
    _x_train_bags, _y_train_bags = BAGS[train_index], BAG_LABELS[train_index]
    _x_test_bags, _y_test_bags = BAGS[test_index], BAG_LABELS[test_index]

    # Convert training set to single instance to fit the estimator
    _x_train_si, _y_train_si = bags_2_si(_x_train_bags,
                                         _y_train_bags,
                                         sparse_input=True)
    
    # Fit an estimator on SI data
    ESTIMATOR.fit(_x_train_si, _y_train_si)
    
    # Predict instances in a bag
    bag_predictions = BagScorer.predict_bags(ESTIMATOR, _x_test_bags, method='mode')
        
    # Estimate metrics on bags
    accuracy.append(scorer['accuracy']\
                    ._score_func(_y_test_bags.reshape(-1,1), 
                                 bag_predictions, 
                                 **(scorer['accuracy']._kwargs)))
    accuracy_balanced.append(scorer['accuracy_balanced']\
                             ._score_func(_y_test_bags, 
                                          bag_predictions,
                                          **(scorer['accuracy_balanced']._kwargs)))
    precision.append(scorer['precision_weighted']
                     ._score_func(_y_test_bags, 
                                  bag_predictions,
                                  **(scorer['precision_weighted']._kwargs)))
    recall.append(scorer['recall_weighted']\
                  ._score_func(_y_test_bags, 
                               bag_predictions,
                               **(scorer['recall_weighted']._kwargs)))



#%% Predict on bags using cross validation with single-instance inference

# Create estimators
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform',
                           algorithm='ball_tree', n_jobs=4)

# Multinomial Native Bayes
multiNB = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

# CommplementNB - Like multinomial but for imbalanced datasets
compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

# Define evaluation metrics
accuracy_scorer = make_scorer(accuracy_score)
bagAccScorer = BagScorer(accuracy_scorer, sparse_input=False) # Accuracy score, no factory function
precision_scorer = make_scorer(precision_score, average='weighted')
bagPreScorer = BagScorer(precision_scorer, sparse_input=False)
recall_scorer = make_scorer(recall_score, average='weighted')
bagRecScorer = BagScorer(recall_scorer, sparse_input=False)

scoring_dense = {'bag_accuracy':bagAccScorer,
           'bag_precision':bagPreScorer,
           'bag_recall':bagRecScorer,
           }

# Convert bags to dense for KNN estimator
train_bags_dense = _densify_bags(train_bags)
train_bags_cat_filter = _filter_bags_by_size(train_bags_cat, 
                                             min_instances=5,
                                             max_instances=1000)

# Cross validate bags
res_knn_infer = cross_validate_bag(estimator=knn, 
                            X=train_bags_dense[:100], # TODO Test whole training set
                            y=train_bag_labels[:100], 
                            groups=None, 
                            scoring=scoring_dense, # Custom scorer... 
                            cv=2,
                            n_jobs=4, 
                            verbose=0, 
                            fit_params=None,
                            pre_dispatch='2*n_jobs', 
                            return_train_score=False,
                            return_estimator=False, 
                            error_score=np.nan)

# Multinomial native bayes supports sparse features...
accuracy_scorer = make_scorer(accuracy_score)
bagAccScorer = BagScorer(accuracy_scorer, sparse_input=True) # Accuracy score, no factory function
precision_scorer = make_scorer(precision_score, average='weighted')
bagPreScorer = BagScorer(precision_scorer, sparse_input=True)
recall_scorer = make_scorer(recall_score, average='weighted')
bagRecScorer = BagScorer(recall_scorer, sparse_input=True)
scoring_sparse = {'bag_accuracy':bagAccScorer,
           'bag_precision':bagPreScorer,
           'bag_recall':bagRecScorer,
           }

res_multinomial_infer = cross_validate_bag(estimator=multiNB, 
                            X=train_bags_cat[:100],  # TODO Test whole training set
                            y=train_labels_cat[:100], 
                            groups=None, 
                            scoring=scoring_sparse, # Custom scorer... 
                            cv=2,
                            n_jobs=4, 
                            verbose=0, 
                            fit_params=None,
                            pre_dispatch='2*n_jobs', 
                            return_train_score=False,
                            return_estimator=False, 
                            error_score=np.nan)

res_comNB_infer = cross_validate_bag(estimator=multiNB, 
                            X=train_bags_cat[:100], # TODO Test whole training set
                            y=train_labels_cat[:100], 
                            groups=None, 
                            scoring=scoring_sparse, # Custom scorer... 
                            cv=2,
                            n_jobs=4, 
                            verbose=0, 
                            fit_params=None,
                            pre_dispatch='2*n_jobs', 
                            return_train_score=False,
                            return_estimator=False, 
                            error_score=np.nan)
