# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 09:02:37 2020

#TODO - Attempt more models
#TODO - Attempt ensemble methods
#TODO - Attempt hyperparameter optimization
#TODO - Feature selection
#TODO - Confidence intervals on accuracy estimates (including confidence 
    interval on any other evaluation metric)
#TODO - Is it a problem that I have a significant number of features compared
    to the number of instances? Should I use feature selection methods?
#TODO - "Second, 5-fold cross-validation is not sufficiently precise. It may 
    be necessary to repeat it 100 times to achieve adequate precision. Third, you 
    have chosen as an accuracy score a discontinuous improper scoring rule 
    (proportion classified correctly). Such an improper scoring rule will lead to 
    selection of the wrong model."
    From https://stats.stackexchange.com/questions/59630/test-accuracy-higher-than-training-how-to-interpret
    
    1. Repeat cross-validation a number of times in order to get a confidence 
    interval on elvaluation metrics
    2. What is the most proper evaluation metric in order to choose a good
    estimator? Accuracy may not be preferable due to inbalanced data labels

@author: z003vrzk
"""

# Python imports
import sys
import os
import configparser
from typing import Union, Dict
import pickle

# Third party imports
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.metrics import (make_scorer, precision_score,
                             recall_score, accuracy_score, 
                             balanced_accuracy_score, confusion_matrix)
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

from mil_load import LoadMIL, load_mil_dataset_from_file
from bag_cross_validate import cross_validate_bag, BagScorer, bags_2_si

# Global declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

loadMIL = LoadMIL(server_name, driver_name, database_name)

#%% Load data
# Load data from DB and run through pipeline
dataset_numeric = {'dataset':None,'bag_labels':None}
dataset_numeric['dataset'], dataset_numeric['bag_labels'] = \
    loadMIL.gather_mil_dataset(pipeline='whole')
dataset_categorical = {'dataset':None,'bag_labels':None}
dataset_categorical['dataset'], dataset_categorical['bag_labels'] = \
    loadMIL.gather_mil_dataset(pipeline='categorical')
    
# Load numeric dataset from file
# numeric_feature_file = r'../data/MIL_dataset.dat'
# dataset_numeric = load_mil_dataset_from_file(numeric_feature_file)
# Load categorical dataset from file
# categorical_feature_file = r'../data/MIL_cat_dataset.dat'
# dataset_categorical = load_mil_dataset_from_file(categorical_feature_file)

# Unpack loaded data
bags = dataset_numeric['dataset']
bag_labels = dataset_numeric['bag_labels']
cat_bags = dataset_categorical['dataset']
cat_bag_labels = dataset_categorical['bag_labels']

# Split numeric dataset
rs = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
train_index, test_index = next(rs.split(bags, bag_labels))
train_bags, train_labels = bags[train_index], bag_labels[train_index]
test_bags, test_labels = bags[test_index], bag_labels[test_index]

# Split categorical dataset
train_bags_cat, train_labels_cat = cat_bags[train_index], cat_bag_labels[train_index]
test_bags_cat, test_labels_cat = cat_bags[test_index], cat_bag_labels[test_index]

# Unpack bags into single instances for training and testing
# Bags to single instances
si_X_train, si_y_train = bags_2_si(train_bags,
                                   train_labels,
                                   sparse_input=True)
si_X_test, si_y_test = bags_2_si(test_bags,
                                 test_labels,
                                 sparse_input=True)

# Unpack categorical
si_X_train_cat, si_y_train_cat = bags_2_si(train_bags_cat,
                                           train_labels_cat,
                                           sparse_input=True)
si_X_test_cat, si_y_test_cat = bags_2_si(test_bags_cat,
                                         test_labels_cat,
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
    

def _print_results_dict(res:Union[Dict[str,list], Dict[str,float]],
                        msg=None) -> None:
    if msg:
        print(msg)
    
    if hasattr(res, 'items'):
        for key, value in res.items():
            print(key, " : ", value)
        
    print("\n\n")
    return None


def pickle_save_to_file(save_path: str, classifier: object) -> None:
    
    if os.path.isfile(save_path):
        msg="You are attenpting to overwrite an existing file at {}\n"\
            .format(save_path)
        msg+="Input Y/y to continue"
        res = input(msg)
        
        if res in ['y','Y']:
            with open(save_path, mode='wb') as f:
                pickle.dump(classifier, f)
            
    else:
        with open(save_path, mode='wb') as f:
            pickle.dump(classifier, f)
            
    return None

#%% Estimators
"""Define several estimators
Evaluate the estimators on the single-instance Multi-Class classification 
problem"""

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

# Print results of single-instance classification problem
_print_results_dict(res_knn_si, 
                    ("Single-instance Multi-Class classification results of " +
                     "KNN Estimator. " +
                     "The instance labels are derived from bag labels.:\n"))

_print_results_dict(res_multinomial_si, 
                    ("Single-instance Multi-Class classification results of " +
                     "Multinomial Native Bayes Estimator. " +
                     "The instance labels are derived from bag labels.:\n"))

_print_results_dict(res_comNB_si, 
                    ("Single-instance Multi-Class classification results of " +
                     "Complement Native Bayes Estimator. " +
                     "The instance labels are derived from bag labels.:\n"))


#%% 
"""Predict on bags using most frequent label assigned to instances
AKA Single-instance inference of a bag label

This section was typed before I made the BagScorer"""

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



#%% 
"""Predict on bags using cross validation with single-instance inference"""

# Create estimators
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform',
                           algorithm='ball_tree', n_jobs=4)

# Multinomial Native Bayes
multiNB = MultinomialNB(alpha=0.5, fit_prior=True, class_prior=None)

# CommplementNB - Like multinomial but for imbalanced datasets
compNB = ComplementNB(alpha=0.5, fit_prior=True, class_prior=None, norm=False)

# SVC - Linear L1 regularized
svmc_l1 = LinearSVC(loss='squared_hinge', penalty='l1', C=5, 
                            dual=False, max_iter=2500)

# SVC Using LibSVM uses the squared l2 loss
svmc = SVC(kernel='rbf', gamma='scale', C=5)

# Filter out bags with only a single instance
_filter = _filter_bags_by_size(train_bags_cat, 
                               min_instances=5,
                               max_instances=1000)

# Convert bags to dense for KNN estimator
_train_bags_dense = _densify_bags(train_bags[_filter])
_train_labels = train_labels[_filter]
# Keep bags sparse for Complement Native Bayes and Multinomial
_train_bags_cat = train_bags_cat[_filter]
_train_labels_cat = train_labels_cat[_filter]

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

# Cross validate bags
res_knn_cv = cross_validate_bag(
    estimator=knn, 
    X=_train_bags_dense, 
    y=_train_labels, 
    groups=None, 
    scoring=scoring_dense, # Custom scorer... 
    cv=2,
    n_jobs=4, 
    verbose=0, 
    fit_params=None,
    pre_dispatch='2*n_jobs', 
    return_train_score=False,
    return_estimator=True, 
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

res_multiNB_cv = cross_validate_bag(
    estimator=multiNB, 
    X=_train_bags_cat,
    y=_train_labels_cat, 
    groups=None, 
    scoring=scoring_sparse, # Custom scorer... 
    cv=2,
    n_jobs=4, 
    verbose=0, 
    fit_params=None,
    pre_dispatch='2*n_jobs', 
    return_train_score=False,
    return_estimator=True, 
    error_score=np.nan)

res_compNB_cv = cross_validate_bag(
    estimator=compNB, 
    X=_train_bags_cat,
    y=_train_labels_cat, 
    groups=None, 
    scoring=scoring_sparse, # Custom scorer... 
    cv=2,
    n_jobs=4, 
    verbose=0, 
    fit_params=None,
    pre_dispatch='2*n_jobs', 
    return_train_score=False,
    return_estimator=True, 
    error_score=np.nan)

res_svmc_l1_cv = cross_validate_bag(
    estimator=svmc_l1, 
    X=_train_bags_cat,
    y=_train_labels_cat, 
    groups=None, 
    scoring=scoring_sparse, # Custom scorer... 
    cv=2,
    n_jobs=4, 
    verbose=0, 
    fit_params=None,
    pre_dispatch='2*n_jobs', 
    return_train_score=False,
    return_estimator=True, 
    error_score=np.nan)

res_svmc_cv = cross_validate_bag(
    estimator=svmc, 
    X=_train_bags_cat,
    y=_train_labels_cat, 
    groups=None, 
    scoring=scoring_sparse, # Custom scorer... 
    cv=2,
    n_jobs=4, 
    verbose=0, 
    fit_params=None,
    pre_dispatch='2*n_jobs', 
    return_train_score=False,
    return_estimator=True, 
    error_score=np.nan)

_print_results_dict(res_knn_cv, 
                    ("Cross validation results of KNN Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_multiNB_cv, 
                    ("Cross validation results of multinomial NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_compNB_cv, 
                    ("Cross validation results of Complement NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_svmc_l1_cv, 
                    ("Cross validation results of Linear SVM Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_svmc_cv, 
                    ("Cross validation results of RBF SVM Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))




#%% 
"""Perform predictions on test set | final evaluation of model performance"""

# Define the training and testing data set
# Use stratified folding to preserve class imbalance
# Filter out bags with only a single instance
_filter_train = _filter_bags_by_size(train_bags, 
                                     min_instances=5,
                                     max_instances=2000)
_filter_test = _filter_bags_by_size(test_bags,
                                    min_instances=5,
                                    max_instances=2000)
_filter_train_cat = _filter_bags_by_size(train_bags_cat, 
                                     min_instances=5,
                                     max_instances=2000)
_filter_test_cat = _filter_bags_by_size(test_bags_cat,
                                    min_instances=5,
                                    max_instances=2000)

# Convert bags to dense for KNN estimator
_train_bags_dense = _densify_bags(train_bags[_filter_train])
_train_labels = train_labels[_filter_train]
_test_bags_dense = _densify_bags(test_bags[_filter_test])
_test_labels = test_labels[_filter_test]
# Keep bags sparse for Complement Native Bayes and Multinomial
_train_bags_cat = train_bags_cat[_filter_train_cat]
_train_labels_cat = train_labels_cat[_filter_train_cat]
_test_bags_cat = test_bags_cat[_filter_test_cat]
_test_labels_cat = test_labels_cat[_filter_test_cat]

# Define hyperparameters
_knn_nneighbors = 3 # Best result of cross validation
_multinb_alpha = 0.5 # Best result of cross validation
_compnb_alpha = 0.9 # Best result of cross validation
_svm_c = 5 # Best result of cross validation

# Define estimators
knn = KNeighborsClassifier(n_neighbors=_knn_nneighbors, weights='uniform',
                           algorithm='ball_tree', n_jobs=4)

# Multinomial Native Bayes
multiNB = MultinomialNB(alpha=_multinb_alpha, fit_prior=True, class_prior=None)

# CommplementNB - Like multinomial but for imbalanced datasets
compNB = ComplementNB(alpha=_compnb_alpha, fit_prior=True, class_prior=None, norm=False)

# SVC - Linear L1 regularized
svmc_l1 = LinearSVC(loss='squared_hinge', penalty='l1', C=_svm_c, 
                            dual=False, max_iter=2500)

# SVC Using LibSVM uses the squared l2 loss
svmc = SVC(kernel='rbf', gamma='scale', C=_svm_c)

# Define scoring metrics
_bagKnnScorer = {
    'bag_accuracy':BagScorer(make_scorer(accuracy_score), 
                             sparse_input=False) ,
    'bag_precision':BagScorer(make_scorer(precision_score, average='micro'), 
                              sparse_input=False),
    'bag_recall':BagScorer(make_scorer(recall_score, average='micro'), 
                           sparse_input=False),
    }

_bagMultiNBScorer = {    
    'bag_accuracy':BagScorer(make_scorer(accuracy_score), 
                             sparse_input=True) ,
    'bag_precision':BagScorer(make_scorer(precision_score, average='micro'), 
                              sparse_input=True),
    'bag_recall':BagScorer(make_scorer(recall_score, average='micro'), 
                           sparse_input=True),
    }

_bagCompNBScorer = {    
    'bag_accuracy':BagScorer(make_scorer(accuracy_score), 
                             sparse_input=True) ,
    'bag_precision':BagScorer(make_scorer(precision_score, average='micro'), 
                              sparse_input=True),
    'bag_recall':BagScorer(make_scorer(recall_score, average='micro'), 
                           sparse_input=True),
    }

_bagsvmcl1Scorer = {    
    'bag_accuracy':BagScorer(make_scorer(accuracy_score), 
                             sparse_input=False) ,
    'bag_precision':BagScorer(make_scorer(precision_score, average='micro'), 
                              sparse_input=False),
    'bag_recall':BagScorer(make_scorer(recall_score, average='micro'), 
                           sparse_input=False),
    }

_bagsvmcScorer = {    
    'bag_accuracy':BagScorer(make_scorer(accuracy_score), 
                             sparse_input=False) ,
    'bag_precision':BagScorer(make_scorer(precision_score, average='micro'), 
                              sparse_input=False),
    'bag_recall':BagScorer(make_scorer(recall_score, average='micro'), 
                           sparse_input=False),
    }


# Fit the estimator
knn = _bagKnnScorer['bag_accuracy'].estimator_fit(knn, 
                                                  _train_bags_dense, 
                                                  _train_labels)
multiNB = _bagMultiNBScorer['bag_accuracy'].estimator_fit(multiNB, 
                                             _train_bags_cat, 
                                             _train_labels_cat)
compNB = _bagCompNBScorer['bag_accuracy'].estimator_fit(compNB, 
                                            _train_bags_cat, 
                                            _train_labels_cat)
svmc_l1 = _bagsvmcl1Scorer['bag_accuracy'].estimator_fit(svmc_l1, 
                                                      _train_bags_dense, 
                                                      _train_labels)
svmc = _bagsvmcScorer['bag_accuracy'].estimator_fit(svmc, 
                                                   _train_bags_dense, 
                                                   _train_labels)

# Predict on the validation set
yhat_knn = _bagKnnScorer['bag_accuracy'].predict_bags(knn, _test_bags_dense)
yhat_multiNB = _bagMultiNBScorer['bag_accuracy'].predict_bags(multiNB, _test_bags_cat)
yhat_compNB = _bagCompNBScorer['bag_accuracy'].predict_bags(compNB, _test_bags_cat)
yhat_svmc_l1 = _bagKnnScorer['bag_accuracy'].predict_bags(svmc_l1, _test_bags_dense)
yhat_svmc = _bagKnnScorer['bag_accuracy'].predict_bags(svmc, _test_bags_dense)

# Predict on training set
yhat_knn_train = _bagKnnScorer['bag_accuracy'].predict_bags(knn, _train_bags_dense)
yhat_multiNB_train = _bagMultiNBScorer['bag_accuracy'].predict_bags(multiNB, _train_bags_cat)
yhat_compNB_train = _bagCompNBScorer['bag_accuracy'].predict_bags(compNB, _train_bags_cat)
yhat_svmc_l1_train = _bagCompNBScorer['bag_accuracy'].predict_bags(svmc_l1, _train_bags_dense)
yhat_svmc_train = _bagCompNBScorer['bag_accuracy'].predict_bags(svmc, _train_bags_dense)

# Calculate evaluation metrics (Final validation test set)
res_knn_infer_test = {
    'bag_accuracy':_bagKnnScorer['bag_accuracy'](knn, _test_bags_dense, _test_labels),
    'bag_precision':_bagKnnScorer['bag_precision'](knn, _test_bags_dense, _test_labels),
    'bag_recall':_bagKnnScorer['bag_recall'](knn, _test_bags_dense, _test_labels),
    }

res_multiNB_infer_test = {
    'bag_accuracy':_bagMultiNBScorer['bag_accuracy'](multiNB, _test_bags_cat, _test_labels_cat),
    'bag_precision':_bagMultiNBScorer['bag_precision'](multiNB, _test_bags_cat, _test_labels_cat),
    'bag_recall':_bagMultiNBScorer['bag_recall'](multiNB, _test_bags_cat, _test_labels_cat),
    }

res_compNB_infer_test = {
    'bag_accuracy':_bagCompNBScorer['bag_accuracy'](compNB, _test_bags_cat, _test_labels_cat),
    'bag_precision':_bagCompNBScorer['bag_precision'](compNB, _test_bags_cat, _test_labels_cat),
    'bag_recall':_bagCompNBScorer['bag_recall'](compNB, _test_bags_cat, _test_labels_cat),
    }

res_smvc_l1_infer_test = {
    'bag_accuracy':_bagsvmcl1Scorer['bag_accuracy'](svmc_l1, _test_bags_dense, _test_labels),
    'bag_precision':_bagsvmcl1Scorer['bag_precision'](svmc_l1, _test_bags_dense, _test_labels),
    'bag_recall':_bagsvmcl1Scorer['bag_recall'](svmc_l1, _test_bags_dense, _test_labels),
    }

res_smvc_infer_test = {
    'bag_accuracy':_bagsvmcScorer['bag_accuracy'](svmc, _test_bags_dense, _test_labels),
    'bag_precision':_bagsvmcScorer['bag_precision'](svmc, _test_bags_dense, _test_labels),
    'bag_recall':_bagsvmcScorer['bag_recall'](svmc, _test_bags_dense, _test_labels),
    }


# Calculate evaluation metrics (Training set)
res_knn_infer_train = {
    'bag_accuracy':_bagKnnScorer['bag_accuracy'](knn, _train_bags_dense, _train_labels),
    'bag_precision':_bagKnnScorer['bag_precision'](knn, _train_bags_dense, _train_labels),
    'bag_recall':_bagKnnScorer['bag_recall'](knn, _train_bags_dense, _train_labels),
    }

res_multiNB_infer_train = {
    'bag_accuracy':_bagMultiNBScorer['bag_accuracy'](multiNB, _train_bags_cat, _train_labels_cat),
    'bag_precision':_bagMultiNBScorer['bag_precision'](multiNB, _train_bags_cat, _train_labels_cat),
    'bag_recall':_bagMultiNBScorer['bag_recall'](multiNB, _train_bags_cat, _train_labels_cat),
    }

res_compNB_infer_train = {
    'bag_accuracy':_bagCompNBScorer['bag_accuracy'](compNB, _train_bags_cat, _train_labels_cat),
    'bag_precision':_bagCompNBScorer['bag_precision'](compNB, _train_bags_cat, _train_labels_cat),
    'bag_recall':_bagCompNBScorer['bag_recall'](compNB, _train_bags_cat, _train_labels_cat),
    }

res_smvc_l1_infer_train = {
    'bag_accuracy':_bagsvmcl1Scorer['bag_accuracy'](svmc_l1, _train_bags_dense, _train_labels),
    'bag_precision':_bagsvmcl1Scorer['bag_precision'](svmc_l1, _train_bags_dense, _train_labels),
    'bag_recall':_bagsvmcl1Scorer['bag_recall'](svmc_l1, _train_bags_dense, _train_labels),
    }

res_smvc_infer_train = {
    'bag_accuracy':_bagsvmcScorer['bag_accuracy'](svmc, _train_bags_dense, _train_labels),
    'bag_precision':_bagsvmcScorer['bag_precision'](svmc, _train_bags_dense, _train_labels),
    'bag_recall':_bagsvmcScorer['bag_recall'](svmc, _train_bags_dense, _train_labels),
    }



# Print information about the data set
msg = ("The {} Estimator was trained on {} traing bags. Final evaluation was " +
       "performed using {} testing bags\n\n")
print(msg.format("KNN", 
                 _train_bags_dense.shape[0], 
                 _test_bags_dense.shape[0]))
print(msg.format("Multinomial Native Bayes", 
                 _train_bags_cat.shape[0], 
                 _test_bags_cat.shape[0]))
print(msg.format("Complement Native Bayes", 
                 _train_bags_cat.shape[0], 
                 _test_bags_cat.shape[0]))

msg = ("A total of {} {} bags were excluded becase they did not contain " +
       "greater than 5 instances\n")
print(msg.format(train_bags.shape[0] - _filter_train.shape[0], "training"))
print(msg.format(test_bags.shape[0] - _filter_test.shape[0], "test"))


# Print test results
_print_results_dict(res_knn_infer_test, 
                    ("Final evaluation results of KNN Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_multiNB_infer_test, 
                    ("Final evaluation results of multinomial NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_compNB_infer_test, 
                    ("Final evaluation results of Complement NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_smvc_l1_infer_test, 
                    ("Final evaluation results of SVMC L1 Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_smvc_infer_test, 
                    ("Final evaluation results of SVMC RBF L2 Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))


# Print training results
_print_results_dict(res_knn_infer_train, 
                    ("Training evaluation of KNN Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_multiNB_infer_train, 
                    ("Training evaluation results of multinomial NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_compNB_infer_train, 
                    ("Training evaluation results of Complement NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_smvc_l1_infer_train, 
                    ("Training evaluation results of SVMC L1 Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_smvc_infer_train, 
                    ("Training evaluation results of SVMC RBF L2 Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))


# Confusion matrix
res_knn_infer_test['confusion'] = confusion_matrix(_test_labels, yhat_knn)
res_multiNB_infer_test['confusion'] = confusion_matrix(_test_labels, yhat_multiNB)
res_compNB_infer_test['confusion'] = confusion_matrix(_test_labels, yhat_compNB)
res_smvc_l1_infer_test['confusion'] = confusion_matrix(_test_labels, yhat_svmc_l1)
res_smvc_infer_test['confusion'] = confusion_matrix(_test_labels, yhat_svmc)
set(sorted(_test_labels))


#%% Save the estimator

# Save all estimators
pickle_save_to_file('./knn_si.clf', knn)
pickle_save_to_file('./multiNB_si.clf', multiNB)
pickle_save_to_file('./compNB_si.clf', compNB)
pickle_save_to_file('./svmc_l1_si.clf', svmc_l1)
pickle_save_to_file('./svmc_rbf_si.clf', svmc)
    
# Save training and validation data
pickle_save_to_file('./data_sidense_trian.dat', (_train_bags_dense, _train_labels))
pickle_save_to_file('./data_sidense_test.dat', (_test_bags_dense, _test_labels))
pickle_save_to_file('./data_sicat_trian.dat', (_train_bags_cat, _train_labels_cat))
pickle_save_to_file('./data_sicat_test.dat', (_test_bags_cat, _test_labels_cat))

    