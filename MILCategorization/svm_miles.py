# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:07:07 2021

@author: vorst
"""

# Python imports
import configparser
import os, sys
from typing import Union
import pickle

# Third party imports
from sklearn.model_selection import (train_test_split, StratifiedShuffleSplit, 
                                     GridSearchCV)
import sklearn as skl
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             balanced_accuracy_score)
from sklearn.metrics import (accuracy_score, recall_score, 
                             make_scorer, precision_score,
                             balanced_accuracy_score)
from scipy.sparse import csr_matrix
import numpy as np
from pyMILES.embedding import embed_all_bags

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from mil_load import load_mil_dataset, LoadMIL, bags_2_si, bags_2_si_generator

# Global declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

LoadMIL = LoadMIL(server_name, driver_name, database_name)


#%%

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


def _print_results_dict(res:Union[dict[str,list], dict[str,float]],
                        msg=None) -> None:
    if msg:
        print(msg)
    
    for key, value in res.items():
        print(key, " : ", value)
        
    print("\n\n")
    return None

#%%
"""Load data"""

# Load data from DB and run through pipeline
# bags, labels = LoadMIL.gather_mil_dataset(pipeline='whole')

# Load data from saved file
dataset_numeric = load_mil_dataset(numeric_feature_file)
dataset_categorical = load_mil_dataset(categorical_feature_file)


# Split data into training and testing sets
"""
# TODO: In the ideal scenario I will hand-pick the most optimal training 
bags. These will most accurately represent the categories that I 
want to identify

Note: Why is the proportion of training instances so low (15%)? It is because
the training instances are used as the concept class.
Instances in the concept class are used to embed the testing set onto a 
feature space, which the SVM then predicts on"""
train_bags_sp, test_bags_sp, train_labels, test_labels = train_test_split(
    dataset_numeric['dataset'], 
    dataset_numeric['bag_labels'],
    train_size=0.15)

# Create dataset for embedding bags into similarity of concept class
ss = StratifiedShuffleSplit(n_splits=1, train_size=0.15)
_concept_index, _train_test_index = next(ss.split(
    dataset_numeric['dataset'],
    dataset_numeric['bag_labels'])
    )
concept_bags = dataset_numeric['dataset'][_concept_index]
concept_labels = dataset_numeric['bag_labels'][_concept_index]
_train_test_bags = dataset_numeric['dataset'][_train_test_index]
_train_test_labels = dataset_numeric['bag_labels'][_train_test_index]

# Create training and validation sets
ss = StratifiedShuffleSplit(n_splits=1, train_size=0.5)
_train_index, _test_index = next(ss.split(
    _train_test_bags, _train_test_labels)
    )

train_bags = _train_test_bags[_train_index]
train_labels = _train_test_labels[_train_index]
test_bags = _train_test_bags[_test_index]
test_labels = _train_test_labels[_test_index]

# Generate dense bags
"""bags: (np.ndarray) shape (i,j,p) where n is the number of bags, j is the 
    number of instances per bag, and p is the feature space per instance 
    in a bag"""
concept_bags = _densify_bags(concept_bags)
train_bags = _densify_bags(train_bags)
test_bags = _densify_bags(test_bags)

"""Create a concept class, the set of all training instances from positive and 
negative bags C = {x^k : k=1, ..., n}
Where x^k is the kth instance in the entire training set
"""
# Create concept class from training instances
C_features, C_labels = bags_2_si(concept_bags, concept_labels)

# Number of training instances
# Number of testing instances
N_TRAIN_INSTANCES = 0
N_TEST_INSTANCES = 0
for x in train_bags:
    N_TRAIN_INSTANCES += x.shape[0]
for x in test_bags:
    N_TEST_INSTANCES += x.shape[0]

print("There are {} single instances in the training set\n".format(N_TRAIN_INSTANCES))
print("There are {} single instances in the testing set\n".format(N_TEST_INSTANCES))

"""Encode data into similarity matrix
NOTICE - testing and training bags are embedded onto the concept class set
This means that the 'testing' set contains a similarity measure to the training
instances. 
I will perform cross validation on the training instances and final evaluation 
on testing instances
embedded_train and test are (j,n) arrays wher each bag is encoded into a 
feature vector which represents a similarity measure between the bag and 
concept class. j is the number of instances in the concept class, and n is the 
number of bags in the test/train set.
A single embedded bag is stored along axis=0, so a single
embedded bag can be sliced like embedded_bag = embedded_set[:,0].
The similarity between the nth bag, and jth concept instance can be found like
 embedded_set[j,n]
For this reason, the estimator will be fed instances as np.transpose(embedded_bags)
because it expects features to be alligned along axis=1"""
embedded_train = embed_all_bags(concept_class=C_features, 
                                bags=[x for x in train_bags],
                                sigma=3,
                                distance='euclidean')
embedded_test = embed_all_bags(concept_class=C_features, 
                                bags=[x for x in test_bags],
                                sigma=3,
                                distance='euclidean')


#%%
"""Perform cross-validation on embedded dataset with two estimators:
1. L1 SVM (regularization SVM with L1 distance)
2. SVM with radial-basis-function distance kernel

Use a custom
"""
GAMMA_SVC = 'scale'
PENALTY = 'l1' # L1 loss penalization
LOSS = 'squared_hinge' # Loss function
C = 1.0 # SVM regularization, inversely proportional

# Define SVM
svmc_l1 = LinearSVC(loss=LOSS, penalty=PENALTY, C=C, 
                            dual=False, max_iter=2500)

# SVC Using LibSVM uses the squared l2 loss
svmc = SVC(kernel='rbf', gamma=GAMMA_SVC, C=C)

# Define grid search parameters
params_l1svc = {'C':[0.5,2,5],
                }
params_svc = {'C':[0.5,2,5,10],
              'kernel':['rbf', 'poly']}

# Define scorers
scoring = {'accuracy':make_scorer(accuracy_score),
           'balanced-accuracy':make_scorer(balanced_accuracy_score),
           'precision':make_scorer(precision_score, average='micro'),
           'recall':make_scorer(recall_score, average='micro'),
            }


# Grid search
svmc_l1_gs = GridSearchCV(
    estimator=svmc_l1,
    param_grid=params_l1svc,
    scoring=scoring,
    n_jobs=6,
    cv=3, # Default 5-fold validation
    refit='accuracy',
    )


svmc_gs = GridSearchCV(
    estimator=svmc,
    param_grid=params_svc,
    scoring=scoring,
    n_jobs=6,
    cv=3, # Default 5-fold validation
    refit='accuracy',
    )

# Filter training and testing instances so that only bags with >= 5 instances
# Are included in grid search
_filter_train = _filter_bags_by_size(train_bags, 
                                      min_instances=5, 
                                      max_instances=2000)
_filter_test = _filter_bags_by_size(test_bags, 
                                    min_instances=5, 
                                    max_instances=2000)
train_embed_filter = embedded_train[:,_filter_train]
train_labels_filter = train_labels[_filter_train]
test_embed_filter = embedded_test[:,_filter_test]
test_labels_filter = test_labels[_filter_test]

# Perform cross validation - REMEMBER that we are performing cross validation
# On the TESTING set this time because the testing bags were embedded using
# the training bags as the concept class
# The estimator will be fed instances as np.transpose(embedded_bags)
# because it expects features to be alligned along axis=1
svmc_l1_gs.fit(np.transpose(train_embed_filter), train_labels_filter)
svmc_gs.fit(np.transpose(train_embed_filter), train_labels_filter)

# Print results of cross-validation using MILES embedding w/ SVM estimator
_print_results_dict(svmc_l1_gs.cv_results_, 
                    ("MILES embedding Multi-Class classification results of " +
                      "SVM L1 Estimator. " +
                      "Bags are embedded into a feature vector:\n"))

_print_results_dict(svmc_gs.cv_results_, 
                    ("MILES embedding Multi-Class classification results of " +
                      "SVM Estimator w/ RBF kernel. " +
                      "Bags are embedded into a feature vector:\n"))


#%%
"""
Final model evaluation on testing data set

Use the preferred model to calculate evaluation metrics on training and
testing instances
"""

# Filter training and testing instances so that only bags with >= 5 instances
# Are included in grid search
_filter_train = _filter_bags_by_size(train_bags, 
                                      min_instances=5, 
                                      max_instances=2000)
_filter_test = _filter_bags_by_size(test_bags, 
                                    min_instances=5, 
                                    max_instances=2000)
train_embed_filter = embedded_train[:,_filter_train]
train_labels_filter = train_labels[_filter_train]
test_embed_filter = embedded_test[:,_filter_test]
test_labels_filter = test_labels[_filter_test]

# Define estimators (Choose from cross-validation)
svmc_l1_best = svmc_l1_gs.best_estimator_
svmc_best = svmc_gs.best_estimator_

# Fit the estimator
# if not check_is_fitted(svmc_l1_best):
#     svmc_l1_best.fit(np.transpose(train_embed_filter), train_labels_filter)
# if not check_is_fitted(smvc_best):
#     smvc_best.fit(np.transpose(train_embed_filter), train_labels_filter)

# Predict on the validation set
yhat_svml1_test = svmc_l1_best.predict(np.transpose(test_embed_filter))
yhat_svmc_test = svmc_best.predict(np.transpose(test_embed_filter))

# Predict on training set
yhat_svml1_train = svmc_l1_best.predict(np.transpose(train_embed_filter))
yhat_svmc_train = svmc_best.predict(np.transpose(train_embed_filter))

# Calculate evaluation metrics (Final validation test set)
res_svmc_l1_test = {
    'balanced_accuracy':balanced_accuracy_score(test_labels_filter, yhat_svml1_test),
    'bag_accuracy':accuracy_score(test_labels_filter, yhat_svml1_test),
    'bag_precision':precision_score(test_labels_filter, yhat_svml1_test, average='weighted'),
    'bag_recall':recall_score(test_labels_filter, yhat_svml1_test, average='weighted'),
    }
res_svmc_test = {
    'balanced_accuracy':balanced_accuracy_score(test_labels_filter, yhat_svmc_test),
    'bag_accuracy':accuracy_score(test_labels_filter, yhat_svmc_test),
    'bag_precision':precision_score(test_labels_filter, yhat_svmc_test, average='weighted'),
    'bag_recall':recall_score(test_labels_filter, yhat_svmc_test, average='weighted'),
    }

# Calculate evaluation metrics (Training set)
res_svmc_l1_train = {
    'balanced_accuracy':balanced_accuracy_score(train_labels_filter, yhat_svml1_train),
    'bag_accuracy':accuracy_score(train_labels_filter, yhat_svml1_train),
    'bag_precision':precision_score(train_labels_filter, yhat_svml1_train, average='weighted'),
    'bag_recall':recall_score(train_labels_filter, yhat_svml1_train, average='weighted'),
    }
res_svmc_train = {
    'balanced_accuracy':balanced_accuracy_score(train_labels_filter, yhat_svmc_train),
    'bag_accuracy':accuracy_score(train_labels_filter, yhat_svmc_train),
    'bag_precision':precision_score(train_labels_filter, yhat_svmc_train, average='weighted'),
    'bag_recall':recall_score(train_labels_filter, yhat_svmc_train, average='weighted'),
    }



# Print information about the data set
msg = ("The {} Estimator was trained on {} traing bags. Final evaluation was " +
        "performed using {} testing bags\n\n")
print(msg.format("SVMC Linear L1 Regularized", 
                  train_embed_filter.shape[1], 
                  test_embed_filter.shape[1]))
print(msg.format("SVMC kernelized L2 Regularized", 
                  train_embed_filter.shape[1], 
                  test_embed_filter.shape[1]))

msg = ("A total of {} {} bags were excluded becase they did not contain " +
        "greater than 5 instances\n")
print(msg.format(train_bags.shape[0] - _filter_train.shape[0], "training"))
print(msg.format(test_bags.shape[0] - _filter_test.shape[0], "test"))


# Print test results
_print_results_dict(res_svmc_l1_test, 
                    ("Final evaluation results of Linear SVMC L1 Regularized Estimator "+
                      "using embedding of bag label:\n"))
_print_results_dict(res_svmc_test, 
                    ("Final evaluation results of Kernel SVMC L2 Regularized Estimator "+
                      "using embedding of bag label:\n"))

# Print training results
_print_results_dict(res_svmc_l1_train, 
                    ("Training evaluation results of Linear SVMC L1 Regularized Estimator "+
                      "using embedding of bag label:\n"))
_print_results_dict(res_svmc_train, 
                    ("Training evaluation results of Kernel SVMC L2 Regularized Estimator "+
                      "using embedding of bag label:\n"))



#%% Pickle and save model for later


# Save the model
with open('./svmc_l1_miles.clf', mode='wb') as f:
    pickle.dump(svmc_l1_best, f)
    
with open('./svmc_rbf_miles.clf', mode='wb') as f:
    pickle.dump(svmc_best, f)

# Save concept class examples for MILES embedding and prediction 
with open('./miles_concept_features.dat', mode='wb') as f:
    pickle.dump(C_features, f)
    
# Save training and validation data
with open('./data_milesembedded_trian.dat', mode='wb') as f:
    pickle.dump((train_embed_filter, train_labels_filter), f)

with open('./data_milesembedded_test.dat', mode='wb') as f:
    pickle.dump((test_embed_filter, test_labels_filter), f)
    
# Save validation results (manually, see validation_results_miles.txt)

