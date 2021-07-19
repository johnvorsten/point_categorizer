# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:07:07 2021

@author: vorst
"""

# Python imports
import configparser
import os, sys
from typing import Union

# Third party imports
from sklearn.model_selection import train_test_split
import sklearn as skl
from sklearn import svm
import numpy as np
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
from mil_load import load_mil_dataset, LoadMIL, bags_2_si
from pyMILES.embedding import embed_all_bags
        
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
"""TODO: In the ideal scenario I will hand-pick the most optimal training 
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


"""Create a concept class, the set of all training instances from positive and 
negative bags C = {x^k : k=1, ..., n}
Where x^k is the kth instance in the entire training set
"""
# Create concept class from training instances
C_features, C_labels = bags_2_si(train_bags_sp, train_labels, sparse=True)

# Generate dense bags
"""bags: (np.ndarray) shape (i,j,p) where n is the number of bags, j is the 
    number of instances per bag, and p is the feature space per instance 
    in a bag"""
train_bags = _densify_bags(train_bags_sp)
test_bags = _densify_bags(test_bags_sp)

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

# Encode data into similarity matrix
# WARNING - testing bags are being embedded onto the training set
# This means that the 'testing' set contains a similarity measure to the training
# instances. I will perform cross validation on the testing instances
embedded_train = embed_all_bags(concept_class=C_features, 
                                bags=test_bags_sp,
                                sigma=3,
                                distance='euclidean')
embedded_test = embed_all_bags(concept_class=C_features, 
                                bags=train_bags_sp,
                                sigma=3,
                                distance='euclidean')





#%%
"""Perform cross-validation on embedded dataset with two estimators:
1. L1 SVM
2. SVM with radial-basis-function distance kernel

Use a custom
"""
GAMMA_SVC = 'scale'
PENALTY = 'l1' # L1 loss penalization
LOSS = 'squared_hinge' # Loss function
C = 1.0 # SVM regularization, inversely proportional

# Define SVM
svmc_l1 = skl.svm.LinearSVC(loss=LOSS, penalty=PENALTY, C=C, 
                            dual=False, max_iter=2500)

# SVC Using LibSVM uses the squared l2 loss
svmc = skl.svm.SVC(kernel='rbf', gamma=GAMMA_SVC, C=C)

# Define grid search parameters
params_l1svc = {'C':[0.5,2,5],
                }
params_svc = {'C':[2,5,10],
              'kernel':['rbf', 'poly']}

# Define scorers
scoring = {'accuracy':skl.metrics.make_scorer(skl.metrics.accuracy_score),
           'precision':skl.metrics.make_scorer(skl.metrics.precision_score, average='weighted'),
           'recall':skl.metrics.make_scorer(skl.metrics.recall_score, average='weighted'),
           }


# Grid search
svmc_l1_gs = skl.model_selection.GridSearchCV(
    estimator=svmc_l1,
    param_grid=params_l1svc,
    scoring=scoring,
    n_jobs=6,
    cv=3, # Default 5-fold validation
    refit='accuracy',
    )


svmc_gs = skl.model_selection.GridSearchCV(
    estimator=svmc,
    param_grid=params_svc,
    scoring=scoring,
    n_jobs=6,
    cv=3, # Default 5-fold validation
    refit='accuracy',
    )

# Filter training and testing instances so that only bags with >= 5 instances
# Are included in grid search
_filter_train = _filter_bags_by_size(train_bags, min_instances=5, max_instances=2000)
_filter_test = _filter_bags_by_size(test_bags, min_instances=5, max_instances=2000)
train_embed_filter = embedded_train[_filter_train]
train_labels_filter = train_labels[_filter_train]
test_embed_filter = embedded_test[_filter_test]
test_labels_filter = test_labels[_filter_test]

# Perform cross validation - REMEMBER that we are performing cross validation
# On the TESTING set this time because the testing bags were embedded using
# the training bags as the concept class
svmc_l1_gs.fit(np.transpose(test_embed_filter), test_labels_filter)
svmc_gs.fit(np.transpose(test_embed_filter), test_labels_filter)

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

# Define the training and testing data set
# Use stratified folding to preserve class imbalance
# Filter out bags with only a single instance
_filter_train = _filter_bags_by_size(train_bags, 
                                     min_instances=5,
                                     max_instances=2000)
_filter_test = _filter_bags_by_size(test_bags,
                                    min_instances=5,
                                    max_instances=2000)

# Convert bags to dense for KNN estimator
train_embed_filter = embedded_train[_filter_train]
train_labels_filter = train_labels[_filter_train]
test_embed_filter = embedded_test[_filter_test]
test_labels_filter = test_labels[_filter_test]

# Define estimators (Choose from cross-validation)



# Define scoring metrics
scoring = {'accuracy':skl.metrics.make_scorer(skl.metrics.accuracy_score),
           'precision':skl.metrics.make_scorer(skl.metrics.precision_score, average='weighted'),
           'recall':skl.metrics.make_scorer(skl.metrics.recall_score, average='weighted'),
           }

# Fit the estimator



# Predict on the validation set
yhat_svml1_test = 
yhat_svmrbf_test = 

# Predict on training set
yhat_svml1_train = 
yhat_svmrbf_train = 

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



# Print information about the data set
msg = ("The {} Estimator was trained on {} traing bags. Final evaluation was " +
       "performed using {} testing bags\n\n")
print(msg.format("KNN", 
                 _train_bags_dense.shape[0], 
                 _test_bags_dense.shape[0]))
print(msg.format("Multinomial Native Bayes", 
                 _train_bags_cat.shape[0], 
                 _test_bags_cat.shape[0]))
print(msg.format("Component Native Bayes", 
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
                    ("Final evaluation results of component NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))


# Print training results
_print_results_dict(res_knn_infer_train, 
                    ("Training evaluation of KNN Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_multiNB_infer_train, 
                    ("Training evaluation results of multinomial NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))
_print_results_dict(res_compNB_infer_train, 
                    ("Training evaluation results of component NB Estimator using inference "+
                     "of bag label based on mode statistic of instance labels:\n"))










































#%% Testing



"""Estimators expect (instance,features). embedded bags are encoded where 
features are along axis=1"""
_reduced_embedded_bags = embedded_bags[:200]
y_test_reduced = y_test[:200]
svmc_l1_gs.fit(np.transpose(_reduced_embedded_bags), y_test_reduced)
svmc_gs.fit(np.transpose(_reduced_embedded_bags), y_test_reduced)


# Define SVM
svmc_l1 = skl.svm.LinearSVC(loss=LOSS, penalty=PENALTY, C=C, dual=False, max_iter=5000)

# SVC Using LibSVM uses the squared l2 loss
svmc = skl.svm.SVC(kernel='rbf', gamma=GAMMA_SVC, C=C)

# Fit and train models
svmc_l1.fit(np.transpose(embedded_bags), y_test)
y_pred_l1 = svmc_l1.predict(np.transpose(embedded_bags))
accuracy_l1 = skl.metrics.accuracy_score(y_test, y_pred_l1)
precision_l1 = skl.metrics.precision_score(y_test, y_pred_l1, average='weighted')
recall_l1 = skl.metrics.recall_score(y_test, y_pred_l1, average='weighted')
conf_l1 = skl.metrics.confusion_matrix(y_test, y_pred_l1)
print("Accuracy L1: {}".format(accuracy_l1))
print("Precision L1: {}".format(precision_l1))
print("Recall L1: {}".format(recall_l1))
print("Confusion Matrix L1\n: {}".format(conf_l1))

svmc.fit(np.transpose(embedded_bags), y_test)
y_pred_rb = svmc.predict(np.transpose(embedded_bags))
accuracy_rb = skl.metrics.accuracy_score(y_test, y_pred_rb)
precision_rb = skl.metrics.precision_score(y_test, y_pred_rb, average='weighted')
recall_rb = skl.metrics.recall_score(y_test, y_pred_rb, average='weighted')
conf_rb = skl.metrics.confusion_matrix(y_test, y_pred_rb)
print("Accuracy L1: {}".format(accuracy_rb))
print("Precision L1: {}".format(precision_rb))
print("Recall L1: {}".format(recall_rb))
print("Confusion Matrix L1\n: {}".format(conf_rb))

