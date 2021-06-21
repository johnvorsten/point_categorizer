# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 20:07:07 2021

@author: vorst
"""

# Python imports
import configparser
import os, sys

# Third party imports
from sklearn.model_selection import train_test_split
import sklearn as skl
from sklearn import svm
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
import mil_load
from pyMILES.embedding import embed_all_bags
        
# Global declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

LoadMIL = mil_load.LoadMIL(server_name,
                           driver_name,
                           database_name)

#%%

# Load
# bags, labels = LoadMIL.gather_mil_dataset(pipeline='whole')
dataset_numeric = LoadMIL.load_mil_dataset(numeric_feature_file)
dataset_categorical = LoadMIL.load_mil_dataset(categorical_feature_file)

# Split data into training and testing sets
"""TODO: In the ideal scenario I will hand-pick the most optimal training 
bags. These will most accurately represent the categories that I 
want to identify"""
X_train_sp, X_test_sp, y_train, y_test = train_test_split(dataset_numeric['dataset'], 
                                                    dataset_numeric['bag_labels'],
                                                    train_size=0.1)

"""Create a concept class, the set of all training instances from positive and 
negative bags C = {x^k : k=1, ..., n}
Where x^k is the kth instance in the entire training set
"""
# Create concept class from training instances
C_features, C_labels = mil_load.SingleInstanceGather.bags_2_si(X_train_sp, 
                                                               y_train, 
                                                               sparse=False)

# Generate dense bags
"""bags: (np.ndarray) shape (i,j,p) where n is the number of bags, j is the 
    number of instances per bag, and p is the feature space per instance 
    in a bag"""
X_train = [x.toarray() for x in X_train_sp]
X_test = [x.toarray() for x in X_test_sp]

# Encode data into similarity matrix
embedded_bags = embed_all_bags(concept_class=C_features, 
                               bags=X_test_sp,
                               sigma=3,
                               distance='euclidean')

# Apply Nystroem transformer for large datasets?
# Use a linear SVM for large datasets? (10,000 sampels is what they recommend)

#%% Create and train SVM
GAMMA_SVC = 'scale'
PENALTY = 'l1' # L1 loss penalization
LOSS = 'squared_hinge' # Loss function
C = 1.0 # SVM regularization, inversely proportional

# Define SVM
svmc_l1 = skl.svm.LinearSVC(loss=LOSS, penalty=PENALTY, C=C, dual=False, max_iter=5000)

# SVC Using LibSVM uses the squared l2 loss
svmc = skl.svm.SVC(kernel='rbf', gamma=GAMMA_SVC, C=C)

# Define grid search parameters
params_l1svc = {'C':[0.5,1,2],
                }
params_svc = {'C':[0.5,1,2],
              'kernel':['rbf', 'poly']}

# Define scorers
scoring = {'accuracy':skl.metrics.make_scorer(skl.metrics.accuracy_score),
           'precision':skl.metrics.make_scorer(skl.metrics.precision_score, average='weighted'),
           'recall':skl.metrics.make_scorer(skl.metrics.recall_score, average='weighted')
           'Confusion':skl.metrics.make_scorer(skl.metrics.confusion_matrix,)}


# Grid search
svmc_l1_gs = skl.model_selection.GridSearchCV(estimator=svmc_l1,
                                 param_grid=params_l1svc,
                                 scoring=scoring,
                                 refit=False,
                                 n_jobs=6,
                                 cv=None, # Default 5-fold validation
                                 refit='accuracy',
                                 )


svmc_gs = skl.model_selection.GridSearchCV(estimator=svmc,
                                 param_grid=params_svc,
                                 scoring=['accuracy','precision','recall'],
                                 n_jobs=6,
                                 refit=False,
                                 cv=None, # Default 5-fold validation
                                 refit='accuracy',
                                 )
# Estimators expect (instance,features). embedded bags are encoded where 
# features are along axis=1
svmc_l1_gs.fit(np.transpose(embedded_bags), y_test)
svmc_gs.fit(np.transpose(embedded_bags), y_test)

# Print results
print("L1 SVM Results: ", svmc_l1_gs.cv_results_, "\n\n")
print("rbf, polynomial SVM Results: ", svmc_gs.cv_results_, "\n\n")

#%% Testing




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

