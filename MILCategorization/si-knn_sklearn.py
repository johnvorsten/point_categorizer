# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 09:02:37 2020

@author: z003vrzk
"""

# Python imports
import sys
import os
from collections import Counter
import configparser

# Third party imports
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import (make_scorer, SCORERS, precision_score,
                             recall_score, accuracy_score, balanced_accuracy_score)
from sklearn.naive_bayes import MultinomialNB, ComplementNB


# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract import extract
from transform import transform_pipeline
from mil_load import bags_2_si, bags_2_si_generator, LoadMIL
from bag_cross_validate import cross_validate_bag, BagScorer

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

# Load dataset
_file = r'../data/MIL_dataset.dat'
_dataset = loadMIL.load_mil_dataset(_file)
_bags = _dataset['dataset']
_bag_labels = _dataset['bag_labels']

# Load cat dataset
_cat_file = r'../data/MIL_cat_dataset.dat'
_cat_dataset = loadMIL.load_mil_dataset(_cat_file)
_cat_bags = _cat_dataset['dataset']
_cat_bag_labels = _cat_dataset['bag_labels']

# Split dataset
rs = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8)
train_index, test_index = next(rs.split(_bags, _bag_labels))
train_bags, train_bag_labels = _bags[train_index], _bag_labels[train_index]
test_bags, test_bag_labels = _bags[test_index], _bag_labels[test_index]

# Split cat dataset
train_index, test_index = next(rs.split(_cat_bags, _cat_bag_labels))
cat_train_bags, cat_train_bag_labels = _cat_bags[train_index], _cat_bag_labels[train_index]
cat_test_bags, cat_test_bag_labels = _cat_bags[test_index], _cat_bag_labels[test_index]

# Unpack bags into single instances for training and testing
# Bags to single instances
si_X_train, si_y_train = bags_2_si(train_bags,
                           train_bag_labels,
                           sparse=True)
si_X_test, si_y_test = bags_2_si(test_bags,
                         test_bag_labels,
                         sparse=True)

# Unpack categorical
si_X_train_cat, si_y_train_cat = bags_2_si(cat_train_bags,
                                   cat_train_bag_labels,
                                   sparse=True)
si_X_test_cat, si_y_train_cat = bags_2_si(cat_test_bags,
                                 cat_test_bag_labels,
                                 sparse=True)

#%% Estimators

# K-NN
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform',
                           algorithm='ball_tree', n_jobs=4)
# knn.fit(Xtrain, Ytrain)
# yhat = knn.predict(Xtest)

# Multinomial Native Bayes
multiNB = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
# multiNB.fit(Xtrain_cat, Ytrain_cat)
# yhat_mnb = multiNB.predict(Xtest_cat)

# CommplementNB - Like multinomial but for imbalanced datasets
compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)
# compNB.fit(Xtrain_cat, Ytrain_cat)
# yhat_cnb = compNB.predict(Xtest_cat)



#%% Cross validation, Single instance categorization

# Define scorers
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='weighted'),
           'recall': make_scorer(recall_score, average='weighted'),
           'balanced_accuracy':'balanced_accuracy',
           }

# # Estimation on single instance bags
# res_knn = cross_validate(estimator=knn, 
#                          X=si_X_train, 
#                          y=si_y_train, 
#                          cv=3, 
#                          scoring=scoring,
#                          )
# res_mnb = cross_validate(estimator=multiNB, 
#                          X=si_X_train_cat, 
#                          y=si_y_train_cat, 
#                          cv=3, 
#                          scoring=scoring,
#                          )
# res_cnb = cross_validate(estimator=compNB, 
#                          X=si_X_train_cat, 
#                          y=si_y_train_cat, 
#                          cv=3, 
#                          scoring=scoring,
#                          )


#%% Predict on bags using most common label assigned to instances

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
BAGS = cat_train_bags
BAG_LABELS = cat_train_bag_labels

# Split bags into training and validation sets
rs = ShuffleSplit(n_splits=CV, test_size=TEST_SIZE, train_size=TRAIN_SIZE)
for train_index, test_index in rs.split(BAGS, BAG_LABELS):

    # Split bags
    _x_train_bags, _y_train_bags = BAGS[train_index], BAG_LABELS[train_index]
    _x_test_bags, _y_test_bags = BAGS[test_index], BAG_LABELS[test_index]

    # Convert training set to single instance to fit the estimator
    _x_train_si, _y_train_si = bags_2_si(_x_train_bags,
                                         _y_train_bags,
                                         sparse=True)
    
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



#%% Predict on bags using cross validation

# Create estimators
knn = KNeighborsClassifier(n_neighbors=10, weights='uniform',
                           algorithm='ball_tree', n_jobs=4)

# Multinomial Native Bayes
multiNB = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

# CommplementNB - Like multinomial but for imbalanced datasets
compNB = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)


# Define evaluation metrics
accuracy_scorer = make_scorer(accuracy_score)
bagAccScorer = BagScorer(accuracy_scorer, sparse=False) # Accuracy score, no factory function
precision_scorer = make_scorer(precision_score, average='binary')
bagPreScorer = BagScorer(precision_scorer, sparse=False)
recall_scorer = make_scorer(recall_score, average='weighted')
bagRecScorer = BagScorer(recall_scorer, sparse=False)

scoring = {'bag_accuracy':bagAccScorer,
           'bag_precision':bagPreScorer,
           'bag_recall':bagRecScorer,
           }

# Cross validate bags
knn_res = cross_validate_bag(estimator=knn, 
                            X=train_bags, 
                            y=train_bag_labels, 
                            groups=None, 
                            scoring=scoring, # Custom scorer... 
                            cv=3,
                            n_jobs=3, 
                            verbose=0, 
                            fit_params=None,
                            pre_dispatch='2*n_jobs', 
                            return_train_score=False,
                            return_estimator=False, 
                            error_score=np.nan)




