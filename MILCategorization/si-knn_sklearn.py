# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 09:02:37 2020

@author: z003vrzk
"""

# Python imports
import sys
import os
from collections import Counter

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
from MILCategorization import mil_load

SingleInstanceGather = mil_load.SingleInstanceGather()
LoadMIL = mil_load.LoadMIL()

#%%

# Load dataset
_file = r'../data/MIL_dataset.dat'
_dataset = LoadMIL.load_mil_dataset(_file)
_bags = _dataset['dataset']
_bag_labels = _dataset['bag_labels']

# Load cat dataset
_cat_file = r'../data/MIL_cat_dataset.dat'
_cat_dataset = LoadMIL.load_mil_dataset(_cat_file)
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
Xtrain, Ytrain = SingleInstanceGather.bags_2_si(train_bags,
                                                train_bag_labels,
                                                sparse=True)
Xtest, Ytest = SingleInstanceGather.bags_2_si(test_bags,
                                              test_bag_labels,
                                              sparse=True)

# Unpack categorical
Xtrain_cat, Ytrain_cat = SingleInstanceGather.bags_2_si(cat_train_bags,
                                                        cat_train_bag_labels,
                                                        sparse=True)
Xtest_cat, Ytest_cat = SingleInstanceGather.bags_2_si(cat_test_bags,
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



#%% Cross validation



# Precision scoring
precision = make_scorer(precision_score, average='weighted')
recall = make_scorer(recall_score, average='weighted')

scorer = {'accuracy':'accuracy',
          'balanced_accuracy':'balanced_accuracy',
          'recall_weighted':recall,
          'precision_weighted':precision}
res_knn = cross_validate(knn, Xtrain, Ytrain, cv=3, scoring=scorer)
res_mnb = cross_validate(multiNB, Xtrain_cat, Ytrain_cat, cv=3, scoring=scorer)
res_cnb = cross_validate(compNB, Xtrain_cat, Ytrain_cat, cv=3, scoring=scorer)


#%% Predict on bags using most common label assigned to instances

"""
# Initial Values
CV = 3
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
results = {}

# Define a scorer and Metrics
precision = make_scorer(precision_score, average='weighted')
recall = make_scorer(recall_score, average='weighted')
accuracy = make_scorer(accuracy_score, average='weighted')
accuracy_balanced = make_scorer(balanced_accuracy_score, average='weighted')
scorer = {'precision_weighted':precision,
          'recall_weighted':recall,
          'accuracy':accuracy,
          'accuracy_balanced':accuracy_balanced}

# Define an estimator
ESTIMATOR = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

# Load raw datasets, Already loaded above
BAGS = cat_train_bags
BAG_LABELS = cat_train_bag_labels

# Split bags into training and validation sets
rs = ShuffleSplit(n_splits=CV, test_size=TEST_SIZE, train_size=TRAIN_SIZE)
for train_index, test_index in rs.split(BAGS, BAG_LABELS))

    # Split bags
    Xtrain, Ytrain = BAGS[train_index], BAG_LABELS[train_index]
    Xtest, Ytest = BAGS[test_index], BAG_LABELS[test_index]

    # Convert training set to single instance to fit the estimator
    Xtrain_si, Ytrain_si = SingleInstanceGather.bags_2_si(cat_train_bags,
                                                          cat_train_bag_labels,
                                                          sparse=True)

    # Fit an estimator on SI data
    ESTIMATOR.fit(Xtrain_si, Ytrain_si)

    # Predict instances in a bag
    bag_predictions = []
    for BAG, BAG_LABEL in zip(Xtest, Ytest):
        y_hat = ESTIMATOR.predict(BAG)
        reduced_label = reduce_bag_label(y_hat, method='mode')
        bag_predictions.append(reduced_label)

    # Estimate metrics on bags
    # TODO



# Predict on bags
bag_labels = []
bag = cat_train_bags[0]
y_hat = multiNB.predict(bag)
label = reduce_bag_label(y_hat, method='mode')
bag_labels.append(label)
"""







