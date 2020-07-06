# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:24:44 2019

@author: z003vrzk
"""

# Python imports
import sys
import os

# Third party imports
import numpy as np
from sklearn.metrics import (auc_roc_score, recall_score,
                             balanced_accuracy_score, precision_score,
                             recall_score, accuracy_score)

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)


#%%

y_true = np.array([0, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 0, 1])
weight = np.array([1, 1, 1, 1, 1, 1])

balanced_accuracy_score(y_true, y_pred)

# Balanced accuracy is the mean of sensitivity and specificity
true_positive = (y_pred == 1) & (y_true == 1)
false_negative = (y_pred == 0) & (y_true == 1)
false_positive = (y_pred == 1) & (y_true == 0)
true_negative = (y_pred == 0) & (y_true == 0)
sensitivity = sum(true_positive) / (sum(true_positive) + sum(false_negative))
specificity = sum(true_negative) / (sum(true_negative) + sum(false_positive))


recall0 = recall_score(y_true, y_pred, pos_label=0)
recall1 = recall_score(y_true, y_pred, pos_label=1)



#%% Testing metrics

y_true = np.array(['ahu','alarm','boiler','chiller','chiller',
                   'exhaust_fan','ahu','ahu','ahu'])
y_pred = np.array(['alarm','ahu','boiler','chiller','chiller',
                   'exhaust_fan','ahu','ahu','ahu'])

# Test accuracy, precision, recall
acc = accuracy_score(y_true, y_pred) # 7 / 9 are correct

"""Precision is ability of estimator to give positive labels to instances
not identified as positive
Precision = TP / (TP + FP)
TP = 7
FP = 2"""
prec = precision_score(y_true, y_pred, average='weighted')

"""Recall is ability of estimator to identify all positive classes
Recall = TP / (TP + FN)
TP = 7
FN = 2"""
rec = recall_score(y_true, y_pred, average='weighted')
rec2 = recall_score(y_true, y_pred, average='weighted', labels=['ahu'])




sum(y_true == y_pred) / len(y_true)