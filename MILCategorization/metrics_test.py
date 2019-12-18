# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:24:44 2019

@author: z003vrzk
"""

import numpy as np
from sklearn.metrics import auc_roc_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score

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