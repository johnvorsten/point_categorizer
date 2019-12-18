# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:11:41 2019

@author: z003vrzk
"""

import numpy as np
from sklearn.metrics import auc_roc_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score

from misvm import bag_set, parse_c45
import misvm

"""
Here, the bags argument is a list of "array-like" (could be NumPy arrays, or a 
list of lists) objects representing each bag. Each (array-like) bag has m rows
 and f columns, which correspond to m instances, each with f features. Of 
 course, m can be different across bags, but f must be the same. Then labels is 
 an array-like object containing a label corresponding to each bag. Each label 
 must be either +1 or -1. You will likely get strange results if you try using 
 0/1-valued labels. After training the classifier, you can call the predict
 function as:
"""

example_set = 


