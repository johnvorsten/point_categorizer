# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:13:22 2021

@author: vorst
"""

# Python imports

# Third party imports

# Local imports
from mil_load import LoadMIL

# Global declarations


#%% Classses and data


# Depending on the type of estimator there will be a differnt pipeline
# Multinomial Native Bayes and Complement Native Bayes use sparse features
# SVM and KNN use dense features
categorical_pipeline = LoadMIL.categorical_transform_pipeline()
numeric_pipeline = LoadMIL.numeric_transform_pipeline()
raw_data = None # TODO
bag = numeric_pipeline.fit_transform(raw_data)


#%% Main

