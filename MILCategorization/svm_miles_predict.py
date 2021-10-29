# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:12:39 2021

@author: vorst
"""

# Python imports
from typing import Union
import pickle
import configparser


# Third party imports
from sklearn.model_selection import (train_test_split, StratifiedShuffleSplit, 
                                     GridSearchCV)
import sklearn as skl
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted
import numpy as np
from pyMILES.embedding import embed_bag

# Local imports
from mil_load import LoadMIL

# Global declarations
SVMC_l1_classifier_filename = r"./svmc_l1_miles.clf"
SVMC_rbf_classifier_filename = r"./svmc_rbf_miles.clf"
concept_class_filename = r"./miles_concept_featuers.dat"


#%% Classses and data

class BasePredictor:
    
    def __init__(self, classifier_filename:Union[str,bytes]):
        self.classifier = self._load(classifier_filename)
        return None
    
    @staticmethod
    def _load(filename:Union[str,bytes]) -> None:
        """Load a trained estimator from a pickled sklearn object
        inputs
        -------
        filename: (str)
        outputs
        -------"""
        with open(filename, mode='rb') as pickled_classifier:
            classifier = pickle.load(pickled_classifier)
            
        return classifier


    def predict(self):
        """Predict on an embedded bag
        inputs
        -------
        outputs
        -------"""
        return None


categorical_pipeline = LoadMIL.categorical_transform_pipeline()
numeric_pipeline = LoadMIL.numeric_transform_pipeline()
raw_data = None # TODO
bag = numeric_pipeline.fit_transform(raw_data)


class MILES_embedding:
    
    def __init__(self, filename:Union[str,bytes]) -> None:
        """"""
        self.C_features = self._load_concept_class(filename)
        
        return None
    
    @staticmethod
    def _load_concept_class(filename:Union[str,bytes]) -> np.ndarray:
        with open(filename, mode='rb') as pickled_features:
            C_features = pickle.load(pickled_features)
            
        return C_features
    
    
    def embed_data(self, bag:np.ndarray):
        """
        inputs
        -------
        bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
        a bag, and p is the instance space of a bag
        sigma: (float) regularization paramter
        distance: (str) 'euclidean' is the only supported distance metric"""
        
        if not LoadMIL.validate_bag(bag):
            raise ValueError("Bag passed is not valid")
            
        return embed_bag(self.C_features, bag, sigma=3, distance='euclidean')
    


class SVMC_L1_miles:
    
    def __init__(self):
        """"""
        return None
    
    
    def load(self):
        """Load a trained estimator from a pickled sklearn object
        inputs
        -------
        outputs
        -------"""
        return None
    
    
    def predict(self):
        """Predict on an embedded bag
        inputs
        -------
        outputs
        -------"""
        return None


# Input sanitization
# The input bag must be of shape (n,p)

