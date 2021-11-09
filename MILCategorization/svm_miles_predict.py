# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:12:39 2021

@author: vorst
"""

# Python imports
from typing import Union, List
import pickle
import configparser
from dataclasses import dataclass

# Third party imports
from sklearn.model_selection import (train_test_split, StratifiedShuffleSplit, 
                                     GridSearchCV)
import sklearn as skl
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted
import numpy as np
from pyMILES.embedding import embed_bag
import pandas as pd

# Local imports
from mil_load import LoadMIL
from transform_mil import Transform

# Global declarations
SVMC_l1_classifier_filename = r"./svmc_l1_miles.clf"
SVMC_rbf_classifier_filename = r"./svmc_rbf_miles.clf"
concept_class_filename = r"./miles_concept_features.dat"

#%% Classses and data

@dataclass
class RawInputData:
    """Raw input data from HTTP Web form
    Note, any default values are not required by the estimator, and are 
    removed by the data cleaning pipeline
    
    Required numeric attributes
    ['DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO', 'SLOPE', 'INTERCEPT']
    
    Required categorical attributes
    ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS','SENSORTYPE', 'DEVUNITS']
    
    Requried text attributes
    ['NAME', 'DESCRIPTOR']
    """
    # Required numeric attributes
    DEVICEHI: float
    DEVICELO: float
    SIGNALHI: float
    SIGNALLO: float
    SLOPE: float
    INTERCEPT: float
    # Required categorical attributes
    TYPE: str
    ALARMTYPE: str
    FUNCTION: str
    VIRTUAL: bool
    CS: str
    SENSORTYPE: str
    DEVUNITS: str
    # Requried text attributes
    NAME: str
    DESCRIPTOR: str


class MILESEmbedding:
    
    def __init__(self, filename:Union[str,bytes]) -> None:
        """"""
        self.C_features = self._load_concept_class(filename)
        
        return None
    
    @staticmethod
    def _load_concept_class(filename:Union[str,bytes]) -> np.ndarray:
        with open(filename, mode='rb') as pickled_features:
            C_features = pickle.load(pickled_features)
        # TODO Error on missing file
        return C_features
    
    def embed_data(self, bag: np.ndarray, sigma: int=3):
        """
        inputs
        -------
        bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
        a bag, and p is the instance space of a bag
        sigma: (float) regularization paramter
        distance: (str) 'euclidean' is the only supported distance metric"""
        
        if not LoadMIL.validate_bag(bag):
            raise ValueError("Bag passed is not valid")
            
        return embed_bag(self.C_features, bag, sigma=sigma, distance='euclidean')
    

class BasePredictor:
    
    def __init__(self, classifier_filename:Union[str,bytes]):
        self.classifier = self._load_predictor(classifier_filename)
        return None
    
    @staticmethod
    def _load_predictor(filename:Union[str,bytes]) -> None:
        """Load a trained estimator from a pickled sklearn object
        inputs
        -------
        filename: (str)
        outputs
        -------"""        
        with open(filename, mode='rb') as pickled_classifier:
            classifier = pickle.load(pickled_classifier)
                    
        return classifier

    def predict(self, data: List[RawInputData]) -> str:
        """Predict on an embedded bag
        inputs
        -------
        data: () Raw data input
        outputs
        -------"""
        
        # Transform raw data
        clean_data = self._transform_data(data)
        
        # Embed raw data
        MILESEmbedder = MILESEmbedding(concept_class_filename)
        embedded_data = MILESEmbedder.embed_data(clean_data)
        
        # Validate data to estimator
        self._validate_transformed_data(embedded_data)
        
        # Predict on embedded vector
        # embedded_data is of shape (n,p)
        prediction = self.classifier.predict(np.transpose(embedded_data))
        
        return prediction

    def _transform_data(self, data: List[RawInputData]):
        """Transform raw data into a usable input to an estimator
        inputs
        -------
        data: (RawInputData) Python dataclass specified for this predictor
        outputs
        -------
        """
        
        df_raw = pd.DataFrame(data)
        clean_data = Transform.numeric_transform_pipeline_MIL().fit_transform(df_raw)
        
        return clean_data

    def _validate_transformed_data(self, data: np.ndarray):
        """Validate that the exact required input data is used for prediction 
        on the estimator
        the embedding is a (j,) array where each bag is encoded into a 
        feature vector which represents a similarity measure between the bag and 
        concept class. j is the number of instances in the concept class
        The concept class is chosen to be 4633 instances long
        inputs
        -------
        data: (np.ndarray) data must be a dense numpy array with shape (n, 3236)
        outputs
        -------
        bool
        """
        
        if data.shape[0] == 4663:
            return True
        else:
            msg=("Invalid data shape. Got {}, required a dense numpy array " +
                 "with shape (4663)".format(data.shape))
            raise ValueError(msg)
            
        return False


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

