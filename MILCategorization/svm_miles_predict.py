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
CONCEPT_FEATURES = 2855 # Number of concept class features

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
    NETDEVID: str
    SYSTEM: str


class MILESEmbedding:
    
    def __init__(self, filename:Union[str,bytes]) -> None:
        """"""
        self.C_features = self._load_concept_class(filename)
        
        return None
    
    @staticmethod
    def _load_concept_class(filename:Union[str,bytes]) -> np.ndarray:
        with open(filename, mode='rb') as pickled_features:
            C_features = pickle.load(pickled_features)
        return C_features
    
    def embed_data(self, bag: np.ndarray, sigma: float=3):
        """
        inputs
        -------
        bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
        a bag, and p is the instance space of a bag
        sigma: (float) regularization paramter
        distance: (str) 'euclidean' is the only supported distance metric"""
        
        if not LoadMIL.validate_bag(bag):
            msg=("The passed bag either contained a) All virtual instances b) "+
            "all L2SL points, or c) No instances made it through the pipeline "+
            "and has a shape of [0,n]")
            raise ValueError(msg)
        
        if not self.validate_bag_size_configuration(bag):
            msg=("Bag passed must have {} features according to configuration " +
             "sanity check. Got {}")
            raise ValueError(msg.format(CONCEPT_FEATURES, bag.shape[1]))
            
        return embed_bag(self.C_features, bag, sigma=sigma, distance='euclidean')
    
    @classmethod
    def validate_bag_size_configuration(cls, bag: np.ndarray) -> bool:
        """inputs
        -------
        bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
            a bag, and p is the instance space of a bag
        outputs
        -------
        (bool) if the instance space (number of features in an instance)
        matches the concept class, and ALSO the expected number for the data
        pipeline (see configuration)
        """
        
        if not bag.shape[1] == CONCEPT_FEATURES:
            msg=("The passed bag is of shape {}. Expected bag "+ 
                 "shape {}. Bag and feature must have the same instance space "+ 
                 "/ shape along axis=1.")
            print(msg.format(bag.shape, CONCEPT_FEATURES))
            return False
        
        return True
    
    def validate_bag_size_concept(self, bag: np.ndarray) -> bool:
        """inputs
        -------
        concept_class: (np.ndarray)
        bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
            a bag, and p is the instance space of a bag
        outputs
        -------
        (bool) if the instance space (number of features in an instance)
        matches the concept class, and ALSO the expected number for the data
        pipeline (see configuration)
        """
        
        if not bag.shape[1] == self.C_features.shape[1]:
            msg=("The passed bag is of shape {}. Loaded features are of "+ 
                 "shape {}. Bag and feature must have the same instance space "+ 
                 "/ shape along axis=1.")
            print(msg.format(bag.shape, self.C_features.shape))
            return False
        
        return True


class BasePredictor:
    
    def __init__(self, classifier_filename:Union[str,bytes]):
        # Load classifier
        self.classifier = self._load_predictor(classifier_filename)
        # Load transform pipeline
        self.numeric_transform_pipeline_MIL = Transform.numeric_transform_pipeline_MIL()
        
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
        clean_data = self.numeric_transform_pipeline_MIL().fit_transform(df_raw)
        
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
        
        if data.shape[0] == CONCEPT_FEATURES:
            return True
        else:
            msg=("Invalid data shape. Got {}, required a dense numpy array " +
                 "with shape {}".format(data.shape, CONCEPT_FEATURES))
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

