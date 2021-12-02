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
from sklearn.utils.validation import check_is_fitted
import numpy as np
from pyMILES.embedding import embed_bag
import pandas as pd

# Local imports
from transform_mil import Transform
from dataclass_serving import RawInputData, RawInputDataPydantic

# Global declarations
SVMC_l1_classifier_filename = r"./svmc_l1_miles.clf"
SVMC_rbf_classifier_filename = r"./svmc_rbf_miles.clf"
CONCEPT_CLASS_FILENAME = r"./miles_concept_features.dat"
N_CONCEPT_FEATURES = 2855 # Number of concept class features

#%% Classses and data

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
    
    def embed_data(self, bag:np.ndarray, sigma:float=3):
        """
        inputs
        -------
        bag: (np.ndarray) of shape (j,p) where j is the number of instances in 
        a bag, and p is the instance space of a bag
        sigma: (float) regularization paramter
        distance: (str) 'euclidean' is the only supported distance metric"""
        
        if not validate_bag(bag):
            msg=("The passed bag either contained a) All virtual instances b) "+
            "all L2SL points, or c) No instances made it through the pipeline "+
            "and has a shape of [0,n]")
            raise ValueError(msg)
        
        if not self.validate_bag_size_configuration(bag):
            msg=("Bag passed must have {} features according to configuration " +
             "sanity check. Got {}")
            raise ValueError(msg.format(N_CONCEPT_FEATURES, bag.shape[1]))
            
        return embed_bag(self.C_features, bag, sigma=sigma, distance='euclidean')
    
    @classmethod
    def validate_bag_size_configuration(cls, bag:np.ndarray) -> bool:
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
        
        if not bag.shape[1] == N_CONCEPT_FEATURES:
            msg=("The passed bag is of shape {}. Expected bag "+ 
                 "shape {}. Bag and feature must have the same instance space "+ 
                 "/ shape along axis=1.")
            print(msg.format(bag.shape, N_CONCEPT_FEATURES))
            return False
        
        return True
    
    def validate_bag_size_concept(self, bag:np.ndarray) -> bool:
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
    
    def _validate_transformed_data(self, data:np.ndarray) -> bool:
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
        
        if data.shape[0] == self.C_features.shape[0]:
            return True
        else:
            msg=("Invalid data shape. Got {}, required a dense numpy array " +
                 "with shape {}")
            raise ValueError(msg.format((data.shape[0], self.C_features.shape[0])))
            
        return False


class BasePredictor:
    
    def __init__(self, classifier_filename:Union[str,bytes]):
        """inputs
        ------
        classifier_filename: (str) name of file for pickled sklearn classifier
        
        Example usage
        basePredictorL1 = BasePredictor(
            classifier_filename=SVMC_l1_classifier_filename)
        # Somehow, create some raw data
        input_data = RawInputData(
            # Required numeric attributes
            DEVICEHI=122.0,
            DEVICELO=32.0,
            SIGNALHI=10,
            SIGNALLO=0,
            SLOPE=1.2104,
            INTERCEPT=0.01,
            # Required categorical attributes
            TYPE="LAI",
            ALARMTYPE="Standard",
            FUNCTION="Value",
            VIRTUAL=0,
            CS="AE",
            SENSORTYPE="VOLTAGE",
            DEVUNITS="VDC",
            # Requried text attributes
            NAME="SHLH.AHU-ED.RAT",
            DESCRIPTOR="RETURN TEMP",
            NETDEVID='test-value',
            SYSTEM='test-system'
            )
        # Load raw data
        dfraw_input = pd.DataFrame(data=[input_data])
        # Create predictions from raw input data
        results_l1 = basePredictorL1.predict(dfraw_input)
        """
        # Load classifier
        self.classifier = self._load_predictor(classifier_filename)
        # Load transform pipeline
        self.numeric_transform_pipeline_MIL = Transform.numeric_transform_pipeline_MIL()
        # Load embedding class member
        self.MILESEmbedder = MILESEmbedding(CONCEPT_CLASS_FILENAME)

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

    def predict(self, data:Union[List[RawInputData], 
                                 RawInputData, 
                                 pd.DataFrame]) -> np.ndarray:
        """Predict on an embedded bag
        inputs
        -------
        data: (list(RawInputData), RawInputData, pandas.DataFrame) Raw data 
        input which is transformed by this class
        outputs
        -------
        prediction: (np.ndarray) of strings which represent the bag label"""
        
        # Transform raw data
        clean_data = self._transform_data(data)
        # Embed raw data
        embedded_data = self.MILESEmbedder.embed_data(clean_data)
        
        # Predict on embedded vector
        # embedded_data is of shape (k,) where k is the number of instances
        # In the concept class (concept class is shape k,p)
        prediction = self.classifier.predict(self._determine_reshape(embedded_data))
        
        return prediction

    def _transform_data(self, data:Union[List[RawInputData], 
                                         RawInputData, 
                                         pd.DataFrame]):
        """Transform raw data into a usable input to an estimator
        inputs
        -------
        data: (list(RawInputData), RawInputData, pandas.DataFrame) Raw data 
        input which is transformed by this class
        outputs
        -------
        clean_data: (np.ndarray) data passed through transformation pipeline
        and output as numpy array
        """
        
        df_raw = pd.DataFrame(data)
        clean_data = self.numeric_transform_pipeline_MIL.fit_transform(df_raw)
        
        return clean_data

    def _determine_reshape(self, data: np.ndarray) -> np.ndarray:
        """"Determine if input data should be reshaped to (1,n) if it is a
        single instance"""
        if data.ndim == 1:
            # A single instance must be of shape (1,n)
            return data.reshape(1,-1)
        elif data.ndim == 2:
            # Multiple instances of shape (j_instances, n_features)
            # If multipple instances are passed then the embedding produces a 
            # (n_features, j_instances) output -> requires transpose
            return np.transpose(data)
        else:
            msg="Data passed with more than 2 dimensions. Got {}"
            raise ValueError(msg.format(data.ndim))
        return data


class SVMCL1MILESPredictor(BasePredictor):
    
    def __init__(self, classifier_filename:Union[str,bytes]):
        """inputs
        ------
        classifier_filename: (str) name of file for pickled sklearn classifier
        
        Example usage
        basePredictorL1 = SVMCL1MILESPredictor(
            classifier_filename=SVMC_l1_classifier_filename)
        # Somehow, create some raw data
        input_data = RawInputData(
            # Required numeric attributes
            DEVICEHI=122.0,
            DEVICELO=32.0,
            SIGNALHI=10,
            SIGNALLO=0,
            SLOPE=1.2104,
            INTERCEPT=0.01,
            # Required categorical attributes
            TYPE="LAI",
            ALARMTYPE="Standard",
            FUNCTION="Value",
            VIRTUAL=0,
            CS="AE",
            SENSORTYPE="VOLTAGE",
            DEVUNITS="VDC",
            # Requried text attributes
            NAME="SHLH.AHU-ED.RAT",
            DESCRIPTOR="RETURN TEMP",
            NETDEVID='test-value',
            SYSTEM='test-system'
            )
        # Load raw data
        dfraw_input = pd.DataFrame(data=[input_data])
        # Create predictions from raw input data
        results_l1 = basePredictorL1.predict(dfraw_input)
        """
        
        super(SVMCL1MILESPredictor, self).__init__(classifier_filename)
        
        return None


class SVMCRBFMILESPredictor(BasePredictor):
    
    def __init__(self, classifier_filename:Union[str,bytes]):
        """inputs
        ------
        classifier_filename: (str) name of file for pickled sklearn classifier
        
        Example usage
        basePredictorRBF = SVMCRBFMILESPredictor(
            classifier_filename=SVMC_l1_classifier_filename)
        # Somehow, create some raw data
        input_data = RawInputData(
            # Required numeric attributes
            DEVICEHI=122.0,
            DEVICELO=32.0,
            SIGNALHI=10,
            SIGNALLO=0,
            SLOPE=1.2104,
            INTERCEPT=0.01,
            # Required categorical attributes
            TYPE="LAI",
            ALARMTYPE="Standard",
            FUNCTION="Value",
            VIRTUAL=0,
            CS="AE",
            SENSORTYPE="VOLTAGE",
            DEVUNITS="VDC",
            # Requried text attributes
            NAME="SHLH.AHU-ED.RAT",
            DESCRIPTOR="RETURN TEMP",
            NETDEVID='test-value',
            SYSTEM='test-system'
            )
        # Load raw data
        dfraw_input = pd.DataFrame(data=[input_data])
        # Create predictions from raw input data
        results_l1 = basePredictorRBF.predict(dfraw_input)
        """
        
        super(SVMCRBFMILESPredictor, self).__init__(classifier_filename)
        
        return None


def validate_bag(bag):
    """Determine if a bag of instances is valid. A bag is valid if the
    resulting bag has at least one instance
    inputs
    ------
    bag : (pd.DataFrame) or (scipy.sparse.csr.csr_matrix)
    outputs
    -------
    is_valid : (bool) True if the bag has one instance at least"""

    # Failure - a group has dupilcate point names are are both deleted
    # during cleaning, causing an empty array to pass to subsequent pipes
    if isinstance(bag, pd.DataFrame):
        all_L2SL = list(set(bag['TYPE'])) == ['L2SL']
        all_virtual = list(set(bag['VIRTUAL'])) == [True]

        if all_L2SL:
            print('Bag contained all L2SL instances and is skipped')
            return False
        if all_virtual:
            print('Bag contained all Virtual instances and is skipped')
            return False

    if bag.shape[0] > 0:
        return True

    return False