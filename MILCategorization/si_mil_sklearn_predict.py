# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:13:22 2021

@author: vorst
"""

# Python imports
from typing import Union, List
import pickle
import configparser
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third party imports
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from bag_cross_validate import BagScorer, bags_2_si

# Local imports
from transform_mil import Transform
from svm_miles_predict import RawInputData

# Global declarations
CompNB_classifier_filename = r"./compNB_si.clf"
KNN_classifier_filename = r"./knn_si.clf"
MultiNB_classifier_filename = r"./multiNB_si.clf"
SVMCL1SI_classifier_filename = r"./svmc_l1_si.clf"
SVMCRBFSI_classifier_filename = r"./svmc_rbf_si.clf"


#%% Classses and data

class BasePredictor(ABC):
    
    @abstractmethod
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str):
        """inputs
        ------
        classifier_filename: (str) name of file for pickled sklearn classifier
        classifier_type: (str) one of ['numeric','categorical'] for numeric or 
        categorical pipelines depening on requirements of the classifier
        
        Example usage
        basePredictorL1 = BasePredictor(
            classifier_filename=SVMCL1SI_classifier_filename)
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
        self.pipeline = self._load_pipeline(pipeline_type)

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
        bag_prediction: (np.ndarray) results of aggregation with single-instance
        inference of a bags label from the instances within the bag"""
        # Transform raw data
        transformed_data = self._transform_data(data)
        # Predict on transformed data
        predictions = self.classifier.predict(self._determine_reshape(transformed_data))
        # Add aggregation of prediction
        bag_prediction = BagScorer.reduce_bag_label(predictions, method='mode')
        
        return np.array(bag_prediction, dtype=np.unicode_)

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
        clean_data (np.ndarray or scipy.sparse.csr_array) results of data
        passed through transformation pipeline
        """
        
        df_raw = pd.DataFrame(data)
        clean_data = self.pipeline.fit_transform(df_raw)
        
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
            return data
        else:
            msg="Data passed with more than 2 dimensions. Got {}"
            raise ValueError(msg.format(data.ndim))
        return data

    def _load_pipeline(self, classifier_type:str) -> Pipeline:
        """Determine which pipeline to load based on the chosen classifier
        and passed classifier_type
        inputs
        ------
        classifier_type: (str) one of ['numeric','categorical'] for numeric or 
        categorical pipelines depening on requirements of the classifier
        If 'numeric' then return Transform.numeric_transform_pipeline_MIL()
        or Transform.categorical_transform_pipeline_MIL()"""
        
        if classifier_type == 'numeric':
            return Transform.numeric_transform_pipeline_MIL()
        elif classifier_type == 'categorical':
            return Transform.categorical_transform_pipeline_MIL()
        else:
            msg=("classifier_type must be one of ['numeric','categorical']. " +
                 "Got {}")
            raise ValueError(msg.format(classifier_type))
        
        return None



class CompNBPredictor(BasePredictor):
    
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str):
        """inputs
        ------
        classifier_filename: (str) name of file for pickled sklearn classifier
        classifier_type: (str) one of ['numeric','categorical'] for numeric or 
        categorical pipelines depening on requirements of the classifier
        
        Example usage
        basePredictorL1 = BasePredictor(
            classifier_filename=SVMCL1SI_classifier_filename)
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
        if pipeline_type != ''
        self.pipeline = self._load_pipeline(pipeline_type)

        return None
    

class KNNPredictor(BasePredictor):
    """KNN Predictors support dense features"""
    pass

class MultiNBPredictor(BasePredictor):
    """Multinomial Native Bayes estimators support sparse features and 
    categorical data"""
    pass

class CompNBPredictor(BasePredictor):
    """Complement Native Bayes estimators support sparse features and 
    categorical data"""
    pass

class SVMCL1SIPredictor(BasePredictor): 
    """SVMC Predictors support dense features"""
    pass

class SVMCRBFSIPredictor(BasePredictor):  
    """SVMC Predictors support dense features"""
    pass



