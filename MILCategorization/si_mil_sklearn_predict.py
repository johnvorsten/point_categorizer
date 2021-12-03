# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:13:22 2021

@author: vorst
"""

# Python imports
from typing import Union, List, MutableMapping
import pickle
import configparser
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third party imports
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from bag_cross_validate import BagScorer

# Local imports
from transform_mil import Transform
from dataclass_serving import RawInputData, RawInputDataPydantic

# Global declarations
CompNB_classifier_filename = r"./compNB_si.clf"
KNN_classifier_filename = r"./knn_si.clf"
MultiNB_classifier_filename = r"./multiNB_si.clf"
SVMCL1SI_classifier_filename = r"./svmc_l1_si.clf"
SVMCRBFSI_classifier_filename = r"./svmc_rbf_si.clf"

#%% Classses and data


class BasePredictor(ABC):
    """Load a pickled scikit-learn estimator, transform raw data, and 
    predict on transformed data using the loaded estimator."""
    
    @abstractmethod
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str):
        """Load a pickled scikit-learn estimator, transform raw data, and 
        predict on transformed data using the loaded estimator.
        inputs
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

    def predict(self, data:Union[List[MutableMapping], 
                                 MutableMapping, 
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
        predictions = self.classifier.predict(
            self._determine_reshape(
                self.custom_transform(transformed_data)))
        # Add aggregation of prediction
        bag_prediction = BagScorer.reduce_bag_label(predictions, method='mode')
        
        return np.array([bag_prediction], dtype=np.unicode_)

    def _transform_data(self, data:Union[List[MutableMapping], 
                                         MutableMapping, 
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
        # Convert list or dictionary to dataframe
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

    def custom_transform(self, data:Union[csr_matrix, np.ndarray]) -> Union[csr_matrix, np.ndarray]:
        """Custom transformer for implementation by derived class. If this
        method is not reimplemented then data is passed through and returned.
        Reimplement this method for custom behavior
        inputs
        ------
        data: (scipy.sparse.csr_matrix, np.ndarray) incoming data after being 
        transformed by pipeline object"""
        return data


class DensifyMixin:
    """Mixin which densifies (converts to dense input from a sparse array)
    the input to a predictor"""
    
    @classmethod
    def custom_transform(cls, data:Union[csr_matrix, np.ndarray]) -> Union[csr_matrix, np.ndarray]:
        """Convert a Numpy array of sparse bags into an array of dense bags
        inputs
        -------
        X: (scipy.sparse.csr_matrix) of shape (n) where n is the total number of 
            bags in the dataset. Each entry of X is of shape 
            (n_instances, n_features) where n_instances is the number of instances
            within a bag, and n_features is the features space of instances.
            n_instances can vary per bag
        outputs
        -------
        dense_bags: (np.ndaray, dtype='object') of shape (n) where n is the total 
            number of bags in the dataset. Each object is a dense numpy array
            of shape (n_instances, n_features). n_instances can vary per bag"""
        
        if isinstance(data, csr_matrix) and data.ndim == 2:
            return cls._densify_single_bag(data)
        
        elif isinstance(data, np.ndarray) and \
            data.ndim == 1 and \
            isinstance(data[0], csr_matrix):
            # Expect an array of objects, and each subobject is a sparse array
            return cls._densify_multiple_bags
        
        else:
            msg=("Incorrect input format. Must be either csr_matrix with 2 "+
                 "dimensions or np.ndarray of sparse arrys")
            raise ValueError(msg)
        
        return None
    
    @staticmethod
    def _densify_multiple_bags(data):
        """Convert a Numpy array of sparse bags into an array of dense bags
        inputs
        -------
        X: (scipy.sparse.csr_matrix) of shape (n) where n is the total number of 
            bags in the dataset. Each entry of X is of shape 
            (n_instances, n_features) where n_instances is the number of instances
            within a bag, and n_features is the features space of instances.
            n_instances can vary per bag
        outputs
        -------
        dense_bags: (np.ndaray, dtype='object') of shape (n) where n is the total 
            number of bags in the dataset. Each object is a dense numpy array
            of shape (n_instances, n_features). n_instances can vary per bag"""
        if not isinstance(data[0], csr_matrix):
            msg="Input must be of type scipy.sparse.csr_matrix. Got {}".format(type(data))
            raise ValueError(msg)
        if data.ndim != 1:
            msg="Input must have single outer dimension. Got {} dims".format(data.ndim)
            raise ValueError(msg)
        
        # Convert sparse bags to dense bags
        n_bags = data.shape[0]
        dense_bags = np.empty(n_bags, dtype='object')
        for n in range(n_bags):
            dense_bags[n] = data[n].toarray()
        return dense_bags

    @staticmethod
    def _densify_single_bag(data:csr_matrix) -> np.ndarray:
        return data.toarray()


class CompNBPredictor(BasePredictor):
    
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str) -> None:
        # Validate input pipeline type for specific predictor
        # Native Bayes predictors use categorical & sparse inputs
        # KNN and SVM predictors use numeric & dense inputs
        if pipeline_type != 'categorical':
            msg=("Pipeline type for this pickled estimator must be "+
                 "'categorical'. Got {}")
            raise ValueError(msg.format(pipeline_type))
        # Base class loads predictor and pipeline
        super(CompNBPredictor, self).__init__(classifier_filename, pipeline_type)
        
        return None
    

class KNNPredictor(DensifyMixin, BasePredictor):
    """KNN Predictors support dense features"""
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str) -> None:
        # Validate input pipeline type for specific predictor
        # Native Bayes predictors use categorical & sparse inputs
        # KNN and SVM predictors use numeric & dense inputs
        if pipeline_type != 'numeric':
            msg=("Pipeline type for this pickled estimator must be "+
                 "'numeric'. Got {}")
            raise ValueError(msg.format(pipeline_type))
        # Base class loads predictor and pipeline
        super(KNNPredictor, self).__init__(classifier_filename, pipeline_type)
        
        return None


class MultiNBPredictor(BasePredictor):
    """Multinomial Native Bayes estimators support sparse features and 
    categorical data"""
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str) -> None:
        # Validate input pipeline type for specific predictor
        # Native Bayes predictors use categorical & sparse inputs
        # KNN and SVM predictors use numeric & dense inputs
        if pipeline_type != 'categorical':
            msg=("Pipeline type for this pickled estimator must be "+
                 "'categorical'. Got {}")
            raise ValueError(msg.format(pipeline_type))
        # Base class loads predictor and pipeline
        super(MultiNBPredictor, self).__init__(classifier_filename, pipeline_type)
        
        return None
    

class SVMCL1SIPredictor(DensifyMixin, BasePredictor): 
    """SVMC Predictors support dense features"""
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str) -> None:
        # Validate input pipeline type for specific predictor
        # Native Bayes predictors use categorical & sparse inputs
        # KNN and SVM predictors use numeric & dense inputs
        if pipeline_type != 'numeric':
            msg=("Pipeline type for this pickled estimator must be "+
                 "'numeric'. Got {}")
            raise ValueError(msg.format(pipeline_type))
        # Base class loads predictor and pipeline
        super(SVMCL1SIPredictor, self).__init__(classifier_filename, pipeline_type)
        
        return None


class SVMCRBFSIPredictor(DensifyMixin, BasePredictor):  
    """SVMC Predictors support dense features"""
    def __init__(self, 
                 classifier_filename:Union[str,bytes],
                 pipeline_type:str) -> None:
        # Validate input pipeline type for specific predictor
        # Native Bayes predictors use categorical & sparse inputs
        # KNN and SVM predictors use numeric & dense inputs
        if pipeline_type != 'numeric':
            msg=("Pipeline type for this pickled estimator must be "+
                 "'numeric'. Got {}")
            raise ValueError(msg.format(pipeline_type))
        # Base class loads predictor and pipeline
        super(SVMCRBFSIPredictor, self).__init__(classifier_filename, pipeline_type)
        
        return None
