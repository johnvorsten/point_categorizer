# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:55:56 2021
TODO: Predict with the following inputs:
    List of RawInputData [RawInputData, RawInputData]
    Single RawInputData (without converting to dataframe)
    pandas dataframe (raw data only)
    
@author: vorst
"""

# Python imports
import configparser
import os
import sys
import unittest
from copy import deepcopy

# Third party imports
import pandas as pd
from sklearn.svm import LinearSVC, SVC
import numpy as np

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from svm_miles_predict import (MILESEmbedding, SVMCL1MILESPredictor, 
                               SVMCRBFMILESPredictor, 
                               BasePredictor, Transform, N_CONCEPT_FEATURES)
from mil_load import LoadMIL
from dataclass_serving import RawInputData, RawInputDataPydantic


# Global declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']
MILES_CONCEPT_FEATURES = "./miles_concept_features.dat"
SVMC_l1_classifier_filename = r"./svmc_l1_miles.clf"
SVMC_rbf_classifier_filename = r"./svmc_rbf_miles.clf"
    
LoadMIL = LoadMIL(server_name, driver_name, database_name)


#%% Class definitions
def generate_input_data():
    # Construct raw data input
    # This is intended to test input gathered from a web form. Not all
    # Attributes that are present in a SQL database are present
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
        SYSTEM='test-system',
        )
    data_list = [deepcopy(input_data) for x in range(10)]
    for i in range(0,10):
        data_list[i].NAME = data_list[i].NAME+str(chr(i+64))
    return input_data, data_list

def generate_input_data_pydantic():
    # Construct raw data input
    # This is intended to test input gathered from a web form. Not all
    # Attributes that are present in a SQL database are present
    input_data_pydantic = RawInputDataPydantic(          
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
            SYSTEM='test-system',
            )
    data_list = [deepcopy(input_data_pydantic) for x in range(10)]
    for i in range(0,10):
        data_list[i].NAME = data_list[i].NAME+str(chr(i+64))
    return input_data_pydantic, data_list


class PredictorTestMixin:
    
    def _generate_data(self):
        # Generate some data from dataclass / pydantic for FastAPI
        self.input_data, self.input_data_list = generate_input_data()
        self.input_data_pydantic, self.input_data_list_pydantic = generate_input_data_pydantic()
        # Convert raw data to dataframe
        self.dfraw_input = pd.DataFrame(data=[self.input_data.__dict__])
        self.dfraw_input_pydantic = pd.DataFrame(data=[self.input_data_pydantic.__dict__])
        return None
    
    def test_predict_dataframe(self):
        # Test estimator with dataframe inputs
        results_input = self.predictor.predict(self.dfraw_input)
        results_load = self.predictor.predict(self.dfraw_load)
        results_input_pydantic = self.predictor.predict(self.dfraw_input_pydantic)

        # Possible labels
        label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
                     'misc','room','rtu','skip','unknown'}
        # Results are a numpy array holding str
        self.assertTrue(results_input[0] in label_set)
        self.assertTrue(results_load[0] in label_set)
        self.assertTrue(results_input_pydantic[0] in label_set)

        return None
    
    def test_predict_pydantic_list(self):
        # Test estimator with Dataclass inputs
        # Inputs MUST be converted to dictionary or list of dictionaries before
        # Passing to .predict
        input_list = [x.__dict__ for x in self.input_data_list_pydantic]
        results = self.predictor.predict(input_list)
        # Possible labels
        label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
                     'misc','room','rtu','skip','unknown'}
        # Results are a numpy array holding str
        self.assertTrue(results[0] in label_set)

        return None
    
    def test_predict_dataclass_list(self):
        # Test estimator with Dataclass inputs
        # Inputs MUST be converted to dictionary or list of dictionaries before
        # Passing to .predict
        input_list = [x.__dict__ for x in self.input_data_list]
        results = self.predictor.predict(input_list)
        # Possible labels
        label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
                     'misc','room','rtu','skip','unknown'}
        # Results are a numpy array holding str
        self.assertTrue(results[0] in label_set)
        return None
    
    def test__transform_data_pydantic(self):
        # Transform raw data (from input class)
        bag_input_pydantic = Transform.numeric_transform_pipeline_MIL()\
            .fit_transform(self.dfraw_input_pydantic)
            
        return None

    def test__transform_list(self):
        # Transform from a list of dictionaries
        input_data_list = self.input_data_list
        input_list = [x.__dict__ for x in input_data_list]
        dfraw = pd.DataFrame(input_list)
        bag = Transform.numeric_transform_pipeline_MIL()\
            .fit_transform(dfraw)
        # Transform from a list of dictionaries
        input_data_list_pydantic = self.input_data_list_pydantic
        input_list = [x.__dict__ for x in input_data_list_pydantic]
        dfraw = pd.DataFrame(input_list)
        bag = Transform.numeric_transform_pipeline_MIL()\
            .fit_transform(dfraw)
            
        return None


class BasePredictorTest(unittest.TestCase):

    def setUp(self):
        
        # Instance of BasePredictor
        self.basePredictorL1 = BasePredictor(
            classifier_filename=SVMC_l1_classifier_filename)
        self.basePredictorRBF = BasePredictor(
            classifier_filename=SVMC_rbf_classifier_filename)
        # Generate some data from dataclass / pydantic for FastAPI
        self._generate_data()
        # Load raw data from file or create raw data
        (self.dfraw_load, self.bag_load, 
         self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='whole')
        
        return None
    
    def _generate_data(self):
        # Generate some data from dataclass / pydantic for FastAPI
        self.input_data, self.input_data_list = generate_input_data()
        self.input_data_pydantic, self.input_data_list_pydantic = generate_input_data_pydantic()
        # Convert raw data to dataframe
        self.dfraw_input = pd.DataFrame(data=[self.input_data.__dict__])
        self.dfraw_input_pydantic = pd.DataFrame(data=[self.input_data_pydantic.__dict__])
        return None
    
    def test__load_predictor(self):
        
        classifierL1 = BasePredictor._load_predictor(SVMC_l1_classifier_filename)
        classifierRBF = BasePredictor._load_predictor(SVMC_rbf_classifier_filename)
        self.assertIsInstance(classifierL1, LinearSVC)
        self.assertIsInstance(classifierRBF, SVC)
        
        return None
        
    def test__transform_data(self):
        
        # The loaded bag (self.bag_load) should match the manually transformed 
        # bag
        bag_manual = Transform.numeric_transform_pipeline_MIL().fit_transform(self.dfraw_load)
        self.assertTrue(np.equal(bag_manual.toarray(), self.bag_load.toarray()).all())
        
        # Embed data
        MILESEmbedder = MILESEmbedding(MILES_CONCEPT_FEATURES)
        embedded_data_manual = MILESEmbedder.embed_data(bag_manual)
        embedded_data_load = MILESEmbedder.embed_data(self.bag_load)
        self.assertTrue(np.equal(
            embedded_data_manual, embedded_data_load
            ).all())
        
        # Transform raw data (from input class)
        bag = Transform.numeric_transform_pipeline_MIL().fit_transform(self.dfraw_input)
        embedded_raw_bag = MILESEmbedder.embed_data(bag)
        self.assertEqual(embedded_raw_bag.shape[0], MILESEmbedder.C_features.shape[0])
        
        return None
    
    def test_predict(self):
        # Test l1 Linear estimator with both types of input
        results_l1_input = self.basePredictorL1.predict(self.dfraw_input)
        results_l1_load = self.basePredictorL1.predict(self.dfraw_load)
        # Test L2 RBF Kernel estimator with both types of input
        results_rbf_input = self.basePredictorRBF.predict(self.dfraw_input)
        results_rbf_load = self.basePredictorRBF.predict(self.dfraw_load)
        
        label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
                     'misc','room','rtu','skip','unknown'}
        
        # Results are a numpy array holding str
        self.assertTrue(results_l1_input[0] in label_set)
        self.assertTrue(results_l1_load[0] in label_set)
        self.assertTrue(results_rbf_input[0] in label_set)
        self.assertTrue(results_rbf_load[0] in label_set)
        
        return None
    

class MILESEmbeddingTest(unittest.TestCase):
    
    def setUp(self):
        
        self.MILESEmbedder = MILESEmbedding(MILES_CONCEPT_FEATURES)
        
        return None
    
    def test__load_concept_class(self):
        
        C_features = self.MILESEmbedder._load_concept_class(MILES_CONCEPT_FEATURES)
        self.assertEqual(C_features.shape[1], N_CONCEPT_FEATURES)
        
        return None
    
    def test_embed_data(self):
        
        test_bag = np.zeros((10, N_CONCEPT_FEATURES))
        embedded_bag = self.MILESEmbedder.embed_data(test_bag, sigma=5.0)
        # The embedding shape should be equal to (k,1) where k is the number
        # Of instances in the concept class
        self.assertEqual(embedded_bag.shape[0], self.MILESEmbedder.C_features.shape[0])
        
        return None
    
    def test_validate_bag_size_concept(self):
        
        # Test bag
        test_bag = np.zeros((10, N_CONCEPT_FEATURES))
        # Correct number of features
        self.assertEqual(
            self.MILESEmbedder.validate_bag_size_configuration(test_bag), True)
        # Incorrect number of features
        test_bag = np.zeros((10, 99))
        self.assertEqual(
            self.MILESEmbedder.validate_bag_size_configuration(test_bag), False)
        
        return None
    
    def test_validate_bag_size_configuration(self):
        
        # Test bag
        test_bag = np.zeros((10, N_CONCEPT_FEATURES))
        # Correct number of features
        self.assertEqual(
            self.MILESEmbedder.validate_bag_size_configuration(test_bag), True)
        # Incorrect number of features
        test_bag = np.zeros((10, 99))
        self.assertEqual(
            self.MILESEmbedder.validate_bag_size_configuration(test_bag), False)
        
        return None


class SVMCL1MILESPredictorTest(unittest.TestCase, PredictorTestMixin):
    
    def setUp(self):
        # Instance of BasePredictor
        self.predictor = SVMCL1MILESPredictor(
            classifier_filename=SVMC_l1_classifier_filename)
        # Generate some data from dataclass / pydantic for FastAPI
        self._generate_data()
        # Load raw data from file or create raw data
        (self.dfraw_load, self.bag_load, 
         self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='whole')
        
        return None
        
    def test_predict(self):
        # Test l1 Linear estimator with both types of input
        results_l1_input = self.predictor.predict(self.dfraw_input)
        results_l1_load = self.predictor.predict(self.dfraw_load)
        # Possible labels
        label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
                     'misc','room','rtu','skip','unknown'}
        # Results are a numpy array holding str
        self.assertTrue(results_l1_input[0] in label_set)
        self.assertTrue(results_l1_load[0] in label_set)
        
        return None


class SVMCRBFMILESPredictorTest(unittest.TestCase, PredictorTestMixin):
    
    def setUp(self):
        # Instance of BasePredictor
        self.predictor = SVMCRBFMILESPredictor(
            classifier_filename=SVMC_rbf_classifier_filename)
        # Generate some data from dataclass / pydantic for FastAPI
        self._generate_data()
        # Load raw data from file or create raw data
        (self.dfraw_load, self.bag_load, 
         self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='whole')
        
        return None
        
    def test_predict(self):
        # Test L2 RBF Kernel estimator with both types of input
        results_rbf_input = self.predictor.predict(self.dfraw_input)
        results_rbf_load = self.predictor.predict(self.dfraw_load)
        # Possible predictions
        label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
                     'misc','room','rtu','skip','unknown'}
        # Results are a numpy array holding str
        self.assertTrue(results_rbf_input[0] in label_set)
        self.assertTrue(results_rbf_load[0] in label_set)
        
        return None


#%% Main

if __name__ == '__main__':
    unittest.main()