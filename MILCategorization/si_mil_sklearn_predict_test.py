# -*- coding: utf-8 -*-
"""
Created on Wed 2021-11-24
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
import hashlib
from copy import deepcopy

# Third party imports
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
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
from si_mil_sklearn_predict import (KNNPredictor, MultiNBPredictor, 
                                    CompNBPredictor, SVMCL1SIPredictor, 
                                    SVMCRBFSIPredictor, BasePredictor)
from mil_load import LoadMIL
from transform_mil import Transform
from dataclass_serving import RawInputData, RawInputDataPydantic

# Global declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
CompNB_classifier_filename = r"./compNB_si.clf"
KNN_classifier_filename = r"./knn_si.clf"
MultiNB_classifier_filename = r"./multiNB_si.clf"
SVMCL1SI_classifier_filename = r"./svmc_l1_si.clf"
SVMCRBFSI_classifier_filename = r"./svmc_rbf_si.clf"

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
        data_list[i].NAME = "BUILDING.Name"+str(i)+".Equipment"
    return input_data

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
        data_list[i].NAME = "BUILDING.Name"+str(i)+".Equipment"
    return input_data_pydantic

class BasePredictorTest(unittest.TestCase):

    def setUp(self):
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
            SYSTEM='test-system'
            )
        
        # Convert raw data to dataframe
        self.dfraw_input = pd.DataFrame(data=[input_data])
        
        # Load raw data from file or create raw data
        (self.dfraw_load, self.bag_load, 
         self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='whole')
        
        return None
    
    def test__load_predictor(self):
        
        classifierCompNB = BasePredictor._load_predictor(CompNB_classifier_filename)
        classifierKNN = BasePredictor._load_predictor(KNN_classifier_filename)
        classifierMultiNB = BasePredictor._load_predictor(MultiNB_classifier_filename)
        classifierSVMCL1 = BasePredictor._load_predictor(SVMCL1SI_classifier_filename)
        classifierSVMCRBF = BasePredictor._load_predictor(SVMCRBFSI_classifier_filename)
        
        self.assertIsInstance(classifierCompNB, ComplementNB)
        self.assertIsInstance(classifierKNN, KNeighborsClassifier)
        self.assertIsInstance(classifierMultiNB, MultinomialNB)
        self.assertIsInstance(classifierSVMCL1, LinearSVC)
        self.assertIsInstance(classifierSVMCRBF, SVC)

        return None


class CompNBPredictorTest(unittest.TestCase):
    
    def setUp(self):
        self.predictor =  CompNBPredictor(
            classifier_filename=CompNB_classifier_filename,
            pipeline_type='categorical')
        
        # Generate some data from dataclass / pydantic for FastAPI
        self.input_data = generate_input_data()
        self.input_data_pydantic = generate_input_data_pydantic()
        self.input_data_list = [self.input_data] * 10
        self.input_data_list_pydantic = [self.input_data_pydantic] * 10
        # Convert raw data to dataframe
        self.dfraw_input = pd.DataFrame(data=[self.input_data.__dict__])
        self.dfraw_input_pydantic = pd.DataFrame(data=[self.input_data_pydantic.__dict__])
        
        # Load raw data from file or create raw data
        (self.dfraw_load, self.bag_load, 
         self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='categorical')
        
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
        for result in results:
            self.assertTrue(result in label_set)

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
        for result in results:
            self.assertTrue(result in label_set)
        return None
    
    def test__transform_data(self):
        
        # The loaded bag (self.bag_load) should match the manually transformed 
        bag_manual = Transform.categorical_transform_pipeline_MIL()\
            .fit_transform(self.dfraw_load)
        self.assertTrue(np.equal(bag_manual.toarray(), 
                                 self.bag_load.toarray()).all())
        
        # Transform raw data (from input class)
        bag_input = Transform.categorical_transform_pipeline_MIL()\
            .fit_transform(self.dfraw_input)
        msg=("The transformed bag has {} features, and the predictor "+
             "has {} features")
        print(msg.format(bag_input.shape[1], 
                         self.predictor.classifier.n_features_in_))
        self.assertEqual(bag_input.shape[1], 
                         self.predictor.classifier.n_features_in_)
        
        return None
    
    def test__transform_data_pydantic(self):
        # Transform raw data (from input class)
        bag_input_pydantic = Transform.categorical_transform_pipeline_MIL()\
            .fit_transform(self.dfraw_input_pydantic)
        msg=("The transformed bag has {} features, and the predictor "+
             "has {} features")
        print(msg.format(bag_input_pydantic.shape[1], 
                         self.predictor.classifier.n_features_in_))
        self.assertEqual(bag_input_pydantic.shape[1], 
                         self.predictor.classifier.n_features_in_)
        return None

    def test__transform_list(self):
        # Transform from a list of dictionaries
        input_data_list = self.input_data_list
        input_list = [x.__dict__ for x in input_data_list]
        dfraw = pd.DataFrame(input_list)
        bag = Transform.categorical_transform_pipeline_MIL()\
            .fit_transform(dfraw)
        # Transform from a list of dictionaries
        input_data_list_pydantic = self.input_data_list_pydantic
        input_list = [x.__dict__ for x in input_data_list_pydantic]
        dfraw = pd.DataFrame(input_list)
        bag = Transform.categorical_transform_pipeline_MIL()\
            .fit_transform(dfraw)
            
        return None

# class KNNPredictorTest(unittest.TestCase):
    
#     def setUp(self):
#         # Instance of BasePredictor
#         self.predictor =  KNNPredictor(
#             classifier_filename=KNN_classifier_filename,
#             pipeline_type='numeric')

#         # Generate some data from dataclass / pydantic for FastAPI
#         input_data = generate_input_data()
#         input_data_pydantic = generate_input_data_pydantic()
#         # Convert raw data to dataframe
#         self.dfraw_input = pd.DataFrame(data=[input_data.__dict__])
#         self.dfraw_input_pydantic = pd.DataFrame(data=[input_data_pydantic.__dict__])
        
#         # Load raw data from file or create raw data
#         (self.dfraw_load, self.bag_load, 
#          self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='whole')
        
#         return None
    
#     def test_predict(self):
#         # Test l1 Linear estimator with both types of input
#         results_input = self.predictor.predict(self.dfraw_input)
#         results_load = self.predictor.predict(self.dfraw_load)
#         # Possible labels
#         label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
#                      'misc','room','rtu','skip','unknown'}
#         # Results are a numpy array holding str
#         self.assertTrue(results_input[0] in label_set)
#         self.assertTrue(results_load[0] in label_set)
        
#         return None
    
#     def test__transform_data(self):
        
#         # The loaded bag (self.bag_load) should match the manually transformed 
#         bag_manual = Transform.numeric_transform_pipeline_MIL()\
#             .fit_transform(self.dfraw_load)
#         self.assertTrue(np.equal(bag_manual.toarray(), 
#                                  self.bag_load.toarray()).all())
        
#         # Transform raw data (from input class)
#         bag_input = Transform.numeric_transform_pipeline_MIL()\
#             .fit_transform(self.dfraw_input)
#         msg=("The transformed bag has {} features, and the predictor "+
#              "has {} features")
#         print(msg.format(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_))
#         self.assertEqual(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_)
        
#         return None


# class MultiNBPredictorTest(unittest.TestCase):
    
#     def setUp(self):
#         # Instance of BasePredictor
#         self.predictor =  MultiNBPredictor(
#             classifier_filename=MultiNB_classifier_filename,
#             pipeline_type='categorical')

#         # Generate some data from dataclass / pydantic for FastAPI
#         input_data = generate_input_data()
#         input_data_pydantic = generate_input_data_pydantic()
#         # Convert raw data to dataframe
#         self.dfraw_input = pd.DataFrame(data=[input_data.__dict__])
#         self.dfraw_input_pydantic = pd.DataFrame(data=[input_data_pydantic.__dict__])
        
#         # Load raw data from file or create raw data
#         (self.dfraw_load, self.bag_load, 
#          self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='categorical')
        
#         return None
    
#     def test_predict(self):
#         # Test l1 Linear estimator with both types of input
#         results_input = self.predictor.predict(self.dfraw_input)
#         results_load = self.predictor.predict(self.dfraw_load)
#         # Possible labels
#         label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
#                      'misc','room','rtu','skip','unknown'}
#         # Results are a numpy array holding str
#         self.assertTrue(results_input[0] in label_set)
#         self.assertTrue(results_load[0] in label_set)
        
#         return None
    
#     def test__transform_data(self):
        
#         # The loaded bag (self.bag_load) should match the manually transformed 
#         bag_manual = Transform.categorical_transform_pipeline_MIL()\
#             .fit_transform(self.dfraw_load)
#         self.assertTrue(np.equal(bag_manual.toarray(), 
#                                  self.bag_load.toarray()).all())
        
#         # Transform raw data (from input class)
#         bag_input = Transform.categorical_transform_pipeline_MIL()\
#             .fit_transform(self.dfraw_input)
#         msg=("The transformed bag has {} features, and the predictor "+
#              "has {} features")
#         print(msg.format(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_))
#         self.assertEqual(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_)
        
#         return None


# class SVMCL1SIPredictorTest(unittest.TestCase):
    
#     def setUp(self):
#         # Instance of BasePredictor
#         self.predictor =  SVMCL1SIPredictor(
#             classifier_filename=SVMCL1SI_classifier_filename,
#             pipeline_type='numeric')

#         # Generate some data from dataclass / pydantic for FastAPI
#         input_data = generate_input_data()
#         input_data_pydantic = generate_input_data_pydantic()
#         # Convert raw data to dataframe
#         self.dfraw_input = pd.DataFrame(data=[input_data.__dict__])
#         self.dfraw_input_pydantic = pd.DataFrame(data=[input_data_pydantic.__dict__])
        
#         # Load raw data from file or create raw data
#         (self.dfraw_load, self.bag_load, 
#          self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='whole')
        
#         return None
    
#     def test_predict(self):
#         # Test l1 Linear estimator with both types of input
#         results_input = self.predictor.predict(self.dfraw_input)
#         results_load = self.predictor.predict(self.dfraw_load)
#         # Possible labels
#         label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
#                      'misc','room','rtu','skip','unknown'}
#         # Results are a numpy array holding str
#         self.assertTrue(results_input[0] in label_set)
#         self.assertTrue(results_load[0] in label_set)
        
#         return None
    
#     def test__transform_data(self):
        
#         # The loaded bag (self.bag_load) should match the manually transformed 
#         bag_manual = Transform.numeric_transform_pipeline_MIL()\
#             .fit_transform(self.dfraw_load)
#         self.assertTrue(np.equal(bag_manual.toarray(), 
#                                  self.bag_load.toarray()).all())
        
#         # Transform raw data (from input class)
#         bag_input = Transform.numeric_transform_pipeline_MIL().fit_transform(self.dfraw_input)
#         msg=("The transformed bag has {} features, and the predictor "+
#              "has {} features")
#         print(msg.format(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_))
#         self.assertEqual(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_)
        
#         return None


# class SVMCRBFSIPredictorTest(unittest.TestCase):
    
#     def setUp(self):
#         # Instance of BasePredictor
#         self.predictor =  SVMCRBFSIPredictor(
#             classifier_filename=SVMCRBFSI_classifier_filename,
#             pipeline_type='numeric')

#         # Generate some data from dataclass / pydantic for FastAPI
#         input_data = generate_input_data()
#         input_data_pydantic = generate_input_data_pydantic()
#         # Convert raw data to dataframe
#         self.dfraw_input = pd.DataFrame(data=[input_data.__dict__])
#         self.dfraw_input_pydantic = pd.DataFrame(data=[input_data_pydantic.__dict__])
        
#         # Load raw data from file or create raw data
#         (self.dfraw_load, self.bag_load, 
#          self.bag_label_load) = LoadMIL.get_single_mil_bag(pipeline='whole')
        
#         return None
    
#     def test_predict(self):
#         # Test l1 Linear estimator with both types of input
#         results_input = self.predictor.predict(self.dfraw_input)
#         results_load = self.predictor.predict(self.dfraw_load)
#         # Possible labels
#         label_set = {'ahu','alarm', 'boiler','chiller','exhaust_fan',
#                      'misc','room','rtu','skip','unknown'}
#         # Results are a numpy array holding str
#         self.assertTrue(results_input[0] in label_set)
#         self.assertTrue(results_load[0] in label_set)
        
#         return None
    
#     def test__transform_data(self):
        
#         # The loaded bag (self.bag_load) should match the manually transformed 
#         bag_manual = Transform.numeric_transform_pipeline_MIL()\
#             .fit_transform(self.dfraw_load)
#         self.assertTrue(np.equal(bag_manual.toarray(), 
#                                  self.bag_load.toarray()).all())
        
#         # Transform raw data (from input class)
#         bag_input = Transform.numeric_transform_pipeline_MIL().fit_transform(self.dfraw_input)
#         msg=("The transformed bag has {} features, and the predictor "+
#              "has {} features")
#         print(msg.format(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_))
#         self.assertEqual(bag_input.shape[1], 
#                          self.predictor.classifier.n_features_in_)
        
#         return None


#%% Main

if __name__ == '__main__':
    unittest.main()