# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:55:56 2021

@author: vorst
"""

# Python imports
import configparser
import os
import sys
import unittest

# Third party imports
import pandas as pd

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from svm_miles_predict import MILESEmbedding, SVMC_L1_miles, RawInputData, concept_class_filename
from mil_load import LoadMIL

# Global declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

LoadMIL = LoadMIL(server_name, driver_name, database_name)
numeric_pipeline = LoadMIL.numeric_transform_pipeline()


#%% Class definitions
    

class MILESPredictTest:
    
    def setUp(self):
        return None
        
    def test_embedding(self):
        return None
    
    def test_SVMC_L1_miles_predict(self):
        return None
    
    def test_(self):
        return None
    
    def test_(self):
        return None
    
    def test_(self):
        return None
    



#%% Main

# Construct raw data input
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
    )

# Convert raw data to dataframe
dfraw = pd.DataFrame(data=[input_data])

# Transform raw data
bag = numeric_pipeline.fit_transform(dfraw)

# Embed data
MILESEmbedder = MILESEmbedding("./miles_concept_features.dat")
embedded_data = MILESEmbedder.embed_data(clean_data)