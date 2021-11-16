# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 21:17:17 2020

@author: z003vrzk
"""
# Python imports
import sys
import os
import configparser

# Third party imports

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from MILCategorization import mil_load

# Globals
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

LoadMIL = mil_load.LoadMIL(server_name,
                           driver_name,
                           database_name)



#%%

if __name__ == '__main__':
    # Pipeline w/ numeric features
    bags, labels = LoadMIL.gather_mil_dataset(pipeline='whole')

    # Save
    LoadMIL.save_mil_dataset(bags, labels, numeric_feature_file)

    # Retrieve
    dataset_numeric = LoadMIL.load_mil_dataset(numeric_feature_file)

    # Pipeline w/o numeric features
    bags, labels = LoadMIL.gather_mil_dataset(pipeline='categorical')

    # Save
    LoadMIL.save_mil_dataset(bags, labels, categorical_feature_file)
    
    # Retrieve
    dataset_categorical = LoadMIL.load_mil_dataset(categorical_feature_file)


