# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:28:03 2020

@author: z003vrzk
"""

# Python imports
import sys
import os
import configparser
import unittest

# Third party imports
from scipy.sparse import csr_matrix
import numpy as np
import sqlalchemy

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from mil_load import (LoadMIL, load_mil_dataset_from_file, bags_2_si_generator, 
                      bags_2_si, legacy_numeric_transform_pipeline_MIL)
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev,
                                              Customers, 
                                              ClusteringHyperparameter, 
                                              Labeling)

# Declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
MIL_DATASET_PATH = r'../data/MIL_dataset.dat'
LoadMIL = LoadMIL(server_name, driver_name, database_name)

#%%

class MILLoadTest(unittest.TestCase):

    def test_bag_data_generator(self):
    
        bag_generator = LoadMIL.bag_data_generator(pipeline='whole',
                                                   verbose=False)
        bag = next(bag_generator)
        self.assertIsInstance(bag[0], csr_matrix)
        
        return None
    
    def test_validate_bag(self):
    
        bag_generator = LoadMIL.bag_data_generator(pipeline='whole',
                                                   verbose=False)
        bag, label = next(bag_generator)
        self.assertTrue(LoadMIL.validate_bag(bag))
    
        return None
    
    def test_gather_mil_dataset(self):
        return None
    
    def test_load_mil_dataset(self):
        
        dataset = load_mil_dataset_from_file(MIL_DATASET_PATH)
        labels = dataset['bag_labels']
        data = dataset['dataset']
        
        self.assertIsInstance(labels, np.ndarray)
        self.assertIsInstance(data, np.ndarray)
    
        return None
    
    def test_legacy_numeric_transform_pipeline_MIL(self):
        
        # Get some raw data
        Insert = extract.Insert(server_name, driver_name, database_name)
        group_id = 15
        sel = sqlalchemy.select([Points]).where(Points.group_id.__eq__(group_id))
        dfraw = Insert.pandas_select_execute(sel)
        
        # Get the legacy pipeline
        full_pipeline = legacy_numeric_transform_pipeline_MIL()
        
        # Transform data
        bag = full_pipeline.fit_transform(dfraw)
        
        # Observe output number of attributes, should be 3236 for compatability
        self.assertEquals(bag.shape[1], 3236)
            
        return None


#%% Main

if __name__ == '__main__':
    unittest.main()