# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 20:28:19 2021

@author: vorst
"""

# Python imports
import sys
import os
import configparser

# Third party imports
import sqlalchemy
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Local imports
from transform_mil import (Transform, EncodingCategories, SetDtypes)
    
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from extract import extract
from extract.SQLAlchemyDataDefinition import (Points)
# Declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']

# Specific configuration for MIL
# Drop these attributes from being included
DROP_ATTRIBUTES = ['CTSYSNAME', 'TMEMBER', 'ALARMHIGH', 'ALARMLOW',
                   'COMBOID', 'PROOFPRSNT', 'PROOFDELAY', 'NORMCLOSE', 
                   'INVERTED', 'LAN', 'DROP', 'POINT', 'ADDRESSEXT', 'DEVNUMBER', 
                   'CTSENSTYPE', 'CONTRLTYPE', 'UNITSTYPE', 'SIGUNITS', 'NUMBERWIRE', 
                   'POWER', 'WIRESIZE', 'WIRELENGTH', 'S1000TYPE','INITVALUE']
# Not used
UNUSED = ['SLOPE', 'INTERCEPT']
# Mapping form attribute name to data type in pandas DF
TYPE_DICT = {'NAME':str, 'NETDEVID':str,
             'DESCRIPTOR':str, 'ALARMTYPE':str,
             'FUNCTION':str, 'VIRTUAL':str,
             'SYSTEM':str, 'CS':str,
             'SENSORTYPE':str, 'DEVUNITS':str}
# Replace NAN wiht these values in specific columns
NAN_REPLACE_DICT = {'NETDEVID':'empty', 'NAME':'remove',
                    'DESCRIPTOR':'empty','TYPE':'mode',
                    'ALARMTYPE':'mode',
                    'FUNCTION':'Value','VIRTUAL':'False',
                    'SYSTEM':'empty','CS':'empty',
                    'SENSORTYPE':'digital','DEVICEHI':'zero',
                    'DEVICELO':'zero','DEVUNITS':'empty',
                    'SIGNALHI':'zero','SIGNALLO':'zero',
                    'SLOPE':'zero','INTERCEPT':'zero'}
# Text attributes
TEXT_CLEAN_ATTRS = ['NAME','DESCRIPTOR','SYSTEM']

# The TYPE attribute can be many categories, but they will be reduced
# To a predefined list
TYPES_FILE = r'../data/typescorrection.csv'
units_df = pd.read_csv(TYPES_FILE, delimiter=',', encoding='utf-8', 
                       engine='python', quotechar='\'')
UNIT_DICT = {}
for idx, unit in (units_df.iterrows()):
    depreciated_value = unit['depreciated_type']
    new_value = unit['new_type']
    if new_value == '0':
        new_value = ''
    UNIT_DICT[depreciated_value] = new_value

# Remove duplicates from the NAME column
DUPE_COLS = ['NAME']
REMOVE_DUPE=True # Remove duplicates from specified attributes
# Categorical attributes in dataset that should be one-hot encoded
CATEGORICAL_ATTRIBUTES = ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS',
                          'SENSORTYPE', 'DEVUNITS']
# Numeric attributes - Scaled
NUM_ATTRIBUTES = ['DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO',
                  'SLOPE', 'INTERCEPT']
# Remove numbers from the NAME attribute, ex AHU01 -> AHU
REPLACE_NUMBERS=True
# Remove instances where the attribute 'VIRTUAL' is TRUE
REMOVE_VIRTUAL=True
# Vocabulary and categories data
CATEGORIES_FILE = r'../data/categorical_categories.dat'



#%%

def save_numpy_string_array_to_text(nparray: np.ndarray, save_path: str) -> None:
    """inputs
    -------
    nparray: (np.ndarray) of dtype string with shape [n,]
    outputs
    -------
    saved text file \r\n delimeted
    """
    with open(save_path, 'wt') as f:
        for vocab in nparray:
            f.write(vocab)
            f.write('\n')
        
    return None


def calc_save_categories_vocabulary():

    # Read raw data from database
    Insert = extract.Insert(server_name, driver_name, database_name)
    sel = sqlalchemy.select([Points])
    dataset_raw = Insert.pandas_select_execute(sel)

    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(
        drop_attributes=DROP_ATTRIBUTES,
        nan_replace_dict=NAN_REPLACE_DICT,
        dtype_dict=TYPE_DICT,
        unit_dict=UNIT_DICT,
        dupe_cols=DUPE_COLS,
        remove_dupe=REMOVE_DUPE,
        replace_numbers=REPLACE_NUMBERS,
        remove_virtual=REMOVE_VIRTUAL,
        text_clean_attributes=TEXT_CLEAN_ATTRS)
    
    string_pipe = SetDtypes(
        type_dict={'TYPE':str, 'ALARMTYPE':str,
                   'FUNCTION':str, 'VIRTUAL':str,
                   'CS':str, 'SENSORTYPE':str,
                   'DEVUNITS':str})

    categories_clean_pipe = Pipeline([
        ('clean_pipe',clean_pipe),
        ('string_pipe',string_pipe)
        ])
    
    # Process raw data with pipeline
    df_clean = categories_clean_pipe.fit_transform(dataset_raw)

    # Calculate categories to be used later
    Encoding = EncodingCategories()
    columns = ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL',
               'CS', 'SENSORTYPE', 'DEVUNITS']
    categories_dict_calc = Encoding.calc_categories_dict(df_clean, columns)
    
    # Save categories in numpy array to be used later
    Encoding.save_categories_to_disc(categories_dict_calc, CATEGORIES_FILE)
    
    # Save vocabulary to file
    VOCAB_ALARMTYPE_PATH = '../data/vocab_alarmtype.txt'
    save_numpy_string_array_to_text(categories_dict_calc['ALARMTYPE'], 
                                    VOCAB_ALARMTYPE_PATH)
    VOCAB_CS_PATH = '../data/vocab_cs.txt'
    save_numpy_string_array_to_text(categories_dict_calc['CS'], 
                                    VOCAB_CS_PATH)
    VOCAB_DEVUNITS_PATH = '../data/vocab_devunits.txt'
    save_numpy_string_array_to_text(categories_dict_calc['DEVUNITS'], 
                                    VOCAB_DEVUNITS_PATH)
    VOCAB_FUNCTION_PATH = '../data/vocab_function.txt'
    save_numpy_string_array_to_text(categories_dict_calc['FUNCTION'], 
                                    VOCAB_FUNCTION_PATH)
    VOCAB_SENSORTYPE_PATH = '../data/vocab_sensortype.txt'
    save_numpy_string_array_to_text(categories_dict_calc['SENSORTYPE'], 
                                    VOCAB_SENSORTYPE_PATH)
    VOCAB_TYPE_PATH = '../data/vocab_type.txt'
    save_numpy_string_array_to_text(categories_dict_calc['TYPE'], 
                                    VOCAB_TYPE_PATH)
    VOCAB_VIRTUAL_PATH = '../data/vocab_virtual.txt'
    save_numpy_string_array_to_text(categories_dict_calc['VIRTUAL'], 
                                    VOCAB_VIRTUAL_PATH)  
    
    return None
    


def main():
    return None
    
#%%

if __name__ == '__main__':
    main()