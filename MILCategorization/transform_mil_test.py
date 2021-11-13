# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 18:10:48 2021

@author: vorst
"""

# Python imports
import sys
import os
import time, re, pickle
import configparser
import unittest

# Third party imports
import sqlalchemy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Local imports
from transform_mil import (Transform, VocabularyText, ReplaceNone, 
                       DataFrameSelector, OneHotEncoder, RemoveAttribute, 
                       RemoveNan, SetDtypes, TextCleaner, UnitCleaner, 
                       DuplicateRemover, VirtualRemover, EncodingCategories)
    
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
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
TYPES_FILE = r'../data/clean_types.csv'
units_df = pd.read_csv(TYPES_FILE)
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
POINTNAME_VOCABULARY_FILENAME = '../data/vocab_name.txt'
DESCRIPTOR_VOCABULARY_FILENAME = '../data/vocab_descriptor.txt'

#%%

class TransformMILTest(unittest.TestCase):

    def test_cleaning_pipe(self):
    
        Insert = extract.Insert(server_name, driver_name, database_name)
    
        customer_id = 15
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
        dataset_raw = Insert.pandas_select_execute(sel)
    
        # Create 'clean' data processing pipeline
        clean_pipe = Transform.cleaning_pipeline(drop_attributes=DROP_ATTRIBUTES,
                                           nan_replace_dict=NAN_REPLACE_DICT,
                                           dtype_dict=TYPE_DICT,
                                           unit_dict=UNIT_DICT,
                                           dupe_cols=DUPE_COLS,
                                           remove_dupe=REMOVE_DUPE,
                                           replace_numbers=REPLACE_NUMBERS,
                                           remove_virtual=REMOVE_VIRTUAL,
                                           text_clean_attributes=TEXT_CLEAN_ATTRS)
    
        df = clean_pipe.fit_transform(dataset_raw)
    
        return df
    
    def test_categorical_pipe(self):
    
        Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                                driver_name='SQL Server Native Client 10.0',
                                database_name='Clustering')
    
        customer_id = 15
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
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
    
        categorical_pipe = Transform.categorical_pipeline(
            categorical_attributes=CATEGORICAL_ATTRIBUTES,
            handle_unknown='ignore',
            categories_file=CATEGORIES_FILE)
    
        df_clean = clean_pipe.fit_transform(dataset_raw)
        ohe_array = categorical_pipe.fit_transform(df_clean).toarray()
    
        # Find more about categorical pipe
        ohe = categorical_pipe.named_steps['catEncoder']
        ohe.categories # ohe.categories_ when categories='auto'
    
        return ohe_array
    
    def test_read_categories(self):
    
        # Ititialize
        categories = Transform._read_categories(CATEGORICAL_ATTRIBUTES, 
                                                CATEGORIES_FILE)
    
        replaceNone = ReplaceNone(CATEGORICAL_ATTRIBUTES)
        dataFrameSelector = DataFrameSelector(CATEGORICAL_ATTRIBUTES)
        oneHotEncoder = OneHotEncoder(categories=categories)
    
        # Get raw database
        Insert = extract.Insert(server_name,
                                driver_name,
                                database_name)
        customer_id = 15
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
        dataset_raw = Insert.pandas_select_execute(sel)
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
        df_clean1 = clean_pipe.fit_transform(dataset_raw)
    
        # Transform
        df0 = replaceNone.fit_transform(df_clean1)
        df1_array = dataFrameSelector.fit_transform(df0)
        ohearray = oneHotEncoder.fit_transform(df1_array).toarray()
    
        # Examine the transformers
        print(df0[CATEGORICAL_ATTRIBUTES].iloc[:5])
        print(df1_array[:5])
        OneHotEncoder.categories
        print("OneHotArray Columns: ", ohearray.columns)
    
        return None
    
    def test_numeric_pipe(self):
    
        Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                                driver_name='SQL Server Native Client 10.0',
                                database_name='Clustering')
    
        customer_id = 15
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
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
    
        numeric_pipe = Transform.numeric_pipeline(numeric_attributes=NUM_ATTRIBUTES)
    
        df_clean = clean_pipe.fit_transform(dataset_raw)
        df_numeric = numeric_pipe.fit_transfor(df_clean)
    
        return df_numeric
    
    def test_text_pipe(self):
    
        Insert = extract.Insert(server_name,
                                driver_name,
                                database_name)
    
        customer_id = 15
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
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
    
        # Create pipeline specifically for clustering text features
        name_vocabulary = VocabularyText.read_vocabulary_disc(POINTNAME_VOCABULARY_FILENAME)
        name_text_pipe = Transform.text_pipeline_label(attributes=['NAME'],
                                                       vocabulary=name_vocabulary)
    
        full_pipeline = Pipeline([('clean_pipe', clean_pipe),
                                  ('text_pipe', name_text_pipe),
                                  ])
    
        dataset = full_pipeline.fit_transform(dataset_raw)
    
        return dataset
    
    def test_timeself(self):
    
        Insert = extract.Insert(server_name,
                                driver_name,
                                database_name)
    
        customer_id = 15
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
        dataset_raw = Insert.pandas_select_execute(sel)
    
        removeAttribute = RemoveAttribute(Transform.drop_attributes)
        removeNan = RemoveNan(Transform.nan_replace_dict)
        setDtypes = SetDtypes(Transform.type_dict)
        textCleaner = TextCleaner(Transform._text_clean_attrs, replace_numbers=True)
        unitCleaner = UnitCleaner(Transform.unit_dict)
        duplicateRemover = DuplicateRemover(Transform.dupe_cols, remove_dupe=True)
        virtualRemover = VirtualRemover(remove_virtual=True)
    
        t0 = time.time()
        df0 = removeAttribute.fit_transform(dataset_raw)
    
        t1 = time.time()
        df1 = removeNan.fit_transform(df0)
    
        t2 = time.time()
        df2 = setDtypes.fit_transform(df1)
    
        t3 = time.time()
        df3 = textCleaner.fit_transform(df2)
    
        t4 = time.time()
        df4 = unitCleaner.fit_transform(df3)
    
        t5 = time.time()
        indicies = duplicateRemover.get_duplicate_indicies(df4, 'NAME')
        print('Duplicate names')
        print(df4['NAME'].iloc[indicies[:50]])
        df5 = DuplicateRemover.fit_transform(df4)
    
        t6 = time.time()
        virtualRemover.fit_transform(df5)
        t7 = time.time()
    
        print('RemoveAttribute : {}'.format(t1 - t0))
        print('RemoveNan : {}'.format(t2 - t1))
        print('SetDtypes : {}'.format(t3 - t2))
        print('TextCleaner : {}'.format(t4 - t3))
        print('UnitCleaner : {}'.format(t5 - t4))
        print('DuplicateRemover : {}'.format(t6 - t5))
        print('VirtualRemover : {}'.format(t7 - t6))
    
        return None
    
    def test_calc_categories_dict(self):
    
        """Generate data to find categories"""
        Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                                driver_name='SQL Server Native Client 10.0',
                                database_name='Clustering')
    
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
    
        df_clean = categories_clean_pipe.fit_transform(dataset_raw)
    
        """Calculate and save categories to be used later"""
        Encoding = EncodingCategories()
        columns = ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL',
                   'CS', 'SENSORTYPE', 'DEVUNITS']
        categories_dict = Encoding.calc_categories_dict(df_clean, columns)
        save_path = r'../data/categorical_categories.dat'
    
        Encoding.save_categories_to_disc(categories_dict, save_path)
        categories_dict1 = Encoding.read_categories_from_disc(save_path)
        for key in set((*categories_dict.keys(), *categories_dict1.keys())):
            assert(np.array_equal(categories_dict[key], categories_dict1[key]))
    
        return None
    
    def test_get_building_suffix(self):
        """Test whether a set of points is a building suffix word"""
    
        Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                                driver_name='SQL Server Native Client 10.0',
                                database_name='Clustering')
    
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(18))
        dataset_raw = Insert.pandas_select_execute(sel)
    
        # Split name variable
        token_pattern = r'\.'
        tokenizer = re.compile(token_pattern)
    
        # Keep track of words
        words = []
    
        # Split each name into tokens
        for idx, word in dataset_raw['NAME'].iteritems():
            parts = tokenizer.split(word)
            words.append(parts)
    
        # Get vocabulary
        vocabularyText = VocabularyText()
        suffix = vocabularyText.get_building_suffix(words)
        print("Suffix found : ", suffix)
    
        return None
    
    def test_get_text_vocabulary(self):
    
        """Generate data to find Vocabulary"""
        Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                                driver_name='SQL Server Native Client 10.0',
                                database_name='Clustering')
    
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
    
        df_clean = clean_pipe.fit_transform(dataset_raw)
    
        # Get vocabulary for DESCRIPTOR feature - a text feature
        vocabulary = VocabularyText\
            .get_text_vocabulary(X=df_clean,
                                 col_name='DESCRIPTOR',
                                 remove_suffix=False,
                                 max_features=80)
    
        # Sove vocabulary
        file_name = r'../data/vocab_descriptor.txt'
        VocabularyText.save_vocabulary(vocabulary, file_name)
    
        return None


#%% Main

if __name__ == '__main__':
    unittest.main()