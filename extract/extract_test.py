# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:00:16 2019

@author: z003vrzk
"""
# Python imports
import os, sys
from pathlib import Path
from collections import namedtuple
import unittest
import configparser

# Third party imports
import pandas as pd
import numpy as np
from sqlalchemy.sql import text as sqltext

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)
from extract import extract

# Local declarations
Extract = extract.Extract()

config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
numeric_feature_file = config['sql_server']['DEFAULT_NUMERIC_FILE_NAME']
categorical_feature_file = config['sql_server']['DEFAULT_CATEGORICAL_FILE_NAME']

global Insert;
Insert = extract.Insert(server_name=server_name,
                        driver_name=driver_name,
                        database_name=database_name)
        
#%%

class InsertTest(unittest.TestCase):
    
    def setUp(self):
        # The Insert class is used for inserting data into a SQL table,
        # And also retrieving data from the collection of tables

        return
    
    def test_clean_dataframe(self,):
        """Delete duplicated columns and replace np.nan values with None
        None is mapped to Null in SQL"""
        df = pd.DataFrame({'ColA':[1,2,3,np.nan], 
                           'ColA':[np.nan, 1,2, np.nan],
                           'ColB':[1,2,3,np.nan]})
        df = extract.Insert.clean_dataframe(df)
        
        self.assertNotIn(True, df.columns.duplicated())
        self.assertNotIn(np.nan, df.values)
        return None
    
    def test_core_select_execute(self):
        """This assumes there is a database reachable with data stored.."""
        
        sql = """SELECT name, *
        FROM {}""".format('sys.master')
        
        sel = sqltext(sql)
        Insert.core_select_execute(sel)
        
        return
    
    def test_pandas_select_execute(self):
        return
    
    def test_(self):
        return
    def test_(self):
        return
    def test_(self):
        return


if __name__ == '__main__':
    # UNC Paths mapped
    UNCMapper = extract.UNCMapping()
    drives = UNCMapper.get_UNC()
    for letter, unc in drives.items():
        print(letter, ": ", unc)

    # Searching for databases
    search_directory = r'D:\SQLTest'
    for mdf_path, ldf_path in Extract.search_databases(search_directory,
                                                       print_flag=True):
        print('Found : {}\nFound : {}'.format(mdf_path, ldf_path))

    # Search database names
    path_iterator = Extract.search_databases(search_directory)
    mdf_path, ldf_path = next(path_iterator)

    # Generate folder name auto
    path_mdf = r"D:\SQLTest\JobDB.mdf"
    save_directory = r'D:\SQLTest\save-database-test'
    Search = extract.Extract()

    folder_name = Search._get_folder_name(str(path_mdf), save_directory=save_directory)
    print(folder_name)

    # Saving a database
    search_directory = r'D:\SQLTest'
    save_directory = r'D:\SQLTest\save-database-test'
    Search = extract.Extract()

    Search.search_and_save(search_directory, save_directory)
    for _dir in os.listdir(save_directory):
        print(_dir)

    # Test retrieving dataframes
    server_name='.\DT_SQLEXPR2008'
    driver_name='SQL Server Native Client 10.0'
    database_name='PipelineDB'
    search_directory=r"D:\Z - Saved SQL Databases"
    df_iterator = Extract.iterate_dataframes(server_name,
                                         driver_name,
                                         database_name,
                                         search_directory)
    points_df, netdev_df, mdf_path = next(df_iterator)

    cols = {}
    for col in points_df.columns:
        try:
            cols[col] += 1
        except KeyError:
            cols[col] = 1

    for key, val in cols.items():
        if val >= 2:
            print(key, val)

    for col in points_df.columns:
        col_type = points_df[col].dtypes
        print(col_type)


    # Create tables
    from SQLAlchemyDataDefinition import Points, Netdev, Customers, create_tables
    server_name = '.\DT_SQLEXPR2008'
    driver_name = 'SQL Server Native Client 10.0'
    database_name = 'Clustering'
    result = create_tables(server_name, driver_name, database_name)



