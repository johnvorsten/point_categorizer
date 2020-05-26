# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:00:16 2019

@author: z003vrzk
"""
# Python imports
import os
from pathlib import Path
from collections import namedtuple

# Third party imports

# Local imports
from . import extract

# Local declarations
Extract = extract.Extract()

#%%


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



