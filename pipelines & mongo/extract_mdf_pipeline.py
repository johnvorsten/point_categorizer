# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:29:09 2019

This module is useful for extracting information stored in SQL databases. It is
part of the 'extract' pipeline.

Methods and classes : 
search_databases : (method) Base function for module. Recursively looks through 
base_directory and copies any instances of SQL databases named JobDB.mdf and 
JobLog.ldf to a default directory on the D drive

main : (method) Iterate through all databases contained in base_directory and 
save them to a .csv file.  Only the chosen tables are saved to a .csv file : 
['POINTBAS','POINTSEN','POINTFUN']


join_dataframes : (method) Joins dataframes with an outer fit, removing 
duplicate columns parameters. Only common columns between the dataframes are 
merged


#Attach a database

mysql = jvsql.MySQLHandling()
cursmaster, connmaster = mysql.create_master_connection()
_pathMDF = r"D:\Z - Saved SQL Databases\44OP-112425_East_Village_North\JobDB.mdf"
_database_name = 'PipelineDB'
mysql.attach(_pathMDF, _database_name)
engine, conn, cursor = mysql.create_PBDB_connection(_database_name)

#mysql.detach(database_name)

#Create dataframes for each relevant table
pointbas = pd.read_sql_table('POINTBAS', engine)
pointfun = pd.read_sql_table('POINTFUN', engine)
pointsen = pd.read_sql_table('POINTSEN', engine)
netdev = pd.read_sql_table('NETDEV', engine)

mysql.detach(_database_name)


@author: z003vrzk
"""

# Third party imports
from pathlib import Path
import os
import numpy as np
import pandas as pd

# Local imports
from sql_tools import MySQLHandling

def search_databases(base_directory, 
                     idx=1, 
                     print_flag=False):
    """Base function for module. Recursively looks through base_directory
    and copies any instances of SQL databases named JobDB.mdf and JobLog.ldf
    to a default directory on the D drive
    parameters
    -------
    base_directory : base directory to look in; recursive from here
    idx : for printing (optional)
    print_flag : enable print (optional)
    output
    -------
    Database path and log database path names ("""
    
    for _dir in os.listdir(base_directory):
        
        current_dir = Path(os.path.join(base_directory, _dir))
        
        if current_dir.is_dir():
            # Search for file type in current directory. If any files in the
            # Directory match the specified file type, then create an iterable
            # of those files
            search_string = '*.mdf'
            database_paths = sorted(current_dir.glob(search_string))
            
            if print_flag:
                print('{}Current directory is directory, name : {}'
                          .format(chr(62)*idx,current_dir))
    
            if len(databases) == 0: #Nothing Found
                search_databases(current_dir, idx=idx+1)
            
            else: #databases found
                # For sql database log file (mandatory)
                log_paths = sorted(current_dir.glob('*.ldf'))
                
                for database_path, log_path in zip(database_paths, log_paths):
                
                    if print_flag:
                        print('{}Database found : {}'.format(chr(62)*idx,str(str(database_path))))
                        print('{}Log found : {}'.format(chr(62)*idx,str(str(log_path))))
                        
                    yield (database_path, log_path)
                    
                search_databases(current_dir, idx=idx+1)
            
        else:
            continue


def join_dataframes(dataframes):
    """Joins dataframes with an outer fit, removing duplicate columns
    parameters. Only common columns between the dataframes are merged
    ------
    dataframes : (list) of dataframes to merge"""
    
    assert type(dataframes) is list, 'Pass dataframes argument as list'
    
    # Merge the 0th dataframe with the 1st, to initialize the df
    common_cols = list(set.intersection(set(dataframes[0].columns), set(dataframes[1].columns)))
    merged_df = dataframes[0].merge(dataframes[1], 
                          on=common_cols, 
                          how='outer')
    
    for df in dataframes[2:]:
        common_cols = list(set.intersection(set(merged_df.columns), set(df.columns)))
        merged_df = merged_df.merge(df, 
                                    on=common_cols, 
                                    how='outer')
        
    return merged_df


def main(base_directory=r"D:\Z - Saved SQL Databases",
         csv_save_path=r".\data\master_pts_db.csv"):
    """Iterate through all databases contained in base_directory and save
    them to a .csv file.  Only the chosen tables are saved to a .csv file : 
        ['POINTBAS','POINTSEN','POINTFUN']"""

    db_iterator = search_databases(base_directory)
    mysql = MySQLHandling()
    cursmaster, connmaster = mysql.create_master_connection()
    
    for database_path, log_path in db_iterator:
        print('\n', database_path, '\n', log_path)
        
        database_name = 'PipelineDB'
        
        try:
            mysql.attach(str(database_path), database_name)
        except:
            a = input('Issue connecting to {}. Skip and continue? (yes/no)'.format(database_path))
            if a == 'yes':
                continue
            else:
                return
            
        engine, conn, cursor = mysql.create_PBDB_connection(database_name)
        
        pointbas = pd.read_sql_table('POINTBAS', engine)
        pointsen = pd.read_sql_table('POINTSEN', engine)
        netdev = pd.read_sql_table('NETDEV', engine)
        pointfun = pd.read_sql_table('POINTFUN', engine)
        
        if 'POINTACTUAL' in pointfun.columns:
            pointfun.drop(columns=['POINTACTUAL'], inplace=True)

        pts_dataframe = join_dataframes([pointbas, pointfun, pointsen])
        
        # Add identifier for each job
        pts_dataframe = pts_dataframe.join(pd.Series(
                data=[str(database_path)]*pts_dataframe.shape[0], name='DBPath'))
        
        csv_save_path = Path(csv_save_path)
        # If the .csv file does not exist then create it
        if not csv_save_path.is_file():
            pts_dataframe.to_csv(csv_save_path)
        
        # Append data to existing .csv if the file exists
        with open(csv_save_path, 'a') as f:
            pts_dataframe.to_csv(f, header=False)
        
        mysql.detach(database_name)


if __name__ == '__main__':
    main(base_directory=r"D:\Z - Saved SQL Databases",
         csv_save_path=r".\data\master_pts_db.csv")




