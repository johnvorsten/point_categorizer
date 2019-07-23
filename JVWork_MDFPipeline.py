# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:29:09 2019

@author: z003vrzk
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import JVWork_MDFPipeline_SQL as jvsql

def search_databases(base_directory, idx=1, print_flag=False):
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
            databases = sorted(current_dir.glob('*.mdf'))
            if print_flag:
                print('{}Current directory is directory, name : {}'
                          .format(chr(62)*idx,current_dir))
    
            
            if len(databases) == 0: #Nothing Found
                search_databases(current_dir, idx=idx+1)
            
            else: #databases found
                logs = sorted(current_dir.glob('*.ldf'))
                
                for database_path, log_path in zip(databases, logs):
                
                    if print_flag:
                        print('{}Database found : {}'.format(chr(62)*idx,str(str(database_path))))
                        print('{}Log found : {}'.format(chr(62)*idx,str(str(log_path))))
                        
                    yield (database_path, log_path)
                    
                search_databases(current_dir, idx=idx+1)
            
        else:
            continue


"""Two ways to iterate thorugh items 

#By specific range
base_directory = r"D:\Z - Saved SQL Databases"
db_iterator = search_databases(base_directory)
for i in range(5):
    database_path, log_path = next(db_iterator)
    print('\n', database_path, '\n', log_path)
   
#Unitl iterator runs out
base_directory = r"D:\Z - Saved SQL Databases"
db_iterator = search_databases(base_directory)
for i in db_iterator:
    database_path, log_path = next(db_iterator)
    print('\n', database_path, '\n', log_path)
"""

def main():

    base_directory = r"D:\Z - Saved SQL Databases"
    db_iterator = search_databases(base_directory)
    mysql = jvsql.MySQLHandling()
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
        pointfun = pd.read_sql_table('POINTFUN', engine)
        if 'POINTACTUAL' in pointfun.columns:
            pointfun.drop(columns=['POINTACTUAL'], inplace=True)
        
        pointsen = pd.read_sql_table('POINTSEN', engine)
        netdev = pd.read_sql_table('NETDEV', engine)

        pts_dataframe = join_dataframes([pointbas, pointfun, pointsen])
        #Add identifier for each job
        pts_dataframe = pts_dataframe.join(pd.Series(
                data=[str(database_path)]*pts_dataframe.shape[0], name='DBPath'))
            
        master_csv_path = "D:\Z - Saved SQL Databases\master_pts_db.csv"
        master_csv_path_obj = Path(master_csv_path)
        if not master_csv_path_obj.is_file():
            pts_dataframe.to_csv(master_csv_path)

        with open(master_csv_path_obj, 'a') as f:
            pts_dataframe.to_csv(f, header=False)
        #TODO : Create NETDEV feature matrix? - not needed
        #TODO : append to netdev for all dataframes ? - May not need
        #TODO : Create into class?
        
        mysql.detach(database_name)
        


def join_dataframes(dataframes):
    """Joins dataframes with an outer fit, removing duplicate columns
    parameters
    ------
    dataframes : list of dataframes to merge"""
    assert type(dataframes) is list, 'Pass dataframes argument as list'
    merged_df = dataframes[0].merge(dataframes[1], on=list(set.intersection(set(dataframes[0]), set(dataframes[1]))), how='outer')
    for df in dataframes[2:]:
        merged_df = merged_df.merge(df, on=list(set.intersection(set(merged_df.columns), set(df.columns))), how='outer')
        
    return merged_df


##Attach a database
#
#mysql = jvsql.MySQLHandling()
#cursmaster, connmaster = mysql.create_master_connection()
#_pathMDF = r"D:\Z - Saved SQL Databases\44OP-112425_East_Village_North\JobDB.mdf"
#_database_name = 'PipelineDB'
#mysql.attach(_pathMDF, _database_name)
#engine, conn, cursor = mysql.create_PBDB_connection(_database_name)
#
#
#
##mysql.detach(database_name)
#
##Create dataframes for each relevant table
#pointbas = pd.read_sql_table('POINTBAS', engine)
#pointfun = pd.read_sql_table('POINTFUN', engine)
#pointsen = pd.read_sql_table('POINTSEN', engine)
#netdev = pd.read_sql_table('NETDEV', engine)
#
#mysql.detach(_database_name)
#









