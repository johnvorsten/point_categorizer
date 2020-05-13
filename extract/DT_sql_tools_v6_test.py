# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:14:49 2020

@author: z003vrzk
"""

from DT_sql_tools_v6 import SQLHandling

if __name__ == "__main__":
    
    server_name = '.\DT_SQLEXPR2008'
    driver_name = 'SQL Server Native Client 10.0'
    path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\SQLTest\JobDB.mdf"
    database_name = 'PBJobDB'
    
    mysql = SQLHandling(server_name=server_name, driver_name=driver_name)
    
    # Attach a database on path_mdf
    database_name = mysql.attach_database(path_mdf, database_name)
    
    # Connects only to database called database_name
    # Use connection and cursor objects to execute SQL if needed
    connection, cursor = mysql.connect_database(database_name) 
    
    # Read a table
    df = mysql.read_table(database_name, table_name='POINTBAS')
    
    # Read sql into a pandas table
    sql = """select top(10) *
    from {}""".format(database_name)
    query_table = mysql.pandas_read_sql(sql, database_name)
    
    # Detach database_name
    mysql.detach_database(database_name)
    
    #%% Get master connection string
    server_name = '.\DT_SQLEXPR2008'
    driver_name = 'SQL Server Native Client 10.0'
    mysql = SQLHandling(server_name=server_name, driver_name=driver_name)
    conn_str = mysql.get_master_connection_str()
    print(conn_str)
    
    #%% Get database connnection string
    server_name = '.\DT_SQLEXPR2008'
    driver_name = 'SQL Server Native Client 10.0'
    mysql = SQLHandling(server_name=server_name, driver_name=driver_name)
    database_name = 'test_db'
    conn_str = mysql.get_database_connection_str(database_name)
    print(conn_str)
    
    
    
