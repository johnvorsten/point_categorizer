# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:14:49 2020

@author: z003vrzk
"""

# Python imports
import unittest

# local imports
from sql_tools import SQLBase

#%%

class SQLTest(unittest.TestCase):

    def test_get_pyodbc_database_connection_str(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

        # Connects only to database called database_name
        # Use connection string to execute SQL if needed
        conn_str = sqlbase.get_pyodbc_database_connection_str(database_name)
        test_str = 'DRIVER={SQL Server Native Client 11.0}; SERVER=.\\DT_SQLEXPRESS; DATABASE=PBJobDB_test; Trusted_Connection=yes;'

        self.assertEqual(conn_str, test_str)
        return None


    def test_get_pyodbc_master_connection_str(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

        # Connects only to database called database_name
        # Use connection string to execute SQL if needed
        conn_str = sqlbase.get_pyodbc_master_connection_str()
        test_str = 'DRIVER={SQL Server Native Client 11.0}; SERVER=.\\DT_SQLEXPRESS; DATABASE=master; Trusted_Connection=yes;'

        self.assertEqual(conn_str, test_str)
        return None


    def test_set_pyodbc_master_connection_str(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

        # Connects only to database called database_name
        # Use connection string to execute SQL if needed
        conn_str = sqlbase._set_pyodbc_master_connection_str()
        conn_str_test = sqlbase.get_pyodbc_master_connection_str()
        conn_str_test2 = 'DRIVER={SQL Server Native Client 11.0}; SERVER=.\\DT_SQLEXPRESS; DATABASE=master; Trusted_Connection=yes;'

        self.assertEqual(conn_str, conn_str_test, conn_str_test2)
        return None

    def test_get_sqlalchemy_connection_str(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

        sqlalchemy_str = sqlbase.get_sqlalchemy_connection_str(database_name)
        sqlalchemy_str_test = 'mssql+pyodbc://.\\DT_SQLEXPRESS/PBJobDB_test?driver={SQL Server Native Client 11.0}&trusted_connection=yes'

        self.assertEqual(sqlalchemy_str, sqlalchemy_str_test)
        return None


    def test_attach_database(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        # Connection
        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

        # Check to see if a database already exists
        (file_used_bool,
         name_used_bool,
         existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)

        """If the name is already in use then detach that database"""
        if name_used_bool:
            sqlbase.detach_database(existing_database_name)
        """If the file is already attached then detach the database"""
        if file_used_bool:
            sqlbase.detach_database(existing_database_name)

        # Attach database
        sqlbase.attach_database(path_mdf, path_ldf, database_name)

        # Execute transactions on the database to make sure its connected
        sql = """SELECT * FROM [master].[sys].[databases]"""
        with sqlbase.master_connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
            while row:
                if row.name == database_name:
                    print('Found Myself')
                    break
                row = cursor.fetchone()
        if row.name != database_name:
            raise(ValueError('Did not fine {} in query'.format(database_name)))

        # Finish and detach database
        sqlbase.detach_database(database_name)

        return None


    def test_detach_database(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

        # Check to see if a database already exists
        (file_used_bool,
         name_used_bool,
         existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)

        # Attach a database on path_mdf
        if not any((file_used_bool, name_used_bool)):
            sqlbase.attach_database(path_mdf, path_ldf, database_name)
        sqlbase.detach_database(database_name)

        # Make sure it is detached
        (file_used_bool,
         name_used_bool,
         existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)

        self.assertTrue(not any((file_used_bool, name_used_bool)))
        return None


    def test_check_existing_database(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

        # Check to see if a database already exists
        (file_used_bool,
         name_used_bool,
         existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)

        # Attach a database on path_mdf
        if not any((file_used_bool, name_used_bool)):
            sqlbase.attach_database(path_mdf, path_ldf, database_name)
        else:
            sqlbase.detach_database(existing_database_name)
            sqlbase.attach_database(path_mdf, path_ldf, database_name)

        """file_used_bool should be True if the database file is already in use
        """
        (file_used_bool,
         name_used_bool,
         existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)

        self.assertTrue(file_used_bool)
        self.assertTrue(name_used_bool)
        self.assertTrue(existing_database_name == database_name)

        sqlbase.detach_database(database_name)

        return None


    def test_read_table(self):

        server_name = '.\DT_SQLEXPRESS'
        driver_name = 'SQL Server Native Client 11.0'
        path_mdf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path_ldf = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB_Log.ldf"
        database_name = 'PBJobDB_test'

        sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)
        # Instantiate a database connection to a specific table
        (file_used_bool,
         name_used_bool,
         existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)
        if not any((file_used_bool, name_used_bool)):
            sqlbase.attach_database(path_mdf, path_ldf, database_name)
        else:
            # The database is already attached
            pass
        """Instantiate a connection to the database. This is required before
        executing SQL against a table. The SQLBase class does not automatically
        connect to a database after attaching"""
        sqlbase.init_database_connection(database_name)

        # Read a table
        sql = """SELECT * FROM [POINTBAS]"""
        df = sqlbase.pandas_execute_sql(sql)
        netdevid = df['NETDEVID'].iloc[0]
        name = df['NAME'].iloc[0]
        netdevid_test = 'JHW-R01-AHU01-PXCM71601'
        name_test = 'JHW.AHU01A.MINOAD'

        # Read sql into a pandas table
        sql = """select top(10) *
        from [{}].[dbo].[POINTBAS]""".format(database_name)
        rows = sqlbase.execute_sql(sql)
        name2 = rows[0].NAME
        netdevid2 = rows[0].NETDEVID

        self.assertTrue(name==name2==name_test)
        self.assertTrue(netdevid==netdevid2==netdevid_test)
        return None

    def test_path_equal(self):

        path1 = r"C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\panel_builder\SQLTest\JHW\JobDB.mdf"
        path2 = r"C:/Users/z003vrzk/.spyder-py3/Scripts/Work/PanelBuilder/panel_builder/SQLTest/JHW/JobDB.mdf"

        self.assertTrue(SQLBase.path_equal(path1, path2))
        return None

    # def test_attach_database_remote(self):

    #     # Connect to remote database
    #     server_name = '.\DT_SQLEXPRESS'
    #     driver_name = 'SQL Server Native Client 11.0'
    #     path_mdf = r"\\ustxca00064sto.ad001.siemens.net\grp$\FSA01313088\Jobs\Automation\Austin\Test\MDT\JobDB.mdf"
    #     path_ldf = r"\\ustxca00064sto.ad001.siemens.net\grp$\FSA01313088\Jobs\Automation\Austin\Test\MDT\JobDB_Log.ldf"
    #     database_name = 'PBJobDB_test'

    #     sqlbase = SQLBase(server_name=server_name, driver_name=driver_name)

    #     # Make sure the database isnt already attached
    #     (file_used_bool,
    #      name_used_bool,
    #      existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)
    #     if not any((file_used_bool, name_used_bool)):
    #         # Attach a database on path_mdf
    #         sqlbase.attach_database(path_mdf, path_ldf, database_name)
    #     else:
    #         # There might be a database with the name name, but not non a remote server
    #         sqlbase.detach_database(database_name)
    #         sqlbase.attach_database(path_mdf, path_ldf, database_name)

    #     # Connects only to database called database_name
    #     # Use connection and cursor objects to execute SQL if needed
    #     (file_used_bool,
    #      name_used_bool,
    #      existing_database_name) = sqlbase.check_existing_database(path_mdf, database_name)

    #     # Detach database_name
    #     sqlbase.detach_database(database_name)

    #     return None



if __name__ == '__main__':
    unittest.main()