"""
Created on Thu Mar 28 14:33:12 2019

A few utility functions for interacting with MSSQL Server databases

This module has a few important assumptions :
1) Use trusted connection (windows authentication - not user & password)

@author: z003vrzk
"""

# Python imports
from datetime import datetime

# Third party imports
import pyodbc
import subprocess
from pathlib import Path
import os
import pandas as pd

# Local imports

# Setup logging
import logging
_log_dir = os.path.join(os.getcwd(), 'logs')
if os.path.isdir(_log_dir):
    pass
else:
    try:
        os.mkdir(_log_dir)
    except FileNotFoundError:
        os.mkdir(os.path.split(_log_dir)[0])
        os.mkdir(_log_dir)

logging.basicConfig(filename=os.path.join(_log_dir, 'sql_logs.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')


#%%

class DepreciationError(Exception):
    pass

class NameUsedError(Exception):
    pass

class SQLBase:


    def __init__(self, server_name, driver_name):
        """A helper class for sql databases. This incldues attaching, detaching,
        and connecting to databases with an sql server. This method only
        supports microsoft authentication (not user and password)
        inputs
        -------
        server_name : (str) name of sql server
        driver_name : (str) type of driver"""

        if server_name is None:
            msg='server_name cannot be {}, try ".\DT_SQLEXPR2008" or ".\DT_SQLEXPRESS"'
            raise(ValueError(msg.format(server_name)))
        else:
            self.server_name = server_name

        if driver_name is None:
            driver_1 = "{SQL Server Native Client 10.0}"
            driver_2 = "{SQL Server Native Client 11.0}"
            msg='driver_name cannot be {}, try {} or {}'
            raise(ValueError(msg.format(driver_name, driver_1, driver_2)))
        else:
            self.driver_name = '{{{driver_name}}}'.format(driver_name=driver_name)

        self._init_master_connection()

        return None


    def _init_master_connection(self):
        """Initialize a master connection on startup to test SQL Server
        connectivity"""
        if 'master_connection_str' not in self.__dict__:
            self._set_pyodbc_master_connection_str()

        try:
            self.master_connection = pyodbc.connect(self.master_connection_str)
        except Exception as e:
            logging.debug(e)
            raise(e)
        return None


    def init_database_connection(self, database_name):
        """Initialize a database connection"""
        connection_str = self.get_pyodbc_database_connection_str(database_name)
        try:
            self.database_connection = pyodbc.connect(connection_str)
        except Exception as e:
            logging.debug(e)
            raise(e)
        return None


    def get_pyodbc_master_connection_str(self):
        """Return the master database connection string"""

        if 'master_connection_str' not in self.__dict__:
            self._set_pyodbc_master_connection_str()

        return self.master_connection_str


    def _set_pyodbc_master_connection_str(self):
        """Set the master database connection string"""
        pyodbc_base='DRIVER={}; SERVER={}; DATABASE=master; Trusted_Connection=yes;'
        self.master_connection_str = pyodbc_base.format(self.driver_name, self.server_name)
        return self.master_connection_str


    def get_pyodbc_database_connection_str(self, database_name):
        """Return a database specific connection string
        inputs
        -------
        database_name : (str) name of database to connect to"""

        driver_name = self.driver_name
        server_name = self.server_name

        # For pyodbc connection only
        connection_string= 'DRIVER={}; SERVER={}; DATABASE={}; Trusted_Connection=yes;'\
                                .format(driver_name,
                                        server_name,
                                        database_name)

        return connection_string


    def get_sqlalchemy_connection_str(self, database_name):
        """Engine connection string for SQL Alchemy"""

        server_name = self.server_name
        driver_name = self.driver_name

        """ Example
        conn_str = 'mssql://.\\DT_SQLEXPR2008/PBJobDB?trusted_connection=yes&driver=SQL+Server+Native+Client+10.0'

        'mssql://.\\DT_SQLEXPR2008/PBJobDB?trusted_connection=yes&driver=SQL+Server+Native+Client+10.0'
        """

        engine_str = r'mssql+pyodbc://{0}/{1}?driver={2}&trusted_connection=yes'\
                        .format(server_name, database_name, driver_name)

        return engine_str


    def attach_database(self,
                        path_mdf,
                        path_ldf,
                        database_name):
        """Used to attach 'database_name' to the sql server. If path_ldf is
        not defined, then this method assumes the log file is in the same
        directory as path_mdf, with the file name 'JobDB_Log.ldf'. Define
        path_ldf if the file name is different

        inputs
        -------
        path_mdf : (str) user specified path to .MDF file. LDF file
        must be in same directory.
        path_ldf : (str or Path) path to database log file
        database_name : (str) name of database to attach. This can be any user
            specified name, but it must be used to attach, detach, and connect to
            a specific database
        output
        -------
        database_name : (str) of actual database name attached as
        """

        # Check if database connection is already instantiated
        if 'master_connection' not in self.__dict__:
            self._init_master_connection()

        try:
            """Check if the selected file is already in use or if the requested
            # database name is already used"""
            (file_used_bool,
             name_used_bool,
             existing_database_name) = self.check_existing_database(path_mdf, database_name)

            if file_used_bool:
                msg = ('File name: {} is already in use\n' +
                       'Connect to existing database instead\n' +
                       'Existing Database : {}'.format(existing_database_name))
                logging.info(msg.format(path_mdf))

                raise FileExistsError(msg)

            # An existing database already is using the requested name
            if name_used_bool:
                str1 = 'Database name: {} is already in use.'.format(database_name)
                now = datetime.now()
                database_name = database_name + now.strftime('%Y%m%d%H%M%S')
                str2 = ' Try a new name instead'.format(database_name)
                msg = str1 + ' ' + str2
                logging.info(msg)

                raise NameUsedError(msg)

            # Flag 1807 must be ON to connect remote databases
            path_is_net_drive = not os.path.splitdrive(path_mdf)[0] in ['C:', 'D:']
            path_is_network_name = os.path.splitdrive(path_mdf)[0].startswith('\\')
            if any((path_is_net_drive, path_is_network_name)):
                self.traceon1807(True)

            _dat_name = database_name + '_dat'
            _log_name = database_name + '_log'
            sql = """CREATE DATABASE [{database_name}]
                    ON PRIMARY (NAME={_dat_name}, FILENAME='{path_mdf}')
                    LOG ON (NAME={_log_name}, FILENAME='{path_ldf}')
                    FOR ATTACH WITH FILESTREAM (DIRECTORY_NAME='Test');
                    """.format(database_name=database_name,
                    path_mdf=path_mdf,
                    path_ldf=path_ldf,
                    _dat_name=_dat_name,
                    _log_name=_log_name)

            self.master_connection.autocommit = True
            with self.master_connection.cursor() as master_cursor:
                master_cursor.execute(sql)
            self.master_connection.autocommit = False

            logging.info('Database : {} connected'.format(database_name))

            if any((path_is_net_drive, path_is_network_name)):
                self.traceon1807(False)

        except Exception as e:
            logging.debug(e)
            raise(e)

        return None


    def detach_database(self, database_name):
        """Used to detach database_name.  Use
        this once I get the information needed from the database.  In addition,
        close the cursor associated with the connection

        inputs
        -------
        database_name : (str) name of database to detach
        """

        detach_str = """USE [master]
        GO
        ALTER DATABASE [{db_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE
        GO
        EXEC master.dbo.sp_detach_db @dbname = N'{db_name}', @skipchecks = 'false'
        GO""".format(db_name=database_name)

        # Check if detach script is made yet
        file_tail = Path('DetachDatabase.sql')
        detach_file = os.path.join(os.getcwd(), file_tail)
        with open(detach_file, mode='w') as f:
            f.write(detach_str)

        subprocess.call(['sqlcmd', '-S', self.server_name, '-i', detach_file])
        logging.info('Database {} removed'.format(database_name))

        return None


    def check_existing_database(self, path_mdf, database_name):
        """Check two conditions :
        1) A database with 'database_name' is already connected
        to the instance of sql server
        2) A database physical_name (operating system file name) is already
        connected to the server instance

        inputs
        -------
        path_mdf : (str) path to master data file .mdf
        database_name : (str) name of database you will try to connect as.
        It is the logical name of the database
        ouputs
        -------
        (file_used_bool, name_used_bool, existing_database_name)
        file_used_bool : (bool) True if the operating system file is already
            attached by the server
        name_used_bool : (bool) True if the database name requested is already
            in use by another database
        existing_database_name : (str) name of existing database if
            file_used_bool is true
        """

        sql = """select [name] as logical_name, physical_name,
        	(select name
        	from [master].[sys].[databases] as t2
        	where t2.database_id = t1.database_id) as database_name
        from sys.master_files as t1"""
        with self.master_connection.cursor() as master_cursor:
            master_cursor.execute(sql)
            rows = master_cursor.fetchall()

        # Assume the file and database name is not initially used
        file_used_bool = False
        name_used_bool = False
        existing_database_name = None

        """If the query indicates the .mdf file name OR database name is already
        Used then change the boolean state"""
        for row in rows:
            if self.path_equal(row.physical_name, path_mdf):
                file_used_bool = True
                existing_database_name = row.database_name
            if row.database_name == database_name:
                name_used_bool = True
                existing_database_name = row.database_name

        return (file_used_bool, name_used_bool, existing_database_name)

    @staticmethod
    def path_equal(path1, path2):
        """Test path equality cross operating system. This is required instead
        of os.path.samefile because SQL .mdf files (especially the master)
        cannot be operated on and file properties examined
        inputs
        -------
        path1 : (str) path like file name
        path2 : (str) path like file name
        returns
        -------
        (bool) True if paths are the same"""

        path1 = os.path.normpath(os.path.normcase(path1))
        path2 = os.path.normpath(os.path.normcase(path2))

        return path1 == path2

    def pandas_execute_sql(self, sql_query):
        """Read a table to dataframe using pyodbc and pandas. The server and driver
        used to instantiate the class is used (self.server_name, self.driver_name)
        inputs
        -------
        sql_query : (str) sql string to execute
        outputs
        -------
        df : (pandas.DataFrame) SQL table"""

        if not 'database_connection' in self.__dict__:
            msg='No database connection initialized. Try self.init_database_connection'
            raise NameError(msg)

        # For pyodbc connection only
        try:
            with self.database_connection as connection:
                df = pd.read_sql(sql_query, connection)
        except Exception as e:
            logging.debug(e)
            raise(e)

        return df

    def execute_sql(self, sql_query):
        """Execute a SQL statement and return rows
        inputs
        -------
        sql_query : (str) sql string to execute
        outputs
        -------
        rows : ()
        """
        if not 'database_connection' in self.__dict__:
            msg='No database connection initialized. Try self.init_database_connection'
            raise NameError(msg)

        try:
            with self.database_connection.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchall()
        except Exception as e:
            logging.debug(e)
            raise(e)

        return rows

    def execute_sql_master(self, sql_query):
        """Execute a SQL statement against system databases only
        (uses the master database)
        inputs
        -------
        sql_query : (str) sql string to execute
        outputs
        -------
        rows : (list) of
        """

        try:
            with self.master_connection.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchall()
        except Exception as e:
            logging.debug(e)
            raise(e)

        return rows

    def get_UNC(self):
        """Return a users mapped network drives. UNC path will be used for
        connecting to networked database"""
        output = subprocess.run(['net', 'use'], stdout = subprocess.PIPE).stdout #Bytes
        output = output.decode() #string
        alphabet = [chr(i) for i in range(65,91)]
        drives = []
        for letter in alphabet:
            if output.__contains__(letter + ':'):
                drives.append(letter)

        output = subprocess.run(['net', 'use'], stdout = subprocess.PIPE).stdout #Bytes
        output = output.decode() #string

        alphabet = [chr(i) for i in range(65,91)]
        drives = []
        for letter in alphabet:
            if output.__contains__(letter + ':'):
                drives.append(letter)

        # Get UNC server names
        output = output.splitlines()
        serverUNC = []
        for lines in output:
            if lines.__contains__('\\'):
                serverUNC.append(lines[lines.index('\\'):len(lines)-1])
        myOutput = {}
        for index, letter in enumerate(drives):
            myOutput[letter] = serverUNC[index]
        return myOutput


    def traceon1807(self, Flag):
        """Turn on/off Trace Flag 1807 based on user input True or False
        Parameters
        ----------
        Flag : (bool) True for turn Trace 1807 ON; False for 1807 OFF"""
        if Flag:
            sql = """DBCC TRACEON(1807)"""
        else:
            sql = """DBCC TRACEOFF(1807)"""

        with self.master_connection.cursor() as master_cursor:
            master_cursor.execute(sql)
            master_cursor.commit()

        return None