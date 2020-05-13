"""
Created on Thu Mar 28 14:33:12 2019

This module can be used to connect databases to the local SQL server installed
by default with design tool
This module has a few important assumptions :
1) Use the SQL server name DT_SQLEXPR2008
2) Use trusted connection (windows authentication - not user & password)

Note : Using the format below I do not have to close connections to SQL server
This is becasue conections are closed after the leave scope (of the with statement)
conn = pyodbc.connect('DRIVER=MySQL ODBC 5.1 driver;SERVER=localhost;DATABASE=spt;UID=who;PWD=testest')
with conn:
    crs = conn.cursor()
    do_stuff


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

class SQLHandling:
    r"""A useful class for connecting to SQL servers using microsoft authentication
    assuming the standard connection properties for the design-tool product
    by Siemens, inc.
    Assumes
    a) Windows/domain authentication
    b) Server name is configurable, but defaults to '.\DT_SQLEXPR2008' if not
        defined
    c) Driver is '{SQL Server Native Client 10.0}' if not defined
    d) path to log file is configurable, but assumes JobDB_Log.ldf if not defined

    Usage example
    server_name = r'.\DT_SQLEXPR2008'
    driver_name = 'SQL Server Native Client 10.0'
    path_mdf = r'C:\Users\z003vrzk\.spyder-py3\Scripts\Work\PanelBuilder\SQLTest\JobDB.mdf'
    database_name = 'PBJobDB'

    # Instantiate a class which creates a master connection (to master db)
    mysql = SQLHandling(server_name=server_name, driver_name=driver_name)

    # If connecting to a networked database turn on flag 1807
    path_is_net_drive = not os.path.splitdrive(path_mdf)[0] in ['C:', 'D:']
    path_is_network_name = os.path.splitdrive(path_mdf)[0].startswith(r'\\')
    if any((path_is_net_drive, path_is_network_name)):
        mysql.traceon1807(True)

    # Attach a database
    mysql.attach_database(path_mdf, database_name)
    mysql.traceon1807(False)


    # Read a table
    df = mysql.read_table(database_name, table_name='POINTBAS')

    # Detach database when you are done
    mysql.detach_database(database_name) #only detaches PBJobDB currently

    # Close cursors when done
    connection.close()
    cursor.close()"""

    def __init__(self, server_name=None, driver_name=None):
        """A helper class for sql databases. This incldues attaching, detaching,
        and connecting to databases with an sql server. This method only
        supports microsoft authentication (not user and password)
        inputs
        -------
        server_name : (str) name of sql server. Defaults to the local computer
        name if no name is given
        driver : (str) type of driver. Default is {SQL Server Natie Client 10.0}.
        If no driver is input, then this will be used"""

        if server_name is None:
            self.server_name = '.\DT_SQLEXPR2008'
        else:
            self.server_name = server_name

        if driver_name is None:
            self.driver_name = '{SQL Server Native Client 10.0}'
        else:
            self.driver_name = '{{{driver_name}}}'.format(driver_name=driver_name)

        # For pyodbc connection only
        self.master_connection_str= 'DRIVER={}; SERVER={}; DATABASE=master; Trusted_Connection=yes;'.format(self.driver_name, self.server_name)

    def get_pyodbc_master_connection_str(self):
        """Return the master database connection string"""
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

    def get_sqlalchemy_connection_str(self, database_name,
                                      driver_name='SQL+Server+Native+Client+10.0'):
        """engine for sqlalchemy"""

        server_name = self.server_name

        """ Example
        conn_str = 'mssql://.\\DT_SQLEXPR2008/PBJobDB?trusted_connection=yes&driver=SQL+Server+Native+Client+10.0'

        'mssql://.\\DT_SQLEXPR2008/PBJobDB?trusted_connection=yes&driver=SQL+Server+Native+Client+10.0'
        """

        engine_str = r'mssql+pyodbc://{0}/{1}?driver={2}&trusted_connection=yes'\
                        .format(server_name, database_name, driver_name)

        return engine_str


    def attach_database(self,
                        path_mdf,
                        database_name,
                        path_ldf):
        """Used to attach 'database_name' to the sql server. If path_ldf is
        not defined, then this method assumes the log file is in the same
        directory as path_mdf, with the file name 'JobDB_Log.ldf'. Define
        path_ldf if the file name is different

        inputs
        -------
        path_mdf : (str or Path) user specified path to .MDF file. LDF file
        must be in same directory.
        database_name : (str) name of database to attach. This can be any user
            specified name, but it must be used to attach, detach, and connect to
            a specific database
        path_ldf : (str or Path) path to database log file

        output
        -------
        database_name : (str) of actual database name attached as
        """
        if isinstance(path_mdf, str):
            path_mdf = Path(path_mdf)
        elif isinstance(path_mdf, Path):
            pass
        else:
            raise ValueError("path_mdf must be of type str or pathlib.Path." +
                             " Value passed was {}".format(type(path_mdf)))

        try:
            # Check if the selected file is already in use or if the requested
            # database name is already used
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

            sql1 = "CREATE DATABASE [{}]".format(database_name)
            sql2 = "ON (Filename = '{pathMDF}'), (Filename = '{pathLDF}')"\
                    .format(pathMDF=path_mdf, pathLDF=path_ldf)
            sql3 = "FOR Attach"
            sql = sql1 + " " + sql2 + " " + sql3

            # Connections are closed after sql is committed
            with pyodbc.connect(self.master_connection_str) as connectionMaster:
                cursorMaster = connectionMaster.cursor()
                connectionMaster.autocommit = True
                cursorMaster.execute(sql)
                # connectionMaster.autocommit = False
                cursorMaster.close()

            logging.info('Database : {} connected'.format(database_name))

            if any((path_is_net_drive, path_is_network_name)):
                self.traceon1807(False)

        except Exception as e:
            logging.debug(e)
            raise(e)

        return database_name

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

        return


    def check_existing_database(self, path_mdf, database_name):
        """Check two conditions :
        1) A database with 'database_name' is already connected
        to the instance of sql server
        2) A database physical_name (operating system file name) is already
        connected to the server instance

        inputs
        -------
        path_mdf : (str or pathlib.Path) path to master data file .mdf
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
        file_used_bool is true"""

        path_mdf = Path(path_mdf)

        sql = """select [name] as logical_name, physical_name,
        	(select name
        	from [master].[sys].[databases] as t2
        	where t2.database_id = t1.database_id) as database_name
        from sys.master_files as t1"""
        with pyodbc.connect(self.master_connection_str) as connectionMaster:
            # Find all database currently connected
            cursorMaster = connectionMaster.cursor()
            cursorMaster.execute(sql)
            rows = cursorMaster.fetchall()
            cursorMaster.close()

        # Assume the file and database name is not initially used
        file_used_bool = False
        name_used_bool = False
        existing_database_name = None

        # If the query indicates the .mdf file name OR database name is already
        # Used then change the boolean state
        for row in rows:
            if Path(row.physical_name) == path_mdf:
                file_used_bool = True
                existing_database_name = row.database_name
            if row.database_name == database_name:
                name_used_bool = True

        return (file_used_bool, name_used_bool, existing_database_name)

    def read_table(self, database_name, table_name):
        """Read a table to dataframe using pyodbc and pandas. The server and driver
        used to instantiate the class is used (self.server_name, self.driver_name)
        inputs
        -------
        database_name : (str) name of database to read from
        table_name : (str) name of table in database_name to return
        outputs
        -------
        df : (pandas.DataFrame) SQL table"""

        driver_name = self.driver_name
        server_name = self.server_name

        # For pyodbc connection only
        pydobc_conn_str= 'DRIVER={}; SERVER={}; DATABASE={}; Trusted_Connection=yes;'\
                                .format(driver_name,
                                        server_name,
                                        database_name)
        sql_query = """SELECT * FROM {}""".format(table_name)

        try:
            with pyodbc.connect(pydobc_conn_str) as connection:
                df = pd.read_sql(sql_query, con=connection)
        except Exception as e:
            logging.info(e)
            raise(e)

        return df

    def pandas_read_sql(self,
                        sql_query,
                        database_name):
        """Read a table to dataframe using pyodbc and pandas. The server and driver
        used to instantiate the class is used (self.server_name, self.driver_name)
        inputs
        -------
        sql_query : (str) sql string to execute
        database_name : (str) name of database to read from
        table_name : (str) name of table in database_name to return
        outputs
        -------
        df : (pandas.DataFrame) SQL table"""

        driver_name = self.driver_name
        server_name = self.server_name

        # For pyodbc connection only
        pydobc_conn_str= 'DRIVER={}; SERVER={}; DATABASE={}; Trusted_Connection=yes;'\
                                .format(driver_name,
                                        server_name,
                                        database_name)
        try:
            with pyodbc.connect(pydobc_conn_str) as connection:
                df = pd.read_sql(sql_query, con=connection)
        except Exception as e:
            logging.info(e)
            raise(e)

        return df

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

        #get UNC server names
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

        with pyodbc.connect(self.master_connection_str) as connectionMaster:
            cursorMaster = connectionMaster.cursor()
            cursorMaster.execute(sql)
            cursorMaster.commit()
            cursorMaster.close()

        return None

