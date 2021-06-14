# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:19:51 2019

@author: z003vrzk
"""

# Python imports
import os, sys
import subprocess
from pathlib import Path, WindowsPath
import shutil
import re
from collections import namedtuple
import time

# Third party imports
import sqlalchemy
from pyodbc import DatabaseError, ProgrammingError
import pandas as pd

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract.sql_tools import SQLHandling
from extract.sql_tools import NameUsedError

# Global
DEFAULT_SQL_MDF_SEARCH_DIRECTORY = r"D:\Z - Saved SQL Databases"


#%%

class UNCMapping():

    def __init__(self):
        pass

    def get_network_connections(self):
        """Run a shell command to get the users network mapped paths (windows)"""

        if not os.name == 'nt':
            raise OSError(("get_network_connections and net use command is" +
                          "only valid on windows systems"))

        output = subprocess.run(['net', 'use'], stdout = subprocess.PIPE).stdout
        output = output.decode()

        return output

    @staticmethod
    def _parse_network_connections(network_string):
        """inputs
        -------
        network_string : (str) output of 'net use' windows cmd"""

        # Regex patterns
        drive_pattern = ': '
        unc_pattern = r'\\'
        drive_dict = {}

        #get UNC server names
        network_lines = network_string.splitlines()

        for line in network_lines:
            match_drive = re.search(drive_pattern, line)
            match_unc = re.search(unc_pattern, line)
            if match_drive and match_unc:
                letter = line[match_drive.start(0) - 1 : match_drive.start(0)]
                unc = line[match_unc.start(0) : len(line) - 1]
                drive_dict[letter] = unc

        return drive_dict

    def get_UNC(self):
        """Return a users mapped network drives. UNC path will be used for
        connecting to networked database"""

        net_use_output = self.get_network_connections()
        drive_dict = self._parse_network_connections(net_use_output)

        return drive_dict


#%%


class Extract():

    def __init__(self):
        """Retrieve SQL databases from remote servers and save them to a local
        disc.
        """
        self.sql_path_tuple = namedtuple('sql_files', ['mdf_path','ldf_path'])

        return None


    def search_and_save(self, 
                        search_directory, 
                        save_directory):
        """Iterate through a directory searching for database files and
        save them to save_directory
        inputs
        -------
        search_directory : (str) Directory to look for .mdf files
        save_directory : (str) Directory to save .mdf files into
        outputs
        -------
        None"""
        if isinstance(search_directory, str):
            search_directory = Path(search_directory)
        elif isinstance(search_directory, Path):
            pass
        else:
            raise ValueError('search_directory must be of type str() or' +
                             ' Path(). Received type : {}'\
                             .format(type(search_directory)))

        paths = [x for x in self.search_databases(search_directory,
                                           print_flag=True)]

        for path_mdf, path_ldf in paths:
            self.save_database(path_mdf, path_ldf, save_directory)
            print('Saved : {}'.format(path_mdf))

        return None


    def search_databases(self, 
                         search_directory, 
                         idx=1, 
                         print_flag=False):
        """Base function for module. Recursively looks through base_directory
        and copies any instances of SQL databases named JobDB.mdf and JobLog.ldf
        to a default directory on the D drive
        parameters
        -------
        search_directory : (str) or (Path) directory to recusively search for
            .mdf and .ldf files
        idx : (int) for printing (optional)
        print_flag : (bool) enable print (optional)
        output
        -------
        (None) Saves databases to default directory"""

        # Search child directories
        for _directory in os.listdir(search_directory):
            current_dir = Path(os.path.join(search_directory, _directory))

            # Search for files in directory
            if current_dir.is_dir():
                mdf_paths = sorted(current_dir.glob('*.mdf'))
                ldf_paths = sorted(current_dir.glob('*.ldf'))

                if print_flag:
                    print('{}Searching directory : {}'\
                          .format(chr(62)*idx, current_dir))

                # No databases found
                if len(mdf_paths) == 0:
                    yield from self.search_databases(current_dir,
                                                      idx=idx+1,
                                                      print_flag=print_flag)

                else:
                    for mdf_path, ldf_path in zip(mdf_paths, ldf_paths):

                        if print_flag:
                            print('{}Database found : {}'\
                                  .format(chr(62)*idx,str(mdf_path)))
                            print('{}Log found : {}'\
                                  .format(chr(62)*idx,str(ldf_path)))

                        db_files = self.sql_path_tuple(mdf_path=mdf_path,
                                             ldf_path=ldf_path)

                        yield db_files


    def save_database(self,
                      path_mdf,
                      path_ldf,
                      save_directory):
        """Saves a database path_mdf and log file path_ldf to
        save_directory. path_mdf and path_ldf are saved into a folder
        inside the destination directory (the folder name is auto-generated)
        Inputs
        -------
        path_mdf : (str) or (Path) database path
        path_ldf : (str) or (Path) log of database path
        save_directory : (str) or (Path) destination folder to save to
        Outputs
        -------
        Saves file to given directory"""

        # Create a root save directory if it doesn't already exist
        if not os.path.isdir(save_directory):
            self._create_save_directory(save_directory)

        try:
            folder_name = self._get_folder_name(str(path_mdf),
                                                 save_directory=save_directory)

            destination_folder = os.path.join(save_directory, folder_name)
            os.mkdir(destination_folder)

            shutil.copy(path_mdf, destination_folder)
            shutil.copy(path_ldf, destination_folder)

        except OSError as e:
            print('It looks like a Database at {} already exists at {}'\
                  .format(path_mdf, save_directory))
            print('Folder skipped')
            raise(e)
            return None

        except FileExistsError:
            print('Folder {} Already exists. Repeat database skipped'.format(
                    destination_folder))
            return None

        except PermissionError:
            print('Permission Denied on {}. Folder not copied'.format(path_mdf))
            return None


    def _create_save_directory(self, save_directory):
        """Creates a save directory if allowed by the user
        inputs
        -------
        save_directory : (str) directory to create"""
        if not os.path.isdir(save_directory):
            x = input('Create a directory at {}? [Y/N]'.format(save_directory))

            if x in ['True','TRUE','Y','y','yes','YES','Yes']:
                os.mkdir(save_directory)
                return True
            else:
                raise OSError("Directory does not existi at {}".format(save_directory))

        return None


    def _get_job_name(self, split_path, match_pattern='44op'):
        """Find the 44op job name from the source path. Return False if no
        job name is found
        inputs
        -------
        split_path : (list) of (str) representing path components
        match_pattern : (str) name pattern to look for (default '44op')
        outputs
        -------
        job_name : (str) if a 44op pattern is found, (False) otherwise"""

        for path in split_path:
            match = re.match(match_pattern, path, re.IGNORECASE)
            if match:
                return path

        return False


    def check_folder_exists(self, 
                            folder_name, 
                            save_directory):
        """Checks if folder_name already exists in save_directory
        Returns True if folder exists
        False if does not exist"""

        try:
            base_dir_list = os.listdir(save_directory)

            for base_dir in base_dir_list:
                if base_dir.__contains__(folder_name):
                    return True
                else:
                    return False
        except FileNotFoundError:
            return False


    def _get_generic_folder(self, save_directory, 
                            affix='No_Name_', 
                            idx=0):
        """Return a generic folder name if none already exists"""

        folder_name = affix + str(idx)

        if os.path.exists(os.path.join(save_directory, folder_name)):
            # Folder already exists, recursively generate
            return self._get_generic_folder(save_directory, idx=idx+1)
        else:
            return folder_name

        raise AssertionError("Function should not return here")

    def _get_folder_name(self, path_mdf, save_directory):
        """Return a string that is the folder name where a path_mdf file and
        path_ldf file will be saved
        inputs
        ------
        path_mdf : (str) or (Path) of source file. The final save path is derived
            from the source file path
        save_directory : (str) destination folder to create new folder in"""

        if isinstance(path_mdf, str):
            path_mdf = Path(path_mdf)
        elif isinstance(path_mdf, Path):
            pass

        split_path = path_mdf.parts
        match_pattern = '44OP'

        job_name = self._get_job_name(split_path, match_pattern)
        # If the job is known check if folder already exists
        if job_name:
            folder_exists = os.path.exists(os.path.join(save_directory, job_name))
        # No job name is known - the database is not known and we want to generate
        # A generic folder name
        else:
            folder_exists = False

        # If the path has a 44op job number and the folder doesn't already exist
        if job_name and not folder_exists:
            folder_name = job_name
            return folder_name

        # No 44op job number found - save into generic named folder
        elif job_name is False:
            folder_name = self._get_generic_folder(save_directory)
            return folder_name

        # Folder name already exists
        elif not folder_exists:
            raise OSError("Folder already exists at {}"\
                          .format(os.path.join(save_directory, job_name)))


    def iterate_dataframes(self, server_name,
                       driver_name,
                       database_name,
                       search_directory=DEFAULT_SQL_MDF_SEARCH_DIRECTORY):
        """Get objects from saved SQL databases
        Saved SQL databases have the tables {POINTBAS, POINTSEN, POINTFUN,
         NETDEV}
        POINTBAS, POINTSEN, POINTFUN all share a common key
        NETDEV includes information referenced by the POINT* tables
        Join all POINT* tables and return the objects
        Return NETDEV points in a separate table

        Yield an iterable which gives points from each saved database

        inputs
        -------
        server_name : (str) SQL Server name, defaults to named instance DT_SQLEXPR2008
            On local machine
        driver_name : (str) SQL driver name, defaults to
            'SQL Server Native Client 10.0'
        database_name : (str) Name of database to attach SQL databases as
        search_directory : (str) Directory to search for .mdf files
        outputs
        -------
        """

        # Find SQL database files under a specific directory
        path_iterator = self.search_databases(search_directory)

        # SQL helper for attaching and detaching found database files to server
        SQLHelper = SQLHandling(server_name=server_name, driver_name=driver_name)

        for mdf_path, ldf_path in path_iterator:
            db_name_working = database_name
            print('\n', mdf_path)
            # time.sleep(1)

            try:
                file_used_bool, name_used_bool, existing_database_name = \
                    SQLHelper.check_existing_database(mdf_path, db_name_working)

                if file_used_bool:
                    # A database from the file source path_mdf is already attached
                    # Use existing database name instead
                    db_name_working = existing_database_name

                elif all((name_used_bool, not file_used_bool)):
                    # A database is already attached under the same name
                    # Detach the database and try to attach my own
                    SQLHelper.detach_database(db_name_working)
                    # time.sleep(1)
                    SQLHelper.attach_database(mdf_path, db_name_working, ldf_path)
                    # time.sleep(1)

                else:
                    # No problems detected. Attach database
                    SQLHelper.attach_database(mdf_path, db_name_working, ldf_path)

            # Other cases
            except Exception as e:
                print(e)
                x = input('Issue connecting to {}. Skip and continue? (yes/no)'\
                          .format(mdf_path))
                if x in ['y','Y','yes','YES','Yes','True','TRUE']:
                    continue
                else:
                    raise e

            # Extract data into pandas dataframe
            points_sql = """SELECT [POINTBAS].[POINTID],
                            	[POINTBAS].[NETDEVID],	[POINTBAS].[NAME],
                            	[POINTBAS].[CTSYSNAME],	[POINTBAS].[DESCRIPTOR],
                            	[POINTBAS].[TYPE],	[POINTBAS].[INITVALUE],
                            	[POINTBAS].[TMEMBER],	[POINTBAS].[ALARMTYPE],
                            	[POINTBAS].[ALARMHIGH],	[POINTBAS].[ALARMLOW],
                            	[POINTBAS].[COMBOID],
                            	[POINTFUN].[FUNCTION],
                            	[POINTFUN].[VIRTUAL],	[POINTFUN].[PROOFPRSNT],
                            	[POINTFUN].[PROOFDELAY],	[POINTFUN].[NORMCLOSE],
                            	[POINTFUN].[INVERTED],	[POINTFUN].[LAN],
                            	[POINTFUN].[DROP],	[POINTFUN].[POINT],
                            	[POINTFUN].[ADDRESSEXT],	[POINTFUN].[SYSTEM],
                            	[POINTFUN].[CS],	[POINTFUN].[DEVNUMBER],
                            	[POINTFUN].[POINTACTUAL],
                            	[POINTSEN].[SENSORTYPE],	[POINTSEN].[CTSENSTYPE],
                            	[POINTSEN].[CONTRLTYPE],	[POINTSEN].[UNITSTYPE],
                            	[POINTSEN].[DEVICEHI],	[POINTSEN].[DEVICELO],
                            	[POINTSEN].[DEVUNITS],	[POINTSEN].[SIGNALHI],
                            	[POINTSEN].[SIGNALLO],	[POINTSEN].[SIGUNITS],
                            	[POINTSEN].[NUMBERWIRE],	[POINTSEN].[POWER],
                            	[POINTSEN].[WIRESIZE],	[POINTSEN].[WIRELENGTH],
                            	[POINTSEN].[S1000TYPE],	[POINTSEN].[SLOPE],
                            	[POINTSEN].[INTERCEPT]
                            FROM POINTBAS
                            FULL JOIN POINTSEN ON POINTBAS.POINTID = POINTSEN.POINTID
                            FULL JOIN POINTFUN ON POINTBAS.POINTID = POINTFUN.POINTID
                            ORDER BY POINTBAS.POINTID ASC"""
            try:
                points = SQLHelper.pandas_read_sql(points_sql, db_name_working)
            except (DatabaseError, ProgrammingError):
                # Column mismatch due to database versions.. :(
                points_sql = """SELECT *
                                FROM POINTBAS
                                FULL JOIN POINTSEN ON POINTBAS.POINTID = POINTSEN.POINTID
                                FULL JOIN POINTFUN ON POINTBAS.POINTID = POINTFUN.POINTID
                                ORDER BY POINTBAS.POINTID ASC"""
                points = SQLHelper.pandas_read_sql(points_sql, db_name_working)

            netdev_sql = """SELECT *
                            FROM NETDEV"""
            netdev = SQLHelper.pandas_read_sql(netdev_sql, db_name_working)

            SQLHelper.detach_database(db_name_working)


            yield points, netdev, mdf_path



#%%

class Insert(SQLHandling):

    def __init__(self,
                 server_name,
                 driver_name,
                 database_name):
        super().__init__(server_name=server_name, driver_name=driver_name)

        # SQLHelper = SQLHandling(server_name=server_name, driver_name=driver_name)
        # self.connection_str = SQLHelper.get_sqlalchemy_connection_str(database_name, driver_name=driver_name)
        self.connection_str = self.get_sqlalchemy_connection_str(database_name, driver_name=driver_name)

        # For interaction with SQLAlchemy ORM
        self.engine = sqlalchemy.create_engine(self.connection_str)

        return None

    @staticmethod
    def clean_dataframe(dataframe):
        """Delete duplicated columns and replace np.nan values with None
        None is mapped to Null in SQL
        inputs
        -------
        dataframe : (pd.DataFrame)
        outputs
        -------
        dataframe : (pd.DataFrame) cleaned dataframe"""

        # Remove duplicate columns
        dataframe = dataframe.loc[:,~dataframe.columns.duplicated()]
        #~dataframe.columns.duplicated returnas a boolean array like [True,True,False]

        # np.nan is invalid float/Null type in mssql server
        dataframe = dataframe.where(dataframe.notnull(), None)

        return dataframe


    def core_insert_dataframe(self, BaseClass, dataframe):
        """inputs
        -------
        BaseClass : (sqlalchemy.ext.declarative.api.DeclarativeMeta) the base
        class used for declarative class definitions. This is used to get the
        metadata associated with BaseClass. The metadata is needed to link the
        ORM and hte SQLAlchemy core. This method use the SQLAlchemy core to
        insert objects (I like it more than the ORM)
        dataframe : (pandas.DataFrame) to insert to table specified by BaseTable
        The dataframe must have columns for each mandatory column in
        BaseTable"""

        table_obj = BaseClass.__dict__['__table__']

        working_df = self.clean_dataframe(dataframe)
        values = working_df.to_dict(orient='records')

        insert_object = table_obj.insert()
        with self.engine.connect() as connection:
            res = connection.execute(insert_object, values)

        return res


    def core_insert_instance(self, BaseClass, dictionary):
        """
        inputs
        -------
        BaseClass : (sqlalchemy.ext.declarative.api.DeclarativeMeta) the base
        class used for declarative class definitions. This is used to get the
        metadata associated with BaseClass. The metadata is needed to link the
        ORM and hte SQLAlchemy core. This method use the SQLAlchemy core to
        insert objects (I like it more than the ORM)
        dictionary : (dict) dictionary of key values to insert"""

        table_obj = BaseClass.__dict__['__table__']
        insert_object = table_obj.insert().values(dictionary)
        with self.engine.connect() as connection:
            res = connection.execute(insert_object)

        return res


    def core_select_execute(self, select):
        """Given a sqlalchemy.sql.selectable.Select object, execute the select
        and return the result
        inputs
        -------
        BaseClass : (sqlalchemy.ext.declarative.api.DeclarativeMeta) the base
        class used for declarative class definitions. This is used to get the
        metadata associated with BaseClass. The metadata is needed to link the
        ORM and hte SQLAlchemy core. This method use the SQLAlchemy core to
        insert objects (I like it more than the ORM)
        select : (sqlalchemy.sql.selectable.Select)

        Example
        """
        with self.engine.connect() as connection:
            res = connection.execute(select).fetchall()

        return res

    def pandas_select_execute(self, select):
        """Given a sqlalchemy.sql.selectable.Select object, execute the select
        object through pandas.read_sql and output a dataframe
        inputs
        -------
        select : (sqlalchemy.sql.selectable.Select) query
        outputs
        -------
        result : (pd.DataFrame) query result

        Example
        """

        with self.engine.connect() as connection:
            res = pd.read_sql(select, connection)

        return res

    @staticmethod
    def _min_check_size(dataframe):
        """Get the minimum of dataframe.shape[0], 5"""

        n = min(dataframe.shape[0], 5)

        return n





