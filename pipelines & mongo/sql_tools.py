"""
Dont use this module in the future - use the graded version in a different package...

Deprecated
"""


import pyodbc
import sqlalchemy
import subprocess
import pandas as pd

class MySQLHandling():
    engineMaster = object
    connMaster = object
    cursorMaster = object
    engine = object
    conn = object
    cursor = object
    
    def create_master_connection(self):
        """Used for connection to the master database under the server
        .\DT_SQLEXPR2008 where . is the users desktop and DT_SQLEXPR2008
        is the SQL server setup on each users desktop.
        
        return cursorMaster, connMaster"""
        #Create pyodbc connection
        self.connMaster = pyodbc.connect('DRIVER={SQL Server Native Client 10.0}; SERVER=.\DT_SQLEXPR2008;DATABASE=master;Trusted_Connection=yes;')
        self.cursorMaster = self.connMaster.cursor()
        return self.cursorMaster, self.connMaster
        
    def attach(self, pathMDF, database_name):
        """Used to attach PBJobDB. Note: The database added
        will have the default name PBJobDB to distinguish it from any databases.
        
        path = user specified path to .MDF file. LDF file must be in same directory.
        Assumed names are JobDB.mdf and JobDB_Log.ldf
        """
        assert type(database_name) is str, 'Must pass database_name as string'
        
        if self.check_db_exist(database_name):
            print('Database name: {} is already connected'.format('PBJobDB'))
            return
        
#        server = '.\DT_SQLEXPR2008' #may need to have this be a user-entered value for computer name
#        scriptLocation = 'C:\SQLTest\AttachDatabase.sql' #may need o be dynamically defined w/ os.getcwd()
#        subprocess.call(['sqlcmd','-S',server,'-i',scriptLocation])
        dirPathIndex = pathMDF.find('JobDB.mdf')
        dirPath = pathMDF[0:dirPathIndex]
        pathLDF = dirPath + 'JobDB_Log.ldf'
                
        sql1 = "CREATE DATABASE [{}]".format(database_name)
        sql2 = "ON (Filename = '{pathMDF}'), (Filename = '{pathLDF}')".format(pathMDF = pathMDF, pathLDF = pathLDF)
        sql3 = "FOR Attach"
        sql = sql1 + " " + sql2 + " " + sql3

        self.connMaster.autocommit = True
        self.cursorMaster.execute(sql)
        self.connMaster.autocommit = False
        print('Database connected')
        
    def detach(self, database_name): #detach PBJobDB
        """Used to detach 'PBJobDB'.  Note: I should use
        this once I get the information needed from the database.  This will
        NOT call cursor.close() and engine.dispose(). Currently only closes PBJobDB
        
        I can add a dynamically named database by reading the sql file
        """
        
        detach_str = """USE [master]
        GO
        ALTER DATABASE [{db_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE
        GO
        EXEC master.dbo.sp_detach_db @dbname = N'{db_name}', @skipchecks = 'false'
        GO""".format(db_name=database_name)
                        
        server = '.\DT_SQLEXPR2008' #may need to have this be a user-entered value for computer name
        scriptPath = r'.\DetachDatabase.sql' #may need o be dynamically defined w/ os.getcwd()
        with open(file='DetachDatabase.sql', mode='w') as f:
            dir(f)
            f.write(detach_str)
        
        subprocess.call(['sqlcmd','-S',server,'-i',scriptPath])
        print('Database removed')
        
    def create_PBDB_connection(self, database_name):
        """Used in connection to PBJobDB database.  This is the connection to
        the database specifeid by the user, and the job database.  User the standard
        global outputs "conn" (pyodbc) and "engine" (sqlalchemy) to execute sql
        querys or manipulate data with pandas
        
        return engine, conn, cursor"""
        
        engine_str = r'mssql+pyodbc://.\DT_SQLEXPR2008/{}?driver=SQL+Server+Native+Client+10.0'.format(database_name)
#        self.engine = sqlalchemy.create_engine('mssql+pyodbc://.\DT_SQLEXPR2008/PBJobDB?driver=SQL+Server+Native+Client+10.0')
        self.engine = sqlalchemy.create_engine(engine_str)
        self.engine.connect()
        
        pydobc_conn_str = r'DRIVER={{SQL Server Native CLient 10.0}};SERVER=.\DT_SQLEXPR2008;DATABASE={};Trusted_Connection=yes;'.format(database_name)
#        self.conn = pyodbc.connect('DRIVER={SQL Server Native CLient 10.0};SERVER=.\DT_SQLEXPR2008;DATABASE=PBJobDB;Trusted_Connection=yes;')
        self.conn = pyodbc.connect(pydobc_conn_str)
        self.cursor = self.conn.cursor()
        return self.engine, self.conn, self.cursor
        
    def check_db_exist(self, database):
        sql = """SELECT name FROM master.sys.databases"""
        self.cursorMaster.execute(sql)
        names = self.cursorMaster.fetchall() #get all names
        names = [name[0] for name in names] #convert row object to list object
        
        return names.__contains__(database) #True if database is connected
        
    def read_table(self, tableName):
        sql = """SELECT * FROM {}""".format(tableName)
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        
        column_ID = [column[0] for column in self.cursor.description]
        return rows, column_ID
    
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
        Flag : True for turn Trace 1807 ON; False for 1807 OFF"""
        if Flag:
            sql = """DBCC TRACEON(1807)"""
        else:
            sql = """DBCC TRACEOFF(1807)"""
        self.cursorMaster.execute(sql)
        self.cursorMaster.commit()
        

def test():
    mysql = MySQLHandling()
    mysql.create_master_connection()
    mysql.traceon1807(True)
    
    path = r'\\usaus000001dat.us009.siemens.net\JobData\JOBS\0 - Engineering Quality\SQLTest\JobDB.mdf'
    database_name = 'PBJobDB'
    mysql.attach(path, database_name)
    mysql.create_PBDB_connection(database_name) #Connects only to database called PBJobDB
    
    mysql.detach(database_name) #only detaches PBJobDB currently










