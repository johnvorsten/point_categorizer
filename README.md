# point_categorizer
Automatically label BAS
The goal of this repository is to create a framework that can automatically label BAS points into known classes.  

The implementation steps are : 
1) Collect data from remote servers
2) Build pipelines for data
3) Save data in .csv files (Actually, a noSQL system would suit this application well b/c we are dealing with sequential data
4) Automatically cluster data (unsupervised clustering)

Files : 
JVWork_ExtractDatabases.py
  A module that contains several functinos for recursively looking for sql databases in a remote location. Used to 
  originally save sql databases from a remote server to a local disc.
  functions{
  get_UNC
  search_databases #Most important, used to recursively search, and save
  save_databases #Interacts with the shell to save remote databasese to local location
  get_database_name #Handles naming of the copied databases to the local disc becasue the remote databases are all named 
    similar things and are in many different locations
  }

JVWork_ExtractDatabases_test.py
  A test script for JVWorl_ExtractDatabases.py. Not important. Maybe remove in the future

JVWork_MDFPipeline.py
  A module that contains several functions for extracting data from SQL databases.
  functions {
  search_database #searches a location for sql databases
  main #iterate through databases saved in local location and join tables into one csv file
  join_dataframes #joins a list of dataframes
  }

JVWork_MDFPipeline_SQL.py
  A module that contains a class useful for connecting to SQL databases and extracting their data.
  Useful for combination with pandas pd.read_sql_table
  class : MySQLHandling
  methods {
  create_master_connection
  attach
  detach
  create_PBDB_connection
  check_db_exist
  read_table
  get_UNC
  traceon1807
  }
  Attributes { #Not Exhaustive
  connMaster
  cursorMaster
  engine
  conn
  cursor
  
  }

JVWork_UnClusterAccuracy.py
  A module that contans a class to estimate the accuracy of the current optimal k clustering method.
  Classes : AccuracyTest
    methods {
    get_correct_k
    tib_optimal_k
    get_word_dictionary
    iterate
    plt_MDS
    plt_gap
    plot_gaps
    plot_reduced_dataset
    }
    Attributes { #Not exhaustive, useful for high level overview
    correct_k_dict_path
    correct_k_dict
    error_df_path
    error_df
    [...]
    }

JVWork_UnClusterPlotting.py
  A module that contains several functions mostly used to make the methods under JVClusterTools in JVWork_UnsupervisedCluster.py
  Calling this module will execute its line scripts, which may take a while.  This modules is best used as a reference.
  
JVWork_UnsupervisedCluster.py
  A module that contains a class to handle unsupervised clustering of data.
  Classes : JVClusterTools
    methods {
    get_database_set
    read_database_set
    get_word_name
    optimalK2
    }
    
JVWork_WholeDBPipeline.py
  A module that contains all classes and definitions to transform raw collected data.
