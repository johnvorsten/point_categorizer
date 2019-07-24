# point_categorizer
Automatically label BAS <br/>
The goal of this repository is to create a framework that can automatically label BAS points into known classes. <br/>
***
The implementation steps are : <br/>
1) Collect data from remote servers <br/>
2) Build pipelines for data <br/>
3) Save data in .csv files (Actually, a noSQL system would suit this application well b/c we are dealing with sequential data <br/>
4) Automatically cluster data (unsupervised clustering) <br/>

Files : <br/>
1. JVWork_ExtractDatabases.py <br/>
    A module that contains several functinos for recursively looking for sql databases in a remote location. Used to  <br/>
    originally save sql databases from a remote server to a local disc.  <br/>
    functions{  <br/>
    get_UNC  <br/>
    search_databases # Most important, used to recursively search, and save  <br/>
    save_databases # Interacts with the shell to save remote databasese to local location  <br/>
    get_database_name #Handles naming of the copied databases to the local disc becasue the remote databases are all named  <br/>
     similar things and are in many different locations  <br/>
   }  
***
2. JVWork_ExtractDatabases_test.py <br/>
    A test script for JVWorl_ExtractDatabases.py. Not important. Maybe remove in the future <br/>
***
3. JVWork_MDFPipeline.py <br/>
    A module that contains several functions for extracting data from SQL databases. <br/>
        functions { <br/>
        search_database #searches a location for sql databases <br/>
        main #iterate through databases saved in local location and join tables into one csv file <br/>
        join_dataframes #joins a list of dataframes <br/>
        } <br/>
***
4. JVWork_MDFPipeline_SQL.py <br/>
    A module that contains a class useful for connecting to SQL databases and extracting their data. <br/>
    Useful for combination with pandas pd.read_sql_table <br/>
        class : MySQLHandling <br/>
            methods { <br/>
            create_master_connection <br/>
            attach <br/>
            detach <br/>
            create_PBDB_connection <br/>
            check_db_exist <br/>
            read_table <br/>
            get_UNC <br/>
            traceon1807 <br/>
            } <br/>
            Attributes { #Not Exhaustive <br/>
            connMaster <br/>
            cursorMaster <br/>
            engine <br/>
            conn <br/>
            cursor <br/>
            <br/>
            } <br/>
***
5. JVWork_UnClusterAccuracy.py <br/>
    A module that contans a class to estimate the accuracy of the current optimal k clustering method. <br/>
        Classes : AccuracyTest <br/>
            methods { <br/>
            get_correct_k <br/>
            tib_optimal_k <br/>
            get_word_dictionary <br/>
            iterate <br/>
            plt_MDS <br/>
            plt_gap <br/>
            plot_gaps <br/>
            plot_reduced_dataset <br/>
            } <br/>
            Attributes { #Not exhaustive, useful for high level overview <br/>
            correct_k_dict_path <br/>
            correct_k_dict <br/>
            error_df_path <br/>
            error_df <br/>
            [...] <br/>
            } <br/>
***
6. JVWork_UnClusterPlotting.py  <br/>
    A module that contains several functions mostly used to make the methods under JVClusterTools in JVWork_UnsupervisedCluster.py <br/>
    Calling this module will execute its line scripts, which may take a while.  This modules is best used as a reference. <br/>
***
7. JVWork_UnsupervisedCluster.py  <br/>
    A module that contains a class to handle unsupervised clustering of data. <br/>
        Classes : JVClusterTools <br/>
        methods { <br/>
        get_database_set <br/>
        read_database_set <br/>
        get_word_name <br/>
        optimalK2 <br/>
        } <br/>
***
8. VWork_WholeDBPipeline.py <br/>
    A module that contains all classes and definitions to transform raw collected data. <br/>
