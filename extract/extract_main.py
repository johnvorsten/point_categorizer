# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:24:45 2020

@author: z003vrzk
"""

# Python imports
import os
import sys
from pathlib import Path
import configparser


# Third party imports
from pyodbc import IntegrityError
import sqlalchemy
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

from extract import extract
from extract.SQLAlchemyDataDefinition import (Customers, Points, Netdev,
                                              ClusteringHyperparameter, Clustering,
                                              Labeling)

# Local declarations
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']

Extract = extract.Extract()
Insert = extract.Insert(server_name,
                        driver_name,
                        database_name)

#%%
"""
Save databases from server to local machine
"""

def main_copy_to_local():

    search_directory = r"R:\JOBS"
    save_directory = r"D:\Z - Saved SQL Databases"

    Extract.search_and_save(search_directory, save_directory)

    return None

#%% Main function
"""Extract infromation from distributed databases and save all information
into a central database
"""

def main():
    """"""

    # Iterate & Extract points information of a database
    working_database_name='PipelineDB' # For extracting information
    search_directory=r"D:\Z - Saved SQL Databases"
    df_iterator = Extract.iterate_dataframes(server_name=server_name,
                                         driver_name=driver_name,
                                         database_name=working_database_name,
                                         search_directory=search_directory)
    i = 0
    for points_df, netdev_df, mdf_path in df_iterator:

        # points_df, netdev_df, mdf_path = next(df_iterator)

        # Check to see if customer is already in database
        sel = sqlalchemy.select([Customers]).where(Customers.name == str(mdf_path))
        with Insert.engine.connect() as connection:
            result = connection.execute(sel).fetchall()

        if result.__len__() < 1:
            # Insert customer name to new database
            customer_dict = {'name': str(mdf_path)}
            result = Insert.core_insert_instance(Customers, customer_dict)
        else:
            # Customer already exists (unique constraint)
            print("Customer {} already exists".format(mdf_path))

        # Get PK of new customer entry
        sel = sqlalchemy.select([Customers]).where(Customers.name == str(mdf_path))
        with Insert.engine.connect() as connection:
            result = connection.execute(sel)
            pk = result.fetchone().id

        # Construct points and netdev with foreign key
        points_df['customer_id'] = [pk] * points_df.shape[0]
        netdev_df['customer_id'] = [pk] * netdev_df.shape[0]

        # Check if points already exist in the database
        # Points dont have to be unique, but I should avoid inserting the
        # Same dataframe multiple times
        test_n = points_df.shape[0]
        sel = sqlalchemy.select([Points.NAME]).where(Points.customer_id.__eq__(pk))
        with Insert.engine.connect() as connection:
            result = connection.execute(sel).fetchall()
        if result.__len__() < test_n:
            # Insert points to database
            result = Insert.core_insert_dataframe(Points, points_df)
        else:
            print("Points not inserted - points already exist")

        # Check if netdev already exist in the database
        # Points dont have to be unique, but I should avoid inserting the
        # Same dataframe multiple times
        test_n = netdev_df.shape[0]
        sel = sqlalchemy.select([Netdev]).where(Netdev.customer_id.__eq__(pk))
        with Insert.engine.connect() as connection:
            result = connection.execute(sel).fetchall()
        if result.__len__() < test_n:
            # Insert points to database
            result = Insert.core_insert_dataframe(Netdev, netdev_df)
        else:
            print("Netdev not inserted - Netdev already exist")

        i += 1
        print(i)

    return None


#%% Main function
"""Take information that was saved in .csv files and move to SQL Server
"""




def enforce_clustering_null(dictionary, replace=0):
    """Replace specific dictionary keys with 0 instead of None/Null"""
    not_null = ['correct_k','n_points','n_len1','n_len2','n_len3',
                'n_len4','n_len5','n_len6','n_len7']

    for key in not_null:
        if dictionary[key] is None:
            dictionary[key] = 0

    return dictionary

def rename_columns(dataframe):
    """Rename dataframe columns for dictionary"""

    df = dataframe.rename(columns={
        'optk_MDS_gap_max':'gap_max',
       'optk_MDS_gap_Tib':'gap_tib',
       'optk_MDS_gap*_max':'gap_star',
       'optk_X_gap_max':'gap_max',
       'optk_X_gap_Tib':'gap_tib',
       'optk_X_gap*_max':'gap_star',
       'optk_TSNE_gap_max':'gap_max',
       'k_TSNE_gap_Tib':'gap_tib',
       'k_TSNE_gap*_max':'gap_star'})

    return df

def check_hyperparameters_inserted(by_size,
                                   clusterer,
                                   distance,
                                   reduce,
                                   n_components):

    sel = sqlalchemy.select([ClusteringHyperparameter])\
        .where(ClusteringHyperparameter.by_size.__eq__(by_size))\
        .where(ClusteringHyperparameter.clusterer.__eq__(clusterer))\
        .where(ClusteringHyperparameter.distance.__eq__(distance))\
        .where(ClusteringHyperparameter.reduce.__eq__(reduce))\
        .where(ClusteringHyperparameter.n_components.__eq__(n_components))

    sel = sqlalchemy.select([ClusteringHyperparameter])

    with Insert.engine.connect() as connection:
        res = connection.execute(sel)

    return res

def remove_unused_keys(BaseClass, dictionary):
    """Remove keys from dictionary if they are not in BaseClass table
    column names"""

    # Find keys to remove
    remove_keys = []
    for key in dictionary.keys():
        if not hasattr(BaseClass, key):
            remove_keys.append(key)

    # Remove keys
    for key in remove_keys:
        dictionary.pop(key)

    return dictionary

def pair_csv_files(base_directory=r'C:\Users\z003vrzk\.spyder-py3\Scripts\ML\point_categorizer\error_dfs'):
    # Location of .csv files
    csv_files = list(Path(base_directory).glob('*.csv'))

    hyper_csvs = [x for x in csv_files if x.stem.__contains__('hyper')]
    df_csvs = [x for x in csv_files if not x.stem.__contains__('hyper')]

    # Pair up hyperparameter dicts and cluster dicts
    pairs = []
    for hyper_csv in hyper_csvs:
        for df_csv in df_csvs:
            if hyper_csv.stem.__contains__(df_csv.stem):
                pairs.append((df_csv, hyper_csv))

                idx = hyper_csvs.index(hyper_csv)
                hyper_csvs.pop(idx)
                idx = df_csvs.index(df_csv)
                df_csvs.pop(idx)

    return pairs


def hyperparameter_type_conversion(dataframe):
    """inputs
    -------
    dataframe : (pd.DataFrame) with cols by_size, clusterer, distance,
        reduce, n_components"""

    by_size = int(bool(dataframe['by_size'][0]))
    clusterer = str(dataframe['clusterer'][0])
    distance = str(dataframe['distance'][0])
    reduce = str(dataframe['reduce'][0])
    n_components = int(dataframe['n_components'][0])

    return by_size, clusterer, distance, reduce, n_components


def insert_csv_to_sql():
    base_directory=r'C:\Users\z003vrzk\.spyder-py3\Scripts\ML\point_categorizer\error_dfs'
    pairs = pair_csv_files(base_directory=base_directory)

    for df_csv, hyper_csv in pairs:
        hyper = pd.read_csv(hyper_csv)

        # Deal with hyperparameter dictionary
        (by_size, clusterer, distance, reduce,
             n_components) = hyperparameter_type_conversion(hyper)

        # See if its already inserted
        sel = sqlalchemy.select([ClusteringHyperparameter]).where(
            sqlalchemy.sql.and_(ClusteringHyperparameter.by_size == by_size,
                                ClusteringHyperparameter.clusterer == clusterer,
                                ClusteringHyperparameter.distance == distance,
                                ClusteringHyperparameter.reduce == reduce,
                                ClusteringHyperparameter.n_components == n_components))
        with Insert.engine.connect() as connection:
            res = connection.execute(sel).fetchall()

        if res.__len__():
            # Get hyperparameters id
            hyperparameter_id = res[0].id
        else:
            # Insert new object
            values = hyper.to_dict(orient='records')[0]
            values = remove_unused_keys(ClusteringHyperparameter, values)
            res = Insert.core_insert_instance(ClusteringHyperparameter, values)
            hyperparameter_id = res.inserted_primary_key[0]

        # Convert to dictionary
        df = pd.read_csv(df_csv)
        df = rename_columns(df)
        df = Insert.clean_dataframe(df)
        df_dicts = df.to_dict(orient='records')

        for df_dict in df_dicts:

            # Get customer ID for each clustered item
            customer_name = df_dict['DBPath']
            sel = sqlalchemy.select([Customers]).where(Customers.name.__eq__(customer_name))
            with Insert.engine.connect() as connection:
                res = connection.execute(sel).fetchone()
                customer_id = res.id

            # Insert information to dictionary
            df_dict['customer_id'] = customer_id
            df_dict['hyperparameter_id'] = hyperparameter_id

            # Insert instance to Table
            df_dict = remove_unused_keys(Clustering, df_dict)
            df_dict = enforce_clustering_null(df_dict)
            Insert.core_insert_instance(Clustering, df_dict)

    return None


#%% Convert mongoDB to SQL

from pymongo import MongoClient

def mongo_to_sql():
    """Convert labeled Monto documents to SQL relationships"""

    # Loop through mongo documents
    client = MongoClient('localhost', 27017)
    db = client['master_points']
    clustered_points = db['clustered_points']

    document = clustered_points.find_one()
    document.keys()
    cluster = document['clustered_points'][0]
    document['clustered_points'][0].keys()

    # Iterate through all mongo documents
    for document in clustered_points.find():

        # Get database tag
        customer_name = document['database_tag']

        # Get customer ID
        sel = sqlalchemy.select([Customers]).where(Customers.name.__eq__(customer_name))
        customers = Insert.core_select_execute(sel)
        customer_id = customers[0].id

        # Iterate through clustered points
        for cluster in document['clustered_points']:
            points = cluster['points'] # dictionary of lists
            label = cluster['label'][0] # Get object from list

            # Create a groupid and insert it
            values = {'bag_label':label}
            res = Insert.core_insert_instance(Labeling, values)
            group_id = res.inserted_primary_key[0]

            # Filter points in this cluster
            pointids = points['POINTID']
            # For each clustered point set the group_id foreign key
            update = Points.__table__.update()\
                .where(Points.POINTID.in_(pointids))\
                .where(Points.customer_id.__eq__(customer_id))\
                .values(group_id=group_id)
            with Insert.engine.connect() as connection:
                res = connection.execute(update)

    return None