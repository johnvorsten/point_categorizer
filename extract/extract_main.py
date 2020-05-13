# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:24:45 2020

@author: z003vrzk
"""

# Python imports
import os

# Third party imports
from pyodbc import IntegrityError
import sqlalchemy
import pandas as pd

# Local imports
import extract
from SQLAlchemyDataDefinition import Customers, Points, Netdev

# Local declarations
Extract = extract.Extract()
server_name = '.\DT_SQLEXPR2008'
driver_name = 'SQL Server Native Client 10.0'
database_name = 'Clustering'
Insert = extract.Insert(server_name, driver_name, database_name)

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
    server_name='.\DT_SQLEXPR2008'
    driver_name='SQL Server Native Client 10.0'
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


