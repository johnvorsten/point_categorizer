# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:29:46 2020

@author: z003vrzk
"""
# Python imports
import os
import sys

# Third party imports
import sqlalchemy

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from ranking import Labeling
from transform import transform_pipeline
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter)
from extract.SQLAlchemyDataDefinition import Labeling as SQLTableLabeling
from clustering.accuracy_visualize import Record, get_records

ExtractLabels = Labeling.ExtractLabels()

#%% Extract labels for all datasets

# Set up connection to SQL
Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                        driver_name='SQL Server Native Client 10.0',
                        database_name='Clustering')

# Get all records relating to one customer
customer_id = 15
sel = sqlalchemy.select([Clustering.id, Clustering.correct_k])\
    .where(Clustering.customer_id.__eq__(customer_id))
res = Insert.core_select_execute(sel)
primary_keys = [x.id for x in res]
correct_k = res[0].correct_k

sel = sqlalchemy.select([Customers.name]).where(Customers.id.__eq__(customer_id))
customer_name = Insert.core_select_execute(sel)[0].name

# Calculate ranking of all records
records = get_records(primary_keys)
best_labels = ExtractLabels.calc_labels(records, correct_k, error_scale=0.8, var_scale=0.2)


#%% Get the best clustering hyperparameter for a dataset

