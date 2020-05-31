# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:58:12 2019

@author: z003vrzk
"""

# Python imports
import sys
import os

# Third party imports
import matplotlib.pyplot as plt

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

# from JVWork_AccuracyVisual import (plt_accuracy, plt_accuracy2,
#                                    plt_distance, import_error_dfs)

from clustering.accuracy_visualize import (Record,
                                           get_records,
                                           plt_indicy_accuracy_bar,
                                           plt_indicy_accuracy_scatter,
                                           plt_distance,
                                           plt_best_n_indicies,
                                           plt_hyperparameters,
                                           plt_loss_curve,
                                           # Depreciated methods
                                           import_error_dfs)

from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev, Customers,
                                              ClusteringHyperparameter, Labeling)

#%%

def test_depreciated_import_records():
    # Doesnt work anymore
    records = import_error_dfs(base_dir=r"C:\Users\z003vrzk\.spyder-py3\Scripts\ML\point_categorizer\error_dfs")
    return records

def test_get_records():
    primary_keys = [1,2,3,4,5]
    records = get_records(primary_keys)
    return records

def test_plt_indicy_accuracy_bar():
    records = get_records([8])
    plt_indicy_accuracy_bar(records)
    return None

def test_plt_indicy_accuracy_scatter():
    records = get_records([8,9,10,11])
    plt_indicy_accuracy_scatter(records)
    return None

def test_plt_distance():
    records = get_records([8,9,10,11])
    plt_distance(records)
    return None

def test_plt_best_n_indicies():
    records = get_records([8,9,10,11])
    plt_best_n_indicies(records)
    return None

def test_plt_hyperparameters():
    records = get_records([8,9,10,11])
    plt_hyperparameters(records, hyperparameter_name='clusterer')
    return None


def test_plt_loss_curve():
    records = ([8,9,10,11])
    # Doesnt work yet
    plt_loss_curve(records)
    return None






#%%

#Fun plotting functions
#plot all datasets (bar graph hyperparameter set v accuracy)
plt_accuracy([records[1]])

#plot a single dataset (bar graph hyperparameter set v accuracy)
plt_accuracy([records[1]],
             instance_tag=r'D:\Z - Saved SQL Databases\44OP-093324_Baylor_Bric_Bldg\JobDB.mdf')



plt_accuracy2([records[1]])

#Sort the x axis on correct number of clusters
plt_distance([records[1]], sort=True, sort_on='correct_k')

#Sort the x axis on number of database points & only plot the closest optimal k index (best prediction)
plt_distance([records[1]], sort=True, sort_on='n_points', closest_meth=True)

for record in records:
    plt_accuracy2([record])


#What do I need to do more calculations of?
for key, subdict in hyper_dict.items():
    print(key)
    print(subdict['records'][0].parent_file)
    print(len(subdict['records']), '\n')

#Import your database (.csv)
csv_file = r'data\master_pts_db.csv'
sequence_tag = 'DBPath'

#Get unique names
unique_tags = pd.read_csv(csv_file, index_col=0, usecols=['DBPath'])
unique_tags = list(set(unique_tags.index))
for  tag in unique_tags:
    plt_accuracy([records[1]],
                 instance_tag=tag)
    plt.paust(5)
    plt.close()



#%% Relationship hyperparameters v. loss

#Third party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()

from JVWork_AccuracyVisual import import_error_dfs
from JVWork_AccuracyVisual import plt_hyperparameters
from JVWork_Labeling import ExtractLabels
Extract = ExtractLabels()


#Import your database (.csv)
csv_file = r'data\master_pts_db.csv'
sequence_tag = 'DBPath'

#Get unique names
unique_tags = pd.read_csv(csv_file, index_col=0, usecols=['DBPath'])
unique_tags = list(set(unique_tags.index))

# Calculate best hyperparameters
tag = unique_tags[5]
if not 'records' in locals():
    records = import_error_dfs()
labels, hyper_dict = Extract.calc_labels(records, tag, best_n='all')

# Plot 'by_size' versus position
plt_hyperparameters(list(labels.values()),
                    hyper_param_field = 'by_size',
                    hyper_param_value=False,
                    plt_density=True,
                    plt_values=False)

# Plot 'clusterer' versus position
plt_hyperparameters(list(labels.values()),
                    hyper_param_field = 'clusterer',
                    hyper_param_value=False,
                    plt_density=True,
                    plt_values=False)

# Plot 'n_components' versus position
plt_hyperparameters(list(labels.values()),
                    hyper_param_field = 'n_components',
                    hyper_param_value=False,
                    plt_density=True,
                    plt_values=False)

# Plot 'reduce' versus position
plt_hyperparameters(list(labels.values()),
                    hyper_param_field = 'reduce',
                    hyper_param_value=False,
                    plt_density=True,
                    plt_values=False)

# Plot 'index' versus position
plt_hyperparameters(list(labels.values()),
                    hyper_param_field = 'index',
                    hyper_param_value=False,
                    plt_density=True,
                    plt_values=False)












