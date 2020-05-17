# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:58:12 2019

@author: z003vrzk
"""

from JVWork_AccuracyVisual import (plt_accuracy, plt_accuracy2, 
                                   plt_distance, import_error_dfs)
import matplotlib.pyplot as plt

records = import_error_dfs()


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












