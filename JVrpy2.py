# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:21:52 2019

@author: z003vrzk
"""

"""Extracting NbClsut returned information"""
import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
#from rpy2.robjects.numpy2ri import numpy2ri
import pandas as pd
import numpy as np
from collections import namedtuple
numpy2ri.activate()

NbClustResults = namedtuple('NbClustResults', 'index_df best_nc_df')

_module_name = 'NbClust'
if not rpy2.robjects.packages.isinstalled(_module_name):
    raise ImportError('{} is not installed'.format(_module_name))
nbclust = importr('NbClust') #Import the NbClust package


def rmatrix_2df(r_matrix):
    """Method for converting an rpy2.robjects.Matrix to a pandas dataframe.
    This method assumed the matrix is 2D and has named dimensions"""
    assert type(r_matrix) is rpy2.robjects.Matrix, 'r_matrix argument is not\
    type rpy2.robjects.Matrix'
    values = np.array(r_matrix)
    row_names = list(r_matrix.rownames)
    col_names = list(r_matrix.colnames)
    df = pd.DataFrame(data=values, 
                      index=row_names, 
                      columns=col_names)
    return df

def nbclust_calc(data, min_nc, max_nc, 
                 distance='euclidean', method='kmeans', 
                 index='all'):
    """Uses the R package NbClust to find the optimal number of clusters
    in a dataset.  Returns the results in a python-friendly way
    parameters
    -------
    data : numpy array of your data of shape (n,p) where n is the number of 
    instances and p is the number of features
    min_nc : NbClust parameter, minimum number of clusters
    max_nc : NbClust parameter, maximum number of clusters
    distance : distance measurement between instances
    method : clustering method to use
    index : indicies to return from NbClust
    see https://www.rdocumentation.org/packages/NbClust/versions/3.0/topics/NbClust
    for information on parameters
    returns
    -------
    NbClustResults : tuple of index_df and best_nc_df
    index_df : a dataframe of all number of clusters and the corresponding metric
    best_nc_df : a dataframe of the best number of clusters for each indicy in
    index_df
    see the NbClust documentation for more information"""
    
    if index == 'jv_custom':
        index = ['KL', 'CH', 'Hartigan', 'CCC', 'Scott', 'Marriot', 'TrCovW', 'TraceW',
           'Friedman', 'Rubin', 'Cindex', 'DB', 'Silhouette', 'Duda', 'PseudoT2',
           'Beale', 'Ratkowsky', 'Ball', 'PtBiserial', 'Gap', 'Frey', 'McClain',
           'Gamma', 'Gplus', 'Tau', 'Dunn', 'SDindex', 'SDbw']
    
    answer = nbclust.NbClust(data, distance=distance,
            min_nc=min_nc, max_nc=max_nc,
            method=method, index=index)
    
    index_df = rmatrix_2df(answer.__getitem__(0))
    best_nc_df = rmatrix_2df(answer.__getitem__(2))
    
    return NbClustResults(index_df, best_nc_df)
    

"""Records
#all_index = rmatrix_2df(answer[0])
#all_crit = rmatrix_2df(answer[1])
#best_nc = rmatrix_2df(answer[2])
#best_partition = np.array(answer[3])

#Save for later - could use with other R libraries
#_module_path = rpy2.robjects.packages.get_packagepath('NbClust')
#nbclust = importr('NbClust', lib_loc=_module_path)
"""