# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:43:28 2019

@author: z003vrzk
"""
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from gap_statistic import optimalK
from JVWork_WholeDBPipeline import JVDBPipe
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

""" Example usage of this module

#Intantiate the class
myDBPipe = JVDBPipe()
myClustering = JVClusterTools()

#Load DataBase from memory
_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
my_iter = myClustering.read_database_set(_master_pts_db)
ind, df = next(my_iter)

#Apply custom cleaning/alteration pipelines to data
df_clean = myDBPipe.cleaning_pipeline(df, remove_dupe=False, 
                                      replace_numbers=False, remove_virtual=True)
df_text = myDBPipe.text_pipeline(df_clean, vocab_size='all')

#Create a vocabulary or words for human comprehension
_vocabulary = df_text.columns.tolist()

_df_systems = df_clean['SYSTEM']
X = df_text.values

#Create the data, one system at a time for manual labeling/inspection
group_iterator = myClustering.get_database_set(_df_systems, X)
_indicies, X = next(group_iterator)

#Visualize some data
_point_names = myClustering.get_word_name(X, _vocabulary)
print(df_clean.loc[_indicies[0], 'DBPath'])
for _i in range(0, min(5, len(_point_names))):
    print(_point_names[_i])
"""
##OR##
"""
myDBPipe = JVDBPipe()
myClustering = JVClusterTools()

_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
_master_pts_db_clean = r"D:\Z - Saved SQL Databases\master_pts_db_clean.csv"
_ptname_hot_path = r"D:\Z - Saved SQL Databases\master_pts_db_name_onehot.csv"
_sys_hot_path = r"D:\Z - Saved SQL Databases\master_pts_db_system_onehot.csv"
_desc_hot_path = r"D:\Z - Saved SQL Databases\master_pts_db_description_onehot.csv"

my_iter = myClustering.read_database_set(_master_pts_db)
ind, df = next(my_iter)

df_clean = myDBPipe.cleaning_pipeline(df, remove_dupe=False, 
                                      replace_numbers=False, remove_virtual=True)

df_text = myDBPipe.text_pipeline(df_clean, vocab_size='all')

_vocabulary = df_text.columns.tolist()

X = df_text.values

#Visualize some data
_point_names = myClustering.get_word_name(X, _vocabulary)
print(df_clean.loc[0, 'DBPath'])
for _i in range(0, min(5, len(_point_names))):
    print(_point_names[_i])
"""

class JVClusterTools():
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_database_set(names, features):
        """Generates an iterator which yields dataframes that all have a common
        attribute in common. Each names and features must have the same length,
        where every feature/row in features must correspond to a name in names
        parameters
        -------
        dbnames : a dataframe, list, or np.array with the names corresponding to
            the features we want to yield
        features : a dataframe, list, or np.array with the features corresponding
            to dbnames
        returns
        -------
        an iterator over featuers that returns a np.array of features[index,:"]
        where index is determined by all indicies in names taht have a common name/label"""
        assert len(features) == len(names), 'Features and names must be same length'
        
        if type(features) == pd.DataFrame:
            features = features.values
        
        if type(names) == pd.DataFrame or pd.Series:
            names = names.values
        
        if type(names) == list:
            names = np.array(names)
        
        unique_names = list(set(names.flat))
        
        for name in unique_names:
            
            indicies = np.where(names==name)[0]
            feature_rows = features[indicies,:]
            
            yield indicies, feature_rows
    
    @staticmethod
    def read_database_set(database_name, column_name='DBPath'):
        """Yields sequential data from memory.
        parameters
        -------
        database_name : path to csv database (string)
        column_tag : column name that contains labels for each sequential set.
            Must be included on each row.
        output
        -------
        iterator over a database grouped by a common column_tag
        yield (indicies, sequence).
        indicies : indicies of pandas dataframe
        sequence : pandas dataframe of database
        
        Example
        my_iter = read_database_set(db_path, column_tag='DBPath')
        'ind, df = next(my_iter)
        print(ind[0],":",ind[-1], " ", df['DBPath'][0])"""
    
        csv_iterator = pd.read_csv(database_name,
                                   index_col=0,
                                   iterator=True,
                                   chunksize=50000,
                                   encoding='mac_roman'
                                   )
        for chunk in csv_iterator:
            
            partial_set = set(chunk[column_name])
            unique_names = list(partial_set)
            
            for name in unique_names:
                
                indicies = np.where(chunk[column_name]==name)[0]
                sequence = chunk.iloc[indicies]
                
                yield indicies, sequence
    
    @staticmethod
    def read_database_ontag(file_path, column_name, column_tag):
        """Let Y denotate the label space. X denotates the instance space.
        Retrieves all axis-0 indicies of column_tag in column_name. This is 
        useful for retrieving all instances in {(Xi, yi) | 1<i<m} whose yi
        match column_tag (assuming column_tag is in the space of Y).
        parameters
        -------
        file_path : path to file
        column_name : column that contains all yi for 1<i<m
        column_tag : value from Y to match for each yi in 1<i<m"""
        
        df = pd.read_csv(file_path, 
                         index_col=0, 
                         usecols=[column_name],
                         encoding='mac_roman')
        
        cols = pd.read_csv(file_path, 
                           index_col=0,
                           encoding='mac_roman',
                           nrows=0).columns.tolist()
        indicies = np.where(df.index == column_tag)[0] + 1
        
        df_whole = pd.read_csv(file_path, 
                         names=cols,
                         encoding='mac-roman',
                         skiprows = lambda x: x not in indicies)
        df_whole.reset_index(drop=True, inplace=True)
        return df_whole
        
    
    
    @staticmethod
    def get_word_name(features, vocabulary):
        """Prints the associated words of a one-hot encoded text phrase
        from the vocabulary. Assumes the order of features and vocabulary
        is in the same order
        parameters
        -------
        features : one-hot encoded feature vector (single vector or array). Must
            be of type np.array or pd.DataFrame
        vocabulary : list or np.array of strings
        output
        -------
        words : nested list of decoded words"""
        assert features.shape[1] == len(vocabulary), 'Features and Vocab must be same length'
        
        if type(features) == pd.DataFrame:
            features = features.values
        
        if type(vocabulary) == pd.DataFrame:
            vocabulary = vocabulary.values
        
        if type(vocabulary) == list:
            vocabulary = np.array(vocabulary)
            
        words = []
        for vector in features:
            
            indicies = np.where(vector==1)[0]
            words_iter = vocabulary[indicies]
            words.append(words_iter)
            
        return words
    
    @staticmethod
    def optimalK2(data, nrefs=3, maxClusters=15):
        """
        Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'n_clusters':[], 
                                  'gap_value':[],  'gap*':[], 
                                  'obs_dispersion':[], 
                                  'ref_dispersion':[] })
        for gap_index, k in enumerate(range(1, maxClusters)):
    
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)
    
            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)
                
                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)
                
                refDisp = km.inertia_
                refDisps[i] = refDisp
    
            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data)
            
            origDisp = km.inertia_
            refDisp_mean = np.mean(refDisps)
    
            # Calculate gap statistic
            gap = np.mean(np.log(refDisps)) - np.log(origDisp) 
            gap_star = np.mean(refDisps) - origDisp
    
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap
            
            resultsdf = resultsdf.append({'n_clusters':k, 'gap_value':gap, 
                                          'ref_disp':refDisps, 'obs_dispersion':origDisp,
                                          'ref_dispersion':refDisp_mean, 'gap*':gap_star}, ignore_index=True)
    
        return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal

