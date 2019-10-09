# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:50:45 2019

# Example Usage
# Local Imports
from JVWork_UnsupervisedCluster import JVClusterTools
myClustering = JVClusterTools()

# Import your database (.csv)
csv_file = r'data\master_pts_db.csv'
sequence_tag = 'DBPath'
column_tag = r'D:\Z - Saved SQL Databases\44OP-131383_SDMC_Surgery_Reno_PH2&3\JobDB.mdf'

database = myClustering.read_database_ontag(csv_file, 
                                            sequence_tag, 
                                            column_tag)

# Create a database client
client = MongoClient('localhost', 27017)

# Convert your .csv database to a document in a collection in a database
df2mongo_raw(database, client)

# Convert a document in a collection in a mongo database to a dataframe
df_retrieve = mongo2df(column_tag, client, collection='raw_databases', mongodb='master_points')

# Retrieve pickled/encoded types
# Gets list of [clusterer, index] names
label_cat = pickle.loads(document['encoded_hyper']['clust_index']['cat'])
# Creates list of names [clusterer & index]
label_cat = np.concatenate((label_cat[0], label_cat[1]))
# Get encoded vector (numpy array)
labels = pickle.loads(document['encoded_hyper']['clust_index']['val'])




@author: z003vrzk
"""



"""Pandas to mongo example"""

from pymongo.errors import ConfigurationError
from pymongo import MongoClient
from pymongo.collection import Collection
import pandas as pd
import numpy as np
import pickle

def df2mongo_raw(dataframe, client):
    """Save a dataframe to a mongo database collection. Function is specially suited
    for my database. Collection and database names are handled by function
    inputs
    -------
    dataframe : a pandas dataframe to be converted to mongo collection object
    client : a  mongo db client object
    output
    -------
    TODO : return an error on failure for some reason"""

    default_collections = ['raw_databases'] #TODO Expand as necessary
    default_dbs = ['master_points']
    
    
    default_structure = {'database_tag':0,
                     'points':0
                     }
    sequence_tag = 'DBPath'
    col_idx = dataframe.columns.to_list().index(sequence_tag)
    database_tag = dataframe.iloc[0, col_idx]
    default_structure['database_tag'] = database_tag
    
    #Old Structure
#    for idx, row in dataframe.iterrows():
#        row_dict = row.to_dict()
#        #Drop certain tags (DBPath)
#        row_dict.pop('DBPath')
#        #Use POINTID to number each nested document
#        pointid = str(row_dict['POINTID'])
#        #Insert documents into the default structure
#        default_structure['points'][pointid] = row_dict
    
    df_dict = dataframe.to_dict()
    for key, subdict in df_dict.items():
        df_dict[key] = list(df_dict[key].values())
        
    default_structure['points'] = df_dict
    
    #Get the database 
    db = client[default_dbs[0]]
    #Get the collection
    collection = db[default_collections[0]]
    #Insert the document into the collection
    collection.insert_one(default_structure)
    
    return True

def mongo2df(database_tag, client, 
             collection='raw_databases', 
             mongodb='master_points'):
    """Retrieve a dataframe from a mongodb database collection. Function is 
    specially suited for my database. Either a) client object is passed or b)
    a collection object is passed, and client and mongodb are None
    inputs
    -------
    database_tag : string of unique database tag
    client : a  mongo db client object
    collection : string or collection object. If a mongodb collection object is passed
    then client and mongodb must be None
    mongodb : 
    output
    -------
    dataframe : """
    
    default_collections = ['raw_databases'] #TODO Expand as necessary
    default_dbs = ['master_points']
    
    if isinstance(client, MongoClient):
        assert isinstance(collection, str), ("client object was passed with \
            invalid collection argument")
        assert isinstance(mongodb, str), ("client object was passed with \
            invalid mongodb argument")
        assert default_collections.__contains__(collection)
        assert default_dbs.__contains__(mongodb)
            
        db = client[mongodb]
        _collection = db[collection]
        mydict = _collection.find_one({'database_tag':database_tag})
        points = mydict['points']
#        #Convert points keys to the POITNID key it contains
#        key1 = next(points.__iter__())
#        value1 = points[key1]
#        columns = list(value1.keys())
        #Create all unique columns
        
        dataframe = pd.DataFrame.from_dict(points, orient='columns')
            
    if isinstance(collection, Collection):
        assert client is None, """client object was passed with  
            invalid collection argument"""
        assert mongodb is None, """client object was passed with  
            invalid mongodb argument"""
        
        mydict = collection.find_one({'database_tag':database_tag})
        points = mydict['points']
        dataframe = pd.DataFrame.from_dict(points, orient='columns')
    
    return dataframe


def correct_encoding(dictionary):
    """Correct the encoding of python dictionaries so they can be encoded to
    mongodb
    inputs
    -------
    dictionary : dictionary instance to add as document
    output
    -------
    new : new dictionary with (hopefully) corrected encodings"""
    
    new = {}
    for key1, val1 in dictionary.items():
        # Nested dictionaries
        if isinstance(val1, dict):
            val1 = correct_encoding(val1)
            
        if isinstance(val1, np.bool_):
            val1 = bool(val1)
            
        if isinstance(val1, np.int64):
            val1 = int(val1)
            
        if isinstance(val1, np.float64):
            val1 = float(val1)

        new[key1] = val1
    
    return new


def get_batch(collection, 
              batch_size, 
              feature_name='db_features', 
              label_name='best_hyper.multiclass'):
    """Retrieve an instance {features, labels} from MongoDB.
    collection : a mongo collection object used for querying the database
    batch_size : number of documents to retrieve
    feature_name : Leave at default. 
    label_name : One of ('best_hyper.multiclass', 'best_hyper.ranking').
    'best_hyper.multiclass' (returns the by_size, n_components, reduce) hyperparameters
    'best_hyper.ranking' returns the clusterer and index pairs
    
    Example Usage
    
    from pymongo import MongoClient
    
    client = MongoClient('localhost', 27017)
    db = client['master_points']
    collection = db['raw_databases']
    
    feature_df, label_df = get_batch(collection, 
                                    batch_size=5, 
                                    label_name='best_hyper.ranking')
    """
    
    # Enforce column order
    feature_cols = ['n_instance', 'n_features', 'len_var', 'uniq_ratio', 
                    'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
                    'n_len6', 'n_len7']
    label_cols = ['by_size', 'n_components', 'reduce']
    label_cols_other = ['clusterer', 'index']
    
    enforce_dict = {'db_features':feature_cols,
                    'single_hypers':label_cols,
                    'rank_hypers':label_cols_other}
    
    cursor = collection.aggregate([
            {'$sample':{'size':batch_size}},
            {'$project':{'db_features':True, 'best_hyper':True, 'database_tag':True}}
            ])
    
    features_df_final = pd.DataFrame()
    labels_df_final = pd.DataFrame()
    
    for document in cursor:
        db_tag = [document['database_tag']]
        features = document['db_features']
        labels = document['best_hyper']
        
        # Get Features
        features_df = pd.DataFrame.from_records(features,
                                                columns=enforce_dict[feature_name], 
                                                index=db_tag)
        features_df_final = features_df_final.append(features_df)
        
        # Get easy labels
        if label_name == 'best_hyper.multiclass':
            labels_df = pd.DataFrame.from_records(labels,
                                                  columns=enforce_dict['single_hypers'],
                                                  index=db_tag)
            labels_df_final = labels_df_final.append(labels_df)
        
        if label_name == 'best_hyper.ranking':
        # Get array labels; Harder becasue of nested arrays
            clusterer = document['best_hyper']['clusterer']
            clusterer_df = pd.DataFrame.from_records(data=clusterer,
                                                     columns=clusterer.keys(),
                                                     index=db_tag)
            clusterer_df = clusterer_df.append(clusterer_df)
            
            indicies = document['best_hyper']['index']
            index_df = pd.DataFrame.from_records(data=indicies,
                                                     columns=indicies.keys(),
                                                     index=db_tag)
            labels_df_final = labels_df_final.append(index_df)
#            print(labels_df_final)
    
    if label_name == 'best_hyper.multiclass':
        return features_df_final, labels_df_final
        
    elif label_name == 'best_hyper.ranking':
        return features_df_final, labels_df_final
    
    else:
        raise NameError('Incorrect input for label_name')
        
    return


def get_encoded_batch(collection, 
              batch_size, 
              label_name='encoded_hyper.by_size'):
    """Retrieve an encoded instance {features, labels} from MongoDB.
    inputs
    -------
    collection : a mongo collection object used for querying the database
    batch_size : number of documents to retrieve
    label_name : One of ('encoded_hyper.by_size', 'encoded_hyper.n_components',
    'encoded_hyper.reduce', 'encoded_hyper.clust_index').
    'encoded_hyper.by_size' - encoded by_size hyperparameter
    'encoded_hyper.n_components' - encoded n_components hyperparameter
    'encoded_hyper.reduce' - encoded reduce hyperparameter
    'encoded_hyper.clust_index' - encoded clusterer and index pair hyperparameter
    output
    -------
    features, feat_cat, labels, label_cat (numpy arrays)
    
    Example Usage
    
    from pymongo import MongoClient
    
    client = MongoClient('localhost', 27017)
    db = client['master_points']
    collection = db['raw_databases']
    
    features, feat_cat, labels, label_cat = get_encoded_batch(collection, 
                                    batch_size=5, 
                                    label_name='encoded_hyper.clust_index')
    """
    
    # Enforce column order
    feature_cols = ['n_instance', 'n_features', 'len_var', 'uniq_ratio', 
                    'n_len1', 'n_len2', 'n_len3', 'n_len4', 'n_len5',
                    'n_len6', 'n_len7']
    
    enforce_dict = {'db_features':feature_cols}
    feature_name='db_features'
    
    cursor = collection.aggregate([
            {'$sample':{'size':batch_size}},
            {'$project':{'db_features':True, 'encoded_hyper':True, 'database_tag':True}}
            ])
    
    features_df_final = pd.DataFrame()
    
    for document in cursor:

        db_tag = [document['database_tag']]
        features = document['db_features']
        
        # Get Features
        features_df = pd.DataFrame.from_records(features,
                                                columns=enforce_dict[feature_name], 
                                                index=db_tag)
        features_df_final = features_df_final.append(features_df)
        features = features_df_final.values
        feat_cat = np.array(features_df_final.columns, dtype=np.unicode_)
        
        # Get Labels
        if label_name == 'encoded_hyper.reduce':
            label_cat = pickle.loads(document['encoded_hyper']['reduce']['cat'])
            labels = pickle.loads(document['encoded_hyper']['reduce']['val'])
        
        if label_name == 'encoded_hyper.by_size':
            label_cat = pickle.loads(document['encoded_hyper']['by_size']['cat'])
            labels = pickle.loads(document['encoded_hyper']['by_size']['val'])
            
        if label_name == 'encoded_hyper.n_components':
            label_cat = pickle.loads(document['encoded_hyper']['n_components']['cat'])
            labels = pickle.loads(document['encoded_hyper']['n_components']['val'])
    
        if label_name == 'encoded_hyper.clust_index':
            label_cat = pickle.loads(document['encoded_hyper']['clust_index']['cat'])
            label_cat = np.concatenate((label_cat[0], label_cat[1]))
            labels = pickle.loads(document['encoded_hyper']['clust_index']['val'])
    
    return features, feat_cat, labels, label_cat


















