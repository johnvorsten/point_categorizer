# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:43:14 2019

@author: z003vrzk
"""

"""This class will use known or predicted number of clusters for a database
to partition the database into their partitions.
For example a database with point names :
BANC.RTU0202.CCT
BANC.EF0101.PRF
BANC.EF0101.RMT
BANC.EF0101.SS
BANC.EF0201.PRF
BANC.EF0201.SS
should be separated into BANC.RTU0202*, BANC.EF101*, BANC.EF0201* groups"""

# Third party imports
from pymongo import MongoClient
import pymongo
from kmodes.kmodes import KModes
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

# Local imports
from point_database_categorizer.JVWork_WholeDBPipeline import JVDBPipe



#%% Inserting documents into new collection after they are clustered

def insert_document_to_document(document, field, insert_item, collection):
    """Inserts a field into an existing document
    inputs
    -------
    document : (dict) document from mongodb to which you want to insert
    field : (str) string of field name
    insert_dict : (dict) dict to insert under field
    collection : (pymongo.Collection.collection) collection that owns document"""
    
    document_id = document['_id']
    result = collection.update_one({'_id':document_id}, {'$set':{field:insert_item}})
    
    return result.modified_count


# Instantiate local classes
myDBPipe = JVDBPipe()
# Create 'clean' data processing pipeline
clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False, 
                                      replace_numbers=False, 
                                      remove_virtual=True)

# Create pipeline specifically for clustering text features
text_pipe = myDBPipe.text_pipeline(vocab_size='all', 
                                   attributes='NAME',
                                   seperator='.',
                                   heirarchial_weight_word_pattern=True)

# Retrieve information from Mongo
client = MongoClient('localhost', 27017)
db = client['master_points']
collection_raw = db['raw_databases']
collection_clustered = db['clustered_points']

_cursor = collection_raw.find()
for document in _cursor:
    
    # pass data through cleaning and text pipeline
    database = pd.DataFrame.from_dict(document['points'], 
                                      orient='columns')
    
    # Test if document already exists
    exists_cursor = collection_clustered.find({'database_tag':document['database_tag']}, {'_id':1})
    if exists_cursor.alive:
        # Dont recalculate
        continue
    
    df_clean = clean_pipe.fit_transform(database)
    X = text_pipe.fit_transform(df_clean).toarray()
    #_word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
    #df_text = pd.DataFrame(X, columns=_word_vocab)
    
    
    # perform clustering
    actual_n_clusters = document['correct_k']
    if actual_n_clusters == 1:
        # Dont cluster - why would you? 
        prediction_agglo = np.ones((X.shape[0]))
        
    if X.shape[0] <= 3:
        # Dont cluster - just pass 1 cluster total
        prediction_agglo = np.ones((X.shape[0]))
        
    else:
        # Cluster
        agglomerative = AgglomerativeClustering(n_clusters=actual_n_clusters, 
                                                affinity='euclidean',
                                                linkage='ward')
        prediction_agglo = agglomerative.fit_predict(X)
    
    # Format of clustered_points_dict is {setid:}
    clustered_points_list = []
    for setid in set(prediction_agglo):
        cluster_dict = {}
        indicies = (prediction_agglo==setid)
        cluster_names = df_clean.iloc[indicies]
        points_dict = cluster_names.to_dict(orient='list')
        cluster_dict['points'] = points_dict
        
        # Add entries into clustered_points_dict
        clustered_points_list.append(cluster_dict)

    try:
        database_tag = document['database_tag']
        collection_clustered.insert_one({'clustered_points':clustered_points_list,
                                         'database_tag':database_tag,
                                         'correct_k':actual_n_clusters})
        
    except pymongo.errors.WriteError:
        continue

#%% TestUnsupervisedClusterPoints
        
from UnsupervisedCluster import UnsupervisedClusterPoints

# Create list of 5 best hyperparam dicts
best_hyperparam_list = [{'by_size': False,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Ratkowsky'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Cindex'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'CCC'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Silhouette'},
 {'by_size': True,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'Hartigan'}]



client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

_cursor = collection.find()

#for document in _cursor:
document = next(_cursor)

myClusterer = UnsupervisedClusterPoints()
database_iterator = myClusterer.split_database_on_panel(document)
#for sub_database in database_iterator:
sub_database = next(database_iterator)

my_pipeline = myClusterer.make_cleaning_pipe(remove_dupe=False,
                 replace_numbers=False, 
                 remove_virtual=True, 
                 vocab_size='all', 
                 attributes='NAME', 
                 seperator='.',
                 heirarchial_weight_word_pattern=True)
database, df_clean, X = my_pipeline(sub_database, 
                                    input_type='DataFrame')




test_list = [{'by_size': False,
  'distance': 'euclidean',
  'clusterer': 'ward.D',
  'n_components': 8,
  'reduce': 'MDS',
  'index': 'all'}]

result = myClusterer.cluster_with_hyperparameter_list(test_list, 
                                                      X)




