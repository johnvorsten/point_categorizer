# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:23:50 2019

@author: z003vrzk
"""
#Third party imports
import pandas as pd
from sklearn.pipeline import Pipeline
from pymongo import MongoClient
import numpy as np

#Local Imports
from JVWork_Labeling import ExtractLabels
from JVWork_UnsupervisedCluster import JVClusterTools
from JVWork_WholeDBPipeline import JVDBPipe
from JVWork_AccuracyVisual import import_error_dfs

Extract = ExtractLabels()
ClusterTools = JVClusterTools()
myDBPipe = JVDBPipe()


#%% Setup MongoDB connection
client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']


#%%
# Extract features for all datasets

#Import your database (.csv)
csv_file = r'data\master_pts_db.csv'
sequence_tag = 'DBPath'

#Get unique names
unique_tags = pd.read_csv(csv_file, index_col=0, usecols=['DBPath'])
unique_tags = list(set(unique_tags.index))

#Import pipelines for use in calculating features
text_pipe = myDBPipe.text_pipeline(vocab_size='all', attributes='NAME',
                           seperator='.')
clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False, 
                              replace_numbers=False, 
                              remove_virtual=True)
mypipe = Pipeline([('clean_pipe', clean_pipe),
                   ('text_pipe',text_pipe)
                   ])


for tag in unique_tags:
    #Extract database
    database = ClusterTools.read_database_ontag(csv_file, 'DBPath', tag)
    
    #Calculate Features
    db_feat = Extract.calc_features(database, mypipe, tag=tag)
    db_feat = db_feat.to_dict()
    
    #Remote the nested dictinoary in db_feat pandas creates
    for key, subdict in db_feat.items():
        db_feat[key] = list(db_feat[key].values())
    assert db_feat['instance'][0] == tag, 'Non-matching tag'
    
    #Remove unnecessary informatino. 'instance' matches 'tag' and 'database_tag'
    db_feat.pop('instance') #Dont need to duplicate this information
    
    #Save labels for all datasets in mongo
    collection.update_one({'database_tag':tag}, {'$set':{'db_features':db_feat}})

#Make sure everything is tagged
a = collection.find( {'db_features':{'$exists':False}} )
a = collection.aggregate([
        {'$match': {'db_features': {'$exists':False} } },
        {'$group': {'_id': {'my_tag':'$database_tag'},
                    'obj_id': {'$push':'$_id'} 
                    }
        }
        ])




#%% Extract labels for all datasets
from pymongo.errors import InvalidDocument
from JVWork_Mongo import correct_encoding
from JVWork_Labeling import ExtractLabels
from JVWork_AccuracyVisual import import_error_dfs
Extract = ExtractLabels()

# New Database
db = client['master_points']
collection = db['raw_databases']

#Import your database (.csv)
csv_file = r'data\master_pts_db.csv'
sequence_tag = 'DBPath'

#Get unique names
unique_tags = pd.read_csv(csv_file, index_col=0, usecols=['DBPath'])
unique_tags = list(set(unique_tags.index))

records = import_error_dfs()

for tag in unique_tags:

    #Calculate Features
    new_labels = {}
    labels, hyper_dict = Extract.calc_labels(records, tag, best_n='all')
    
    for key in labels.keys():
        new_labels[str(key)] = labels[key]
        
    #Save labels for all datasets in mongo
    try:
        collection.update_one({'database_tag':tag}, {'$set':{'hyper_labels':new_labels}})
    except InvalidDocument:
        # Python on Windows and Pymongo are not forgiving
        # If you have foreign data types you have to convert them
        corrected_dict = correct_encoding(new_labels)
        collection.update_one({'database_tag':tag}, 
                              {'$set':{'hyper_labels':corrected_dict}})
        
#Make sure everything is tagged
a = collection.find( {'hyper_labels':{'$exists':False}} )

a = collection.aggregate([
        {'$match': {'hyper_labels': {'$exists':False} } },
        {'$group': {'_id': {'my_tag':'$database_tag'},
                    'obj_id': {'$push':'$_id'} }}])
    
a = collection.aggregate([
        {'$group': {'_id':'$database_tag', 'hypers':{'$first':'$hyper_labels'}}},
        {'$project': {'num_fields':{'$size':{'$objectToArray':'$hypers'}}}},
        {'$match': {'num_fields':{'$lt':30}}}
        ])

collection.delete_one({'database_tag':r'D:\Z - Saved SQL Databases\44OP-263742-AUS_TFC_TRC_Renovation\JobDB.mdf'})
pass


#%% by_size

#Third party imports
from pymongo import MongoClient

# Local Imports
from JVWork_Labeling import choose_best_hyper
    
# New Database
client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

# Goal : choose the best by_size hyperparameter for a database

for db_dict in collection.find():
    # Extract saved labels
    labels = db_dict['hyper_labels']
    
    best_by_size = choose_best_hyper(labels, 'by_size')
    
    collection.update_one({'_id':db_dict['_id']},
                          {'$set':{'best_hyper.by_size':best_by_size}})



a = collection.aggregate([
        {'$group': {'_id':'$database_tag', 'hypers':{'$first':'$hyper_labels'}}},
        {'$project': {'num_fields':{'$size':{'$objectToArray':'$hypers'}}}},
        {'$match': {'num_fields':{'$lt':30}}}
        ])
    
b = collection.aggregate([
        {'$group':{'_id':'$database_tag', 'by_size':{'$first':'$best_hyper.by_size'}}}
        ])
pass

for doc in a:
    print(doc['num_fields'])  
    
for doc in b:
    print(doc['by_size'])
    

#%% Clusterer OR Index
    
# TODO : ranking system

for db_dict in collection.find():
    # Extract saved labels
    labels = db_dict['hyper_labels']
    
    ranked_item = []
    for key, label in labels.items():
        ranked_item.append(label['index'])
    
    collection.update_one({'_id':db_dict['_id']}, 
                           {'$set':{'best_hyper.index':ranked_item}})


for i in range(1, len(labels)+1):
    print(labels[str(i)]['clusterer'], ranked_clusterer[i-1])

#%% n_components
    
for db_dict in collection.find():
    # Extract saved labels
    labels = db_dict['hyper_labels']
    
    best_n_components = choose_best_hyper(labels, 'n_components')
    
    collection.update_one({'_id':db_dict['_id']}, 
                           {'$set':{'best_hyper.n_components':best_n_components}})
    
a = collection.aggregate([
        {'$group': {'_id':'$database_tag', 'hypers':{'$first':'$best_hyper'}}},
        {'$project': {'num_fields':{'$size':{'$objectToArray':'$hypers'}}}},
        {'$match': {'num_fields':{'$lt':2}}}
        ])
    
b = collection.aggregate([
        {'$group':{'_id':'$database_tag', 'n_components':{'$first':'$best_hyper.n_components'}}}
        ])
pass

for doc in a:
    print(doc['num_fields'])  
    
for doc in b:
    print(doc['n_components'])
    
#%% reduce

for db_dict in collection.find():
    # Extract saved labels
    labels = db_dict['hyper_labels']
    
    best_reduce = choose_best_hyper(labels, 'reduce')
    
    collection.update_one({'_id':db_dict['_id']}, 
                           {'$set':{'best_hyper.reduce':best_reduce}})
    
a = collection.aggregate([
        {'$group': {'_id':'$database_tag', 'hypers':{'$first':'$best_hyper'}}},
        {'$project': {'num_fields':{'$size':{'$objectToArray':'$hypers'}}}},
        {'$match': {'num_fields':{'$lt':3}}}
        ])
    
b = collection.aggregate([
        {'$group':{'_id':'$database_tag', 'reduce':{'$first':'$best_hyper.reduce'}}}
        ])
pass

for doc in a:
    print(doc['num_fields'])  
    
for doc in b:
    print(doc['reduce'])


#%% Save encoded labels
pass

from JVWork_Labeling import get_unique_labels

from pymongo import MongoClient
from bson.binary import Binary
import pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np

unique_labels = get_unique_labels()
client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']

# TODO index, clusterer

clust_uniq = sorted(unique_labels['clusterer'])
ind_uniq = sorted(unique_labels['index'])
#a = [clust_uniq, ind_uniq]

one_hot = OneHotEncoder(categories=[clust_uniq, ind_uniq])
#one_hot.fit([clust_uniq, ind_uniq])
#one_hot.fit(np.array(clust_uniq).reshape(-1,1))
#one_hot.fit(np.array(ind_uniq).reshape(-1,1))

#one_hot.categories_ = [np.array(clust_uniq), np.array(ind_uniq)]

for db_dict in collection.find():
    # Extract saved labels
    clust_labels = list(db_dict['best_hyper']['clusterer']) # Maintains order
    idx_labels = list(db_dict['best_hyper']['index'])
    
    labels_array = np.array([clust_labels, idx_labels]).transpose()
    
    encoded = one_hot.fit_transform(labels_array).toarray()
    categories = one_hot.categories_
    encoded_pickle = Binary(pickle.dumps(encoded, protocol=2), subtype=128)
    cat_pickle = Binary(pickle.dumps(categories, protocol=2), subtype=128)
    
    collection.update_one({'_id':db_dict['_id']}, 
                           {'$set':{'encoded_hyper.clust_index.cat':cat_pickle,
                                    'encoded_hyper.clust_index.val':encoded_pickle}})





# Retrieve
categories = pickle.loads(db_dict['encoded_hyper']['by_size']['cat'])












