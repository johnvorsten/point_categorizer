# -*- coding: utf-8 -*-
"""
Created on Sat May 30 16:29:46 2020

@author: z003vrzk
"""

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
    labels, hyper_dict = Extract.calc_labels(records, tag)

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

# This document has no points and causes errors
collection.delete_one({'database_tag':r'D:\Z - Saved SQL Databases\44OP-263742-AUS_TFC_TRC_Renovation\JobDB.mdf'})
pass

#%% Save by_size label

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


for document in collection.find():
    # Extract saved labels
    labels = document['hyper_labels']

    ranked_item = []
    for key, label in labels.items():
        ranked_item.append(label['index'])

    collection.update_one({'_id':document['_id']},
                           {'$set':{'best_hyper.index':ranked_item}})


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