# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:57:56 2019

A collection is the equivalent to a table in a relational database

#Make a connection
from pymongo import MongoClient
client = MongoClient('localhost', 27017)

#Get a database
#db= client.test_database
db = client['test_database']

#Get a collection
collection = db['test_collection']

#Storing data into a collection
mydata = {'database tag':r'D:\path\to\database',
          'point1':{'column1':1, 'column2':2, 'column3':'pointname'},
          'other_tag':'some tag'}

mydata2 = [{'POINTID': 4,
             'NETDEVID': np.nan,
             'NAME': 'NAMC.L01.AHU04.MAO',
             'CTSYSNAME': np.nan,
             'DESCRIPTOR': 'MIX AIR OUTPUT'},
             {"author": "Eliot",
            "title": "MongoDB is fun",
            "text": "and pretty easy too!",
            "date": datetime.datetime(2009, 11, 10, 10, 45)}]

#Create a collection
mycollection = db.points_database 
mycollection_id = mycollection.insert_one(mydata).inserted_id
mycollection.insert_many(mydata2)

#List all collections on a database
db.list_collection_names()


@author: z003vrzk
"""

#Import 3rd party libraries
import pandas as pd
import numpy as np
import pymongo
import datetime
from pymongo import MongoClient

#Local imports
from JVWork_Mongo import (df2mongo, mongo2df)
from JVWork_UnsupervisedCluster import JVClusterTools
myClustering = JVClusterTools()
client = MongoClient('localhost', 27017)

#%%
#Save all databases in mongodb

#Import your database (.csv)
csv_file = r'data\master_pts_db.csv'
sequence_tag = 'DBPath'

#Get unique names
unique_tags = pd.read_csv(csv_file, index_col=0, usecols=['DBPath'])
unique_tags = list(set(unique_tags.index))

#Save all databasese to mongodb
for column_tag in unique_tags:
    database = myClustering.read_database_ontag(csv_file, 
                                                sequence_tag, 
                                                column_tag)
    df2mongo(database, client)

#%% Creating a unique index on a text field
# This will not work becasue text fields are divided into keys
# And the set of keys is used as an index - this means if two
# fields have overlapping keys there will be duplicate index



data1 = {'database_tag':'test_tag_1',
         '_fts':'1',
         '_ftsx':'2',
         'other_label':23423}
data2 = {'database_tag':'test_tag_1',
         '_fts':'5',
         '_ftsx':'6',
         'other_label':452}

db_test = client['test_database']
collection_test = db_test['points_database']
collection_test.create_index([('database_tag', pymongo.TEXT)], 
                              name='custom_index',
                              unique=True)

collection_test.index_information()

collection_test.insert_one(data1)
collection_test.insert_one(data2)


a = collection.aggregate([
        {'$group':{
                '_id':{'database_tag':'$database_tag'},
                'dups':{'$push':'$_id'},
                'count':{'$sum':1}
                }
        },
        {'$match': {
                'count':{'$gt':1}
                }
        }
        ])

for document in a:
    #Get the collection ids
    ids = document['dups']
    
    #Choose the trailing ids and delete
    for dupid in ids[:-1]:
        ack = collection.delete_one({'_id':dupid})
        if ack:
            print(f'{dupid} deleted')
        else:
            print(f'{dupid} not found')


bb = collection.find({'database_tag':{'$regex':'134749'}})
for document in bb:
    print(document['database_tag'])


db = client['master_points']
collection = db['raw_databases']

collection.create_index([('database_tag',pymongo.TEXT)], 
                         name='custom_index',
                         unique=True,
                         default_language='none')

collection.index_information()
#collection.drop_index('database_tag')








