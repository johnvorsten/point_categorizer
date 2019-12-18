# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:11:14 2019

# Find all documents with duplicate field values
# The field is <database_tag> and we want to find duplicate values on that field
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
    #Choose the trailign ids and delete

    for dupid in ids[:-1]:
        ack = collection.delete_one({'_id':dupid})
        if ack:
            print(f'{dupid} deleted')
        else:
            print(f'{dupid} not found')


collection.aggregate([
                    {<pipeline1>},
                    {<pipeline2>}
                    ])

#The '_id' field is mandatory. I optionally chose to define it as a object/document
# Instead, I could do {'_id':'database_tag'} instead of {'_id':{<field>:<operator>}}
# Note, the <operator_expression>:<argument> can also be called an aggregator
<pipeline1> = {<aggregator>: {
                    '_id' : {<field>:<operator>},
                    <field> : {<operator_expression>:<argument>},
                    <field> : {<operator_expression>:<argument>} 
                            }
                }

# Find documents where a field does not exist
# Then find the document _id of the document with a missing field
# Note : $group may not be necessary
a = collection.aggregate([
    {'$match': {'db_features': {'$exists':False} } },
    {'$group': {'_id': {'my_tag':'$database_tag'},
                'obj_id': {'$push':'$_id'} 
                }
    }
    ])

# A simple nested query
bb = collection.find({'points.POINTID.0':9})

# A simple query on a nested array
c = collection.find({
        'points.DESCRIPTOR': {'$all':['MIX AIR OUTPUT']}
        })

c = collection.find({
        <field>: {<query_operator>:[<value1>,<value2>]}
        })

# A query to see if a field exists
collection.find({'db_features':{'$exists':True}})
collectino.find({<field>:{<operator>:<argument>}})
    
# Group a collection based on a specific field in its documents
# Pass through a document/field ($first is useful here)
# Find the number of fields in a document or nested field
# Match fields of a certain length
a = collection.aggregate([
        {'$group': {'_id':'$database_tag', 'hypers':{'$first':'$hyper_labels'}}},
        {'$project': {'num_fields':{'$size':{'$objectToArray':'$hypers'}}}},
        {'$match': {'num_fields':{'$lt':30}}}
        ])
pass

# Pass along documents with the requested fields to the next stage of the pipeline
# Can be existing documents from input fields
# Get a random number of documents
# Only keep the fields feature_name and label_name in the final aggergation
cursor = collection.aggregate([
        {'$sample':{'size':batch_size}},
        {'$project':{feature_name:True, label_name:True}}
        ])
    
    
pass
# TODO fix Ward.D2 to ward.D2 and Ward.D to ward.D
# Project is a useful aggregator - it just passes the specified field through 
# to the next pipeline
# Match a list or array

a = collection.aggregate([
        {'$project':{'best_hyper.clusterer':'$best_hyper.clusterer',
                     '_id':'$_id'}},
         {'$match':{'best_hyper.clusterer':{'$in':['Ward.D']}}}
         ])

for document in a:
    clusterer = document['best_hyper']['clusterer']
    
    for idx, clust in enumerate(clusterer):
        if clust == 'Ward.D':
            clusterer[idx] = 'ward.D'
            
    assert 'Ward.D' not in clusterer, 'Try again'
    
    collection.update_one({'_id':document['_id']},
                      {'$set':{'best_hyper.clusterer':clusterer}})



@author: z003vrzks
"""

