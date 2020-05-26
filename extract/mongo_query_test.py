# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:08:51 2020

@author: z003vrzk
"""

# Third party imports
from pymongo import MongoClient

# Local imports
from mongo_query import MongoQuery


#%% testing

if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client['test']
    test_collection = db['test_collection']

    test_document = test_collection.find_one()

    _id = test_document['_id']
    array_index = 1
    label = 'test_label'

    result = test_collection.update_one({'_id':_id},
                            {'$set':{'clustered_points.'+str(array_index)+'.label':label}})

    result.acknowledged
    result.modified_count

    modified_document = next(test_collection.find({'_id':_id}))
    modified_document['clustered_points'][0]['points'] == modified_document['clustered_points'][1]['points']

#%% Test mongo methods

if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client['test']
    test_collection = db['test_collection']

    # Testing retrieve documents
    myQueryClass = MongoQuery(test_collection)
    my_generator = myQueryClass.retrieve_document_missing_labels()
    document = next(my_generator)

    # Testing retrieving dataframe
    dataframe_generator = myQueryClass.retrieve_points_dataframe(document)
    index, dataframe = next(dataframe_generator)

    # Testing splitting dataframe
    upper_split, lower_split = myQueryClass.split_dataframe_on_index(dataframe, [0,1,2,3,4,5])
    new_df = pd.DataFrame.from_dict(upper_split['points'])

    print(document['database_tag'])
    _id = document['_id']
    myQueryClass.append_points_to_array(_id, lower_split)
    myQueryClass.update_points_on_index(_id, 0, upper_split)

#%% Test query to actual data

if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client['master_points']
    clustered_points = db['clustered_points']

    document = clustered_points.find_one()
