# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:57:25 2019

Use these mongodb queries to update and label documents.
My database is called master_points
My collection is called clustered_points
A test collection exists (for testing duh)

The main functionalities satisfied by this class :
1) Retrieve a document that is missing labels. Each document in clustered_points
has a nested object called clustered_points. Find objects in clustered_ponits
array that do not have a 'label' field. The whole document is returned
if one element of clustered_points does not have the field 'label'
{[...]
clustered_points [
        {{points:{POINTID:[ARRAY], NETDEVID:[ARRAY], etc..}},
         {label:string_label}
         },
        {{points:{POINTID:[ARRAY], NETDEVID:[ARRAY], etc..}},
         {label:string_label}
         }
]
2) Change a specific 'points' object in clustered_points. I can assign it a new
    points object using mongo's update_one and $set operator. The motivation is
    if I want to split a cluster up into smaller systems, I can do this by
    updating the current points object with a bisection of the current points
    object, and saving the split off object into a new points object
3) Append a 'points' object to the clustered_ponits array.  I can assign the new
    object any points object I want. This will be useful when splitting existing
    clusteres into finer systems. See above for motivation
4) append a 'label' field:value pair to an object in the clustered_points
    array

@author: z003vrzk
"""

# Third party imports
from pymongo import MongoClient
import sys
import os
import pandas as pd
import copy

# Local imports
_CWD = os.getcwd()
_PARTS = _CWD.split(os.sep)
_BASE_DIR = [os.path.join(part) for part in _PARTS[:-4]]
_PACKAGE_PATH = os.path.join('C:\\', *_BASE_DIR)
if _PACKAGE_PATH not in sys.path:
    sys.path.insert(0, _PACKAGE_PATH)

from clustering.unsupervised_cluster import UnsupervisedClusterPoints


#%%
"""Retrieve and clean database points from mongo
The goal is to view names in a GUI so I can easily label and divide clustered
points"""


class MongoQuery():

    def __init__(self, collection_object):
        """Use this class to interact with mongodb. It contains several useful
        methods :
        1) Retrieve a document that is missing labels. Each document in clustered_points
        has a nested object called clustered_points. Find objects in clustered_ponits
        array that do not have a 'label' field. The whole document is returned
        if one element of clustered_points does not have the field 'label'
        {[...]
        clustered_points [
                {{points:{POINTID:[ARRAY], NETDEVID:[ARRAY], etc..}},
                 {label:string_label}
                 },
                {{points:{POINTID:[ARRAY], NETDEVID:[ARRAY], etc..}},
                 {label:string_label}
                 }
        ]
        2) Change a specific 'points' object in clustered_points. I can assign it a new
            points object using mongo's update_one and $set operator. The motivation is
            if I want to split a cluster up into smaller systems, I can do this by
            updating the current points object with a bisection of the current points
            object, and saving the split off object into a new points object
        3) Append a 'points' object to the clustered_ponits array.  I can assign the new
            object any points object I want. This will be useful when splitting existing
            clusteres into finer systems. See above for motivation
        4) append a 'label' field:value pair to an object in the clustered_points
            array

        Example usage :
        client = MongoClient('localhost', 27017)
        db = client['master_points']
        clustered_points = db['clustered_points']

        mongoQuery = MongoQuery(clustered_points)
            """
        # Instantiate collection
        if not collection_object.name == 'clustered_points':
            user_input = input('Are you shure you want to use the {} collection\
                            instead of the recommended "clustered_points"\
                            collection? (Y/N)\n>>>'.format(collection_object.name))
            if user_input in ['True','true','Y','y']:
                confirm = True

            if confirm:
                # Create collection object based on user input
                self.collection = collection_object
            else:
                raise ValueError("User escaped creating\
                                 collection : {}".format(collection_object))

        elif collection_object.name == 'clustered_points':
            self.collection = collection_object

        else:
            raise ValueError("Invalid collection_object : {}".format(collection_object))


    def retrieve_document_missing_labels(self):
        # Find documents where not all clustered_points are labeled
        """Use this to generate mongodb documents where not all clustered
        points have a label. It will find documents where any of the key:value
        pairs under 'clustered_points' do not have the 'lable' key
        inputs
        -------
        None
        outputs
        --------
        document : (dict) document from mongodb

        Example Usage :
        myQueryClass = MongoQuery(test_collection)
        my_generator = myQueryClass.retrieve_document_missing_labels()
        document = next(my_generator)
        """
        collection = self.collection
        cursor = collection.find({
                'clustered_points': {
                        '$elemMatch': {'label': {'$exists':False}}}})

        for document in cursor:

            yield document


    def update_points_on_index(self, _id, array_index, new_points_object):
        """Use this to update one of the objects under 'clustered_points' with
        a new points dictionary. This is useful when you want to remove or
        add points to the clustered_points object
        inputs
        -------
        _id : (BSON) _id from mongodb identifying a unique document
        array_index : (int) index in the array that you want to modify
        new_points_object : (dict, or other) json like object that you want to
        use instead of the existing object in the array"""
        # Update a document points array
        collection = self.collection
        result = collection.update_one({'_id':_id},
                           {'$set':
                               {'clustered_points.'+str(array_index):new_points_object}})
        return result.modified_count


    def append_points_to_array(self, _id, new_item):
        """Add an object to the clustered_points array
        inputs
        -------
        _id : (BSON) mongodb document to modify. points are pushed onto the
        clustered_points array
        new_item : (dict) should be nested with a 'points' key and the nested
        pandas dataframe converted to dictionary"""
        collection = self.collection
        result = collection.update_one({'_id':_id},
                                    {'$push':{'clustered_points':new_item}})
        return result.modified_count


    def add_label_to_cluster(self, _id, array_index, label):
        # Add a label to an object in an array
        assert isinstance(label, list), 'Passed label must be type list'
        collection = self.collection
        result = collection.update({'_id':_id},
                                {'$set':{'clustered_points.'
                                         +str(array_index)+
                                         '.label':label}})

        return result.modified_count


    @staticmethod
    def retrieve_points_dataframe(document):
        """Given a document, yield all the points clusters from that saved
        document. Returns a generator of dataframes that can be iteratred over
        inputs
        -------
        document : (dict) document from mongodb containing the 'clustered_points'
        key
        outputs
        ------
        dataframe_generator : (generator) generator over a documents clustered
        points in dataframe format"""

        for index, cluster_dict in enumerate(document['clustered_points']):
            # The points saved in the 'clustered_points' collection have
            # Already been passed through a cleaning pipeline
            if 'label' in cluster_dict.keys():
                continue

            dataframe_clean = pd.DataFrame.from_dict(cluster_dict['points'],
                                                   orient='columns')

            yield index, dataframe_clean


    @staticmethod
    def split_dataframe_on_index(dataframe, split_indicies):
        """Splits a dataframe into (2) parts. One are all points that are
        above an absolute index, and the other are all points below an absolute
        index. This is used to divide a cluster of points into two independent
        clusters. The result is NOT a dataframe, but a nested dictionary
        that enforces the document structure in mongodb
        inputs
        -------
        dataframe : (pd.DataFrame) to split based on
        index : (int) absolute index to split dataframe on
        outputs
        -------
        top_split, bottom_split : (dict) that can be directly saved in mongodb
        It has the form {'points':split_dictionary} where split_dictionary
        has all the column keys of the dataframe like {'col1':list1,'col2':list2}"""

        df_top = dataframe.drop(axis='index',index=split_indicies)
        df_bottom = dataframe.iloc[split_indicies]

        top_dict = df_top.to_dict(orient='list')
        bottom_dict = df_bottom.to_dict(orient='list')

        top_split = {'points':top_dict}
        bottom_split = {'points':bottom_dict}

        return top_split, bottom_split




