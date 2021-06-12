# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:25:55 2020

@author: z003vrzk
"""

# Python imports
import sys
import os
import traceback
import pickle

# Third party imports
import numpy as np
import sqlalchemy
from sqlalchemy.sql import text as sqltext
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.sparse import vstack

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract import extract
from transform import transform_pipeline
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev,
                          Customers, ClusteringHyperparameter, Labeling)

# Globals



#%%


class LoadMIL:

    def __init__(self, server_name, driver_name, database_name):
        self.Insert = extract.Insert(server_name=server_name,
                                     driver_name=driver_name,
                                     database_name=database_name)
        return None


    def bag_data_generator(self, pipeline, verbose=False):
        """Return a bag of commonly labeled data
        Bags are defined in SQL Server in the Points table on the group_id
        Froeign Key"""

        # Retrieve all unique bag labels
        sql = """SELECT distinct group_id
        FROM {}
        WHERE IsNumeric(group_id) = 1
        ORDER BY group_id ASC""".format(Points.__tablename__)
        sel = sqltext(sql)
        group_ids = self.Insert.core_select_execute(sel)

        # Retrieve bag label for each group_id
        sql_bag = """SELECT id, bag_label
                FROM {}
                WHERE id = {}"""

        # Create the pipeline
        if pipeline == 'whole':
            full_pipeline = self.mil_transform_pipeline()
        elif pipeline == 'categorical':
            full_pipeline = self.naive_bayes_transform_pipeline()
        else:
            raise ValueError('pipeline must be one of ["whole","categorical"]')

        for row in group_ids:
            group_id = row.group_id

            sel = sqltext(sql_bag.format(Labeling.__tablename__, group_id))
            with self.Insert.engine.connect() as connection:
                res = connection.execute(sel)
                label = res.fetchone().bag_label

            # Load the dataset
            sel = sqlalchemy.select([Points]).where(Points.group_id.__eq__(group_id))
            dfraw = self.Insert.pandas_select_execute(sel)

            # Validate raw dataset
            if not self.validate_bag(dfraw):
                continue

            # Transform the dataset
            try:
                bag = full_pipeline.fit_transform(dfraw)
            except ValueError as e:
                print('Transform error, Skipped Group ID : ', group_id)

                if verbose:
                    traceback.print_exc()
                    print(dfraw)
                    x = input("Do you want to continue and discard this bag? : ")
                    if x in ['y','yes','Y','Yes','True','TRUE']:
                        continue
                    else:
                        raise e
                else:
                    continue

            # Validate cleaned dataset
            if not self.validate_bag(bag):
                continue

            yield bag, label


    def validate_bag(self, bag):
        """Determine if a bag of instances is valid. A bag is valid if the
        resulting bag has at least one instance
        inputs
        ------
        bag : (pd.DataFrame) or (scipy.sparse.csr.csr_matrix)
        outputs
        -------
        is_valid : (bool) True if the bag has one instance at least"""

        # Failure - a group has dupilcate point names are are both deleted
        # during cleaning, causing an empty array to pass to subsequent pipes
        if isinstance(bag, pd.DataFrame):
            all_L2SL = list(set(bag['TYPE'])) == ['L2SL']
            all_virtual = list(set(bag['VIRTUAL'])) == [True]

            if all_L2SL:
                print('Bag contained all L2SL instances and is skipped')
                return False
            if all_virtual:
                print('Bag contained all Virtual instances and is skipped')
                return False

        if bag.shape[0] > 0:
            return True

        return False


    def mil_transform_pipeline(self):
        """
        inputs
        ------
        None : This pipeline is made to have static features. There is no need
        for passing variables
        outputs
        ------
        full_pipeline : (sklearn.pipeline) of all features for MIL / bag learning
        """

        # Transform pipeline
        Transform = transform_pipeline.Transform()

        # Cleaning pipeline
        clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                                 nan_replace_dict=None,
                                                 dtype_dict=None,
                                                 unit_dict=None,
                                                 remove_dupe=True,
                                                 replace_numbers=True,
                                                 remove_virtual=True)

        # Text feature encoders
        name_file = r'../data/vocab_name.txt'
        name_vocabulary = transform_pipeline.VocabularyText.read_vocabulary_disc(name_file)
        name_text_pipe = Transform.text_pipeline_label(attributes=['NAME'],
                                                  vocabulary=name_vocabulary)
        descriptor_file = r'../data/vocab_descriptor.txt'
        descriptor_vocabulary = transform_pipeline.VocabularyText.read_vocabulary_disc(descriptor_file)
        descriptor_text_pipe = Transform.text_pipeline_label(attributes=['DESCRIPTOR'],
                                                             vocabulary=descriptor_vocabulary)


        # Categorical Features
        categorical_pipe = Transform.categorical_pipeline(categorical_attributes=None,
                              handle_unknown='ignore',
                              categories_file=r'../data/categorical_categories.dat')

        # Numeric features
        numeric_pipe = Transform.numeric_pipeline(numeric_attributes=None)

        # Union
        combined_features = FeatureUnion(transformer_list=[
            ('CategoricalPipe', categorical_pipe),
            ('NameTextPipe',name_text_pipe),
            ('DescriptorTextPipe',descriptor_text_pipe),
            ('NumericPipe',numeric_pipe),
            ])
        full_pipeline = Pipeline([
            ('CleaningPipe', clean_pipe),
            ('CombinedCategorical',combined_features),
            ])

        return full_pipeline


    def naive_bayes_transform_pipeline(self):
        """Pipeline that only includes categorical and text features for
        naive_bayes categorical estimators
        inputs
        ------
        None : This pipeline is made to have static features. There is no need
        for passing variables
        outputs
        ------
        full_pipeline : (sklearn.pipeline) of all features for MIL / bag learning
        """

        # Transform pipeline
        Transform = transform_pipeline.Transform()

        # Cleaning pipeline
        clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                                 nan_replace_dict=None,
                                                 dtype_dict=None,
                                                 unit_dict=None,
                                                 remove_dupe=True,
                                                 replace_numbers=True,
                                                 remove_virtual=True)

        # Text feature encoders
        name_file = r'../data/vocab_name.txt'
        name_vocabulary = transform_pipeline.VocabularyText.read_vocabulary_disc(name_file)
        name_text_pipe = Transform.text_pipeline_label(attributes=['NAME'],
                                                  vocabulary=name_vocabulary)
        descriptor_file = r'../data/vocab_descriptor.txt'
        descriptor_vocabulary = transform_pipeline.VocabularyText.read_vocabulary_disc(descriptor_file)
        descriptor_text_pipe = Transform.text_pipeline_label(attributes=['DESCRIPTOR'],
                                                             vocabulary=descriptor_vocabulary)


        # Categorical Features
        categorical_pipe = Transform.categorical_pipeline(categorical_attributes=None,
                              handle_unknown='ignore',
                              categories_file=r'../data/categorical_categories.dat')

        # Numeric features - EXCLUDE
        # numeric_pipe = Transform.numeric_pipeline(numeric_attributes=None)

        # Union
        combined_features = FeatureUnion(transformer_list=[
            ('CategoricalPipe', categorical_pipe),
            ('NameTextPipe',name_text_pipe),
            ('DescriptorTextPipe',descriptor_text_pipe),
            # ('NumericPipe',numeric_pipe), # EXCLUDE
            ])
        full_pipeline = Pipeline([
            ('CleaningPipe', clean_pipe),
            ('CombinedCategorical',combined_features),
            ])

        return full_pipeline


    def gather_mil_dataset(self, pipeline='whole'):
        """Return all bags as a numpy array
        inputs
        ------
        pipeline : (str) one of ['whole','categorical']. Use 'whole' for KNN
            and estimators that can handle numeric features. Use 'categorical'
            for estimators that only use categorical faetures (naive bayes)
        outputs
        ------
        dataset : (np.array) of bags with rank 3.
            Each bag is an array of instances
            Each instance is a CSR array of features

            Example
            dataset = gather_mil_dataset() # np.array rank 3
            bag0 = dataset[0] # np.array rank 2
            instance0_1 = bag0[1] # CSR Array
            """

        # Initialize list until I can later form array
        dataset = []
        bag_labels = []

        # Initialize generator
        bag_generator = self.bag_data_generator(pipeline)

        for bag, label in bag_generator:
            dataset.append(bag)
            bag_labels.append(label)

        x = np.array(dataset)
        y = np.array(bag_labels)

        return x, y


    @staticmethod
    def load_mil_dataset(file_name=r'../data/MIL_dataset.dat'):
        """
        inputs
        ------
        file_name : (str) name of data file to load pickled object from
        outputs
        -------
        dataset : (dict) with keys ['dataset','bag_labels']
        Object type depends on how the objet was saved - see save_mil_dataset"""

        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)

        return dataset


    @staticmethod
    def save_mil_dataset(bags, bag_labels, file_name=r'../data/MIL_dataset.dat'):
        """Picke and save an object
        input
        -------
        bags : (np.array) rank 3 of bags. Each bag is a rank 2 matrix of instances
        bag_labels : (np.array) string labels corresponding to each bag.
            Should have same shape as first dimension of bags"""

        msg='Bags and labels must have the same first dimension'
        assert bag_labels.shape[0] == bags.shape[0], msg

        MILData = {'dataset':bags,'bag_labels':bag_labels}

        with open(file_name, 'wb') as f:
            pickle.dump(MILData, f)

        return None



class SingleInstanceGather:


    def __init__(self):
        return None


    @staticmethod
    def bags_2_si_generator(bags, bag_labels, sparse=False):
        """Convert a n x (m x p) array of bag instances into a k x p array of
        instances. n is the number of bags, and m is the number of instances within
        each bag. m can vary per bag. k is the total number of instances within
        all bags. k = sum (m for bag in n). p is the feature space of each instance
        inputs
        -------
        bags : (iterable) containing bags of shape (m x p) sparse arrays
        bag_labels : (iterable) containing labels assocaited with each bag. Labels
            are expanded and each instance within a bag inherits the label of the
            bag
        sparse : (bool) if True, the output instances are left as a sparse array.
            Some sklearn estimators can handle sparse feature inputs
        output
        -------
        instances, labels : (generator) """

        for bag, label in zip(bags, bag_labels):
            # Unpack bag into instances

            if sparse:
                instances = bag # Sparse array
            else:
                instances = bag.toarray() # Dense array

            labels = np.array([label].__mul__(instances.shape[0]))

            yield instances, labels


    @classmethod
    def bags_2_si(cls, bags, bag_labels, sparse=False):
        """Convert a n x (m x p) array of bag instances into a k x p array of
        instances. n is the number of bags, and m is the number of instances within
        each bag. m can vary per bag. k is the total number of instances within
        all bags. k = sum (m for bag in n). p is the feature space of each instance
        inputs
        -------
        bags : (iterable) containing bags of shape (m x p) sparse arrays
        bag_labels : (iterable) containing labels assocaited with each bag. Labels
            are expanded and each instance within a bag inherits the label of the
            bag
        sparse : (bool) if True, the output instances are left as a sparse array.
            Some sklearn estimators can handle sparse feature inputs
        output
        -------
        instances, labels : (np.array) or (scipy.sparse.csr.csr_matrix)
        depending on 'sparse'"""

        # Initialize generator over bags
        bag_iterator = cls.bags_2_si_generator(bags,
                                               bag_labels,
                                               sparse=sparse)

        # Initialize datasets
        instances, labels = [], []

        # Gather datasets
        for part_instances, part_labels in bag_iterator:

            instances.append(part_instances)
            labels.append(part_labels)

        # Flatten into otuput shape - [k x p] instances and [k] labels
        if sparse:
            # Row-stack sparse arrays into a sinlge  k x p sparse array
            instances = vstack(instances)
            labels = np.concatenate(labels)
        else:
            # Row-concatenate dense arrays into a single k x p array
            instances = np.concatenate(instances)
            labels = np.concatenate(labels)

        return instances, labels