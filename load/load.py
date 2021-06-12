# -*- coding: utf-8 -*-
"""
Created on Sun May 17 09:45:37 2020

@author: z003vrzk
"""

# Python imports
import os
import sys

# Third party imports
import pandas as pd
import numpy as np

# Local imports

"""Add modules in lateral packages to python path. This is necessary because
relative imports are not possible when running this module as a top level module
AKA with a name == __main__"""
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)


#%%

class Load():

    def __init__(self):
        pass

    @staticmethod
    def get_database_set(names, features):
        """Generates an iterator which yields dataframes that all have a common
<<<<<<< HEAD
        attribute. Each names and features must have the same length,
=======
        attribute in common. Each names and features must have the same length,
>>>>>>> cb4b1f01b7e16d3046036f24cec8de0bba93eed5
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

