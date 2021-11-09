# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 07:53:26 2019

This is a package of useful pipelines and transformation classes used for
processing data. This module contains:

 'DataFrameSelector' - Retrieve values of certain columns from pd.DataFrame
 'DuplicateRemover' - Remove duplicates from named columns of a pd.DataFrame
 'RemoveAttribute' - Removes named attributes/columns from pd.DataFrames
 'RemoveNan' - Removes nan values from named columns in pd.DataFrame. Contains
 useful custom ways to deal with nan (aka replace w/ text)
 'SetDtypes' - Changes column data types of pd.DataFrame
 'StandardScaler' - Scales values along columns
 'TextCleaner' - Processes text attributes/columns with regex
 'TextToWordDict' - Creates a one-hot encoded dataframe from text instances
 'UnitCleaner' - A custom mapping of words in my data to a standard set of
 units/words
 'VirtualRemover' - A custom class to remove instances with the "virtual"
 instance type
 'WordDictToSparseTransformer' - Converts your word dictionary to a sparse
 matrix of encoded words

Example usage:

@author: z003vrzk
"""

# Python imports
import re
import statistics
from statistics import StatisticsError
import pickle
import os, sys
import configparser

# Third party imports
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

# Globals
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']


#%%


class RemoveAttribute(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.drop(columns=self.columns)
        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributeNames):
        self.attributeNames = attributeNames
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attributeNames].values


class ArrayResize(BaseEstimator, TransformerMixin):

    def __init__(self, shape):
        self.shape = shape
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.reshape(self.shape)


class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, columns, replace_numbers=True):
        """columns: columns to clean text in. Must be list of column indicies
        replace_numbers: Boolean for wither to repalce numbers with empty string"""
        self.REPLACE_BY_EMPTY_RE= re.compile('[/(){}\[\]\|@\\\,;]')
        self.BAD_SYMBOLS_RE = re.compile('[^a-zA-Z0-9 _.]')
        self.NUMBERS_RE = re.compile('[0-9]')
        self.columns = columns
        self.replace_numbers = replace_numbers
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #Mapp a function across desired columns
        for col in self.columns:
            X[col] = X[col].astype(str)
            X[col] = X[col].apply(lambda x: self.clean_text(x))
        return X

    def clean_text(self, text):

        text = text.lower()
        text = self.REPLACE_BY_EMPTY_RE.sub('', text)
        text = self.BAD_SYMBOLS_RE.sub('', text)

        if self.replace_numbers:
            text = self.NUMBERS_RE.sub('', text)
        return text


class SetDtypes(BaseEstimator, TransformerMixin):

    def __init__(self, type_dict):
        """
        inputs
        ------
        type_dict: (dict) where keys are column names and values are data types
        Example
        type_dict = {'col1':int,'col2':str,'col3':np.int32}"""
        self.type_dict = type_dict
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        for col, my_type in self.type_dict.items():
            X[col] = X[col].astype(my_type)

        return X


class RemoveNan(BaseEstimator, TransformerMixin):

    def __init__(self, type_dict):
        self.type_dict = type_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #Mapp a function across desired columns
        for col, method in self.type_dict.items():

            if method == 'remove':
                X = X.dropna(axis=0, subset=[col])
            elif method == 'empty':
                X[col] = X[col].fillna(value='', axis=0)
            elif method == 'zero':
                X[col] = X[col].fillna(value=0, axis=0)
            elif method == 'mode':
                col_mode = X[col].mode()[0]
                X[col] = X[col].fillna(value=col_mode, axis=0)
            else:
                X[col] = X[col].fillna(value=method, axis=0)

        X.reset_index(drop=True, inplace=True)
        return X


class ReplaceNone(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        """Fill None values with a string 'None'
        inputs
        -------
        columns: (list) of string denoting column names in X"""
        self.columns = columns
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Note - pd.DataFrame.fillna() will replace python None objects as
        well as np.nan values"""
        for col in self.columns:
            X[col].fillna(value='None', axis=0, inplace=True)

        return X


class UnitCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, unit_dict):
        self.unit_dict = unit_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        unit_dict = self.unit_dict

        for old_unit, new_unit in unit_dict.items():
            # Match the whole string - case insensitive
            reg_str = r'(?i)^' + re.escape(old_unit) + '$'
            X['DEVUNITS'] = X['DEVUNITS'].replace(to_replace=reg_str,
                                                  value=new_unit,
                                                  regex=True)

        return X


class TextToWordDict(BaseEstimator, TransformerMixin):

    def __init__(self, seperator = ' ', heirarchial_weight_word_pattern=False):
        """parameters
        -------
        seperator: (str or list of str) seperator between each text instance.
            Used in re.split()
        heirarchial_weight_word_pattern: (bool) setting this to True will
            weight each word by the order that it appears in the input sequence.
            For example, the word phrase 'foo.bar.baz.fizz' will be given
            counts that relate inversely to their position in the sequence. The
            resulting word count will be a counter object of
            Counter({'foo': 4, 'bar': 2, 'baz': 1, 'fizz': 1}). If used with
            the text_pipeline or WordDictToSparseTransformer this will be
            encoded into the array [4,3,2,1].
        """
        self.seperator = seperator
        self.heirarchial_weight_word_pattern = heirarchial_weight_word_pattern

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Takes a text iterable and returns a dictionary of key:value
        where key is a word/phrase that appears in each instance of X, and value
        is the total number of times key appears in an instance"""
        regex_pattern = '|'.join(map(re.escape, self.seperator))
        X_ListDictionary = []
        word_lengths = []

        # Find mode of word lengths for heirarchial_weight_word_pattern
        for point_name in X:
            word_length = len(re.split(regex_pattern, point_name))
            word_lengths.append(word_length)

        try:
            word_length_mode = statistics.mode(word_lengths)
        except StatisticsError:
            c = Counter(word_lengths)
            word_length_mode = c.most_common(1)[0][0]

        # Create counter dictionary to save word counts
        for point_name in X:

            if self.heirarchial_weight_word_pattern:

                # Give different weights
                word_counts = Counter(re.split(regex_pattern, point_name))
                n_words = len(word_counts)
                weight_sequence = [_idx for _idx in range(word_length_mode, 0, -1)]

                if n_words > len(weight_sequence):
                    # If n_words is greater than the mode some values in
                    # word_counts will be set to 0 or error will happen
                    # For out of range
                    n_ones_to_append = [1] * (n_words-len(weight_sequence))
                    weight_sequence.extend(n_ones_to_append)

                for key, new_count in zip(word_counts.keys(),
                                       weight_sequence):
                    word_counts[key] = new_count

            else:
                word_counts = Counter(re.split(regex_pattern, point_name))

            X_ListDictionary.append(word_counts)

        return np.array(X_ListDictionary)


class WordDictToSparseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, vocabulary_size='all'):
        """parameters
        -------
        vocabulary_size: int or 'all'. int will return specified size. 'all'
        will return all unique words available"""
        self.vocabulary_size = vocabulary_size
        return None

    def fit(self, X, y=None):

        self.master_dict = {}

        for counterObject in X:

            for word, count in counterObject.items():

                if self.master_dict.__contains__(word):
                    self.master_dict[word] += count
                else:
                    self.master_dict[word] = count

        #Get top list of vocabs..
        self.sortedWords = sorted(self.master_dict, key=self.master_dict.get, reverse=True)
        if type(self.vocabulary_size) == str:
            self.vocabulary = self.sortedWords[:]
            self.vocabulary_size = len(self.vocabulary)
        else:
            self.vocabulary = self.sortedWords[:self.vocabulary_size]
        self.vocabulary.insert(0, 'Other')

        return self

    def transform(self, X, y=None):
        # iterate through each counter object in the [Counter({})] passed to this transformer
        # For each item in the list, output a sparse matrix based on # of words that match the master dict
        row_ind = []
        col_ind = []
        data = []
        for row, counterObject in enumerate(X):

            for word, count in counterObject.items():

                row_ind.append(row)
                try:
                    col_ind.append(self.vocabulary.index(word))
                except ValueError:
                    col_ind.append(0)
                data.append(count)

        return csr_matrix((data, (row_ind, col_ind)), shape=(len(X), self.vocabulary_size+1))


class DuplicateRemover(BaseEstimator, TransformerMixin):

    def __init__(self, dupe_cols, remove_dupe=True):
        """
        inputs
        -------
        dupe_cols: (list) of string where each string is a column name in
            the dataframe. Where the column has duplicated values in its index
            those rows / index will be removed
        remove_dup: (bool) to remove duplicates from the columns in dupe_cols

        outputs
        -------
        X: (pd.DataFrame) with duplicate"""
        assert type(dupe_cols) == list, 'dupe_cols must be list'
        self.dupe_cols = dupe_cols
        self.remove_dupe = remove_dupe
        return None

    def fit(self, X, y=None):
        # Nothing

        return self

    def transform(self, X, y=None): #Delete repeats

        if self.remove_dupe:

            for col in self.dupe_cols:
                duplicate_indicies = self.get_duplicate_indicies(X, col)
                if len(duplicate_indicies) == 0:
                    # There are no duplicates found
                    continue
                X.drop(index=duplicate_indicies, axis=0, inplace=True)
                # Reset index so we can remove rows from multiple iterations
                # Of columns
                X.reset_index(drop=True, inplace=True)

        return X

    def get_duplicate_indicies(self, X, column):
        """Given a dataframe and column name find the duplicate values
        in the column
        inputs
        -------
        column: (str) name of column
        X: (pd.DataFrame) dataframe to find duplicates in
        output
        -------
        duplicates: (list) of duplicate values in a column"""

        duplicates = []
        counts = {}

        for index, value in X[column].iteritems():
            try:
                counts[value] += 1
            except:
                counts[value] = 1

        for value, count in counts.items():
            if count >= 2:
                indicies = np.where(X[column] == value)[0]
                duplicates.append(indicies)

        # Flatten the duplicate indicies
        if len(duplicates) >= 1:
            duplicate_indicies = np.hstack(duplicates)
        else:
            duplicate_indicies = duplicates

        return duplicate_indicies


class VirtualRemover(BaseEstimator, TransformerMixin):

    def __init__(self, remove_virtual=False):
        assert type(remove_virtual) ==  bool, 'remove_virtual must be boolean'
        self.remove_virtual = remove_virtual

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None): #Delete repeats

        if self.remove_virtual:
            # Check for both bool and str data types depeneding
            # On how data enters the stream
            bool_indicies = np.where(X['VIRTUAL'] == True)[0]
            str_indicies = np.where(X['VIRTUAL'] == 'True')[0]
            indicies = np.concatenate((bool_indicies, str_indicies), axis=0)

            X.drop(index=indicies, axis=0, inplace=True)
            X.reset_index(drop=True, inplace=True)

        return X


#%%

DROP_ATTRIBUTES = ['CTSYSNAME', 'TMEMBER', 'ALARMHIGH', 'ALARMLOW',
        'COMBOID', 'PROOFPRSNT', 'PROOFDELAY', 'NORMCLOSE', 'INVERTED',
        'LAN', 'DROP', 'POINT', 'ADDRESSEXT', 'DEVNUMBER', 'CTSENSTYPE',
        'CONTRLTYPE', 'UNITSTYPE', 'SIGUNITS', 'NUMBERWIRE', 'POWER',
        'WIRESIZE', 'WIRELENGTH', 'S1000TYPE','INITVALUE']

UNUSED = ['SLOPE', 'INTERCEPT']

TYPE_DICT = {'NAME':str, 'NETDEVID':str,
             'DESCRIPTOR':str, 'ALARMTYPE':str,
             'FUNCTION':str, 'VIRTUAL':str,
             'SYSTEM':str, 'CS':str,
             'SENSORTYPE':str, 'DEVUNITS':str}

NAN_REPLACE_DICT = {'NETDEVID':'empty', 'NAME':'remove',
                          'DESCRIPTOR':'empty','TYPE':'mode',
                          'ALARMTYPE':'mode',
                          'FUNCTION':'Value','VIRTUAL':'False',
                          'SYSTEM':'empty','CS':'empty',
                          'SENSORTYPE':'digital','DEVICEHI':'zero',
                          'DEVICELO':'zero','DEVUNITS':'empty',
                          'SIGNALHI':'zero','SIGNALLO':'zero',
                          'SLOPE':'zero','INTERCEPT':'zero'}

TEXT_CLEAN_ATTRS = ['NAME','DESCRIPTOR','SYSTEM']



# Modify units
TYPES_FILE = r'../data/clean_types.csv'
units_df = pd.read_csv(TYPES_FILE)
UNIT_DICT = {}
for idx, unit in (units_df.iterrows()):
    depreciated_value = unit['depreciated_type']
    new_value = unit['new_type']
    if new_value == '0':
        new_value = ''
    UNIT_DICT[depreciated_value] = new_value

# Remove duplicates from the NAME column
DUPE_COLS = ['NAME']

# Categorical attributes in dataset that should be one-hot encoded
CATEGORICAL_ATTRIBUTES = ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS',
                   'SENSORTYPE', 'DEVUNITS']
NUM_ATTRIBUTES = ['DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO',
                       'SLOPE', 'INTERCEPT']

TEXT_ATTRIBUTES = ['NAME', 'DESCRIPTOR', 'SYSTEM']

REMOVE_DUPE=True
REPLACE_NUMBERS=True
REMOVE_VIRTUAL=True

CATEGORIES_FILE = r'../data/categorical_categories.dat'
    
class Transform():

    def __init__(self):
        """Transformation pipeline
        inputs
        -------
        None
        """
        return None

    @classmethod
    def numeric_pipeline(cls):
        """Return a numeric pipeline
        The numeric pipeline will scale numeric attributes in your dataset
        using sklearn.preprocessing.StandardScaler
        inputs
        ------
        numeric_attributes: (list) of str. Each string is a column name of
            a pandas dataframe. If None, then the classes standard numeric
            attributes will be used instead (see NUM_ATTRIBUTES)
        outputs
        -------
        num_pipeline: (sklearn.pipeline.Pipeline) to transform data
            Call cat_pipeline.fit_transform(dataframe) to transform your data
        """

        num_pipeline = Pipeline([
                ('selector', DataFrameSelector(NUM_ATTRIBUTES)),
                ('std_scaler',StandardScaler())
                ])

        return num_pipeline

    @classmethod
    def categorical_pipeline(cls,
                             handle_unknown='ignore'):
        """Return a categorical pipeline
        The categorical pipeline will one-hot encode categorical attributes
        in your dataset
        inputs
        ------
        categorical_attributes: (list) of str. Each string is a column name of
            a pandas dataframe. If None, then the classes standard categorical
            attributes will be used instead (see CATEGORIAL_ATTRIBUTES)
        handle_unknown = (str) how to handle categorical values in sklearn
            one of ['error' | 'ignore']
        outputs
        -------
        cat_pipeline: (sklearn.pipeline.Pipeline) to transform data
            Call cat_pipeline.fit_transform(dataframe) to transform your data
        """

        categories = cls._read_categories(CATEGORICAL_ATTRIBUTES,
                                           CATEGORIES_FILE)

        cat_pipeline = Pipeline([
                ('ReplaceNone', ReplaceNone(CATEGORICAL_ATTRIBUTES)),
                ('DataFrameSelector', DataFrameSelector(CATEGORICAL_ATTRIBUTES)),
                ('OneHotEncoder', OneHotEncoder(categories=categories,
                                                handle_unknown=handle_unknown)),
                ])

        return cat_pipeline

    @classmethod
    def _read_categories(cls, categorical_attributes, categories_file):
        """Read saved categories from disc - see EncodingCategories"""

        categories_dict = EncodingCategories.read_categories_from_disc(categories_file)
        categories_array = []

        for col_name in categorical_attributes:
            categories = categories_dict[col_name]
            categories_array.append(categories)

        return np.array(categories_array, dtype='object')

    @classmethod
    def cleaning_pipeline(cls):
        """Cleaning pipeline
        Remove attributes (self.drop_attributes)
        Remove nan (self.nan_replace_dict)
        Change data types (self._type_dict)
        Clean Text (TEXT_CLEAN_ATTRS, REPLACE_NUMBERS)
        Unit Cleaner (self.unit_dict)
        Remove Duplicates (DUPE_COLS, REMOVE_DUPE)
        Virtual Remove (remove_virtual)
        NOTE: To use this pipeline you should pass a pandas dataframe to its
        fit_transform() method
        inputs
        -------
        ouput
        -------
        A sklearn pipeline object containing modifier classes.
        To view the modifiers
        see Pipeline.named_steps attribute or Pipeline.__getitem__(ind)

        Example usage
        X = pd.DataFrame(data, columns, index)
        cleaning_pipeline = cleaning_pipeline(REMOVE_DUPE=True,
                                              REPLACE_NUMBERS=True,
                                              remove_virtual=True)
        dataframe = cleaning_pipeline.fit_transform(X)"""

        cleaning_pipeline = Pipeline([
                ('RemoveAttribute', RemoveAttribute(DROP_ATTRIBUTES)),
                ('RemoveNan', RemoveNan(NAN_REPLACE_DICT)),
                ('SetDtypes', SetDtypes(TYPE_DICT)),
                ('TextCleaner', TextCleaner(TEXT_CLEAN_ATTRS, replace_numbers=REPLACE_NUMBERS)),
                ('UnitCleaner', UnitCleaner(UNIT_DICT)),
                ('DuplicateRemover', DuplicateRemover(DUPE_COLS, remove_dupe=REMOVE_DUPE)),
                ('VirtualRemover', VirtualRemover(remove_virtual=REMOVE_VIRTUAL))
                ])

        return cleaning_pipeline


class EncodingCategories:

    def __init__(self):

        return None

    @staticmethod
    def read_categories_from_disc(file_path):
        """Read a set of saved categories from disc"""

        with open(file_path, 'rb') as f:
            categories_dict = pickle.load(f)

        return categories_dict

    @staticmethod
    def save_categories_to_disc(categories_dict, save_path):
        """Pickle a dictionary representing all possible categories related
        to each column of a dataframe
        inputs
        -------
        categories_dict: (dict) with key representing column name and
        value is an array representing unique valeus under column_name
        Example
        categories_dict = {'TYPE': <class 'numpy.ndarray'>
                           'ALARMTYPE': <class 'numpy.ndarray'>
                           'FUNCTION'': <class 'numpy.ndarray'>
                           'VIRTUAL'': <class 'numpy.ndarray'>
                           'CS'': <class 'numpy.ndarray'>
                           'SENSORTYPE'': <class 'numpy.ndarray'>
                           'DEVUNITS': <class 'numpy.ndarray'>}"""

        with open(save_path, 'wb') as f:
            pickle.dump(categories_dict, f)

        return None

    @staticmethod
    def calc_categories_dict(dataframe, columns):
        """Get the categories of value from the columns of a dataframe
        inputs
        -------
        dataframe: (pd.DataFrame) data to extract categories from
        columns: (list) of strings representing column names of the passed
            dataframe. Example ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL',
                                'CS', 'SENSORTYPE', 'DEVUNITS'] """
        categories_dict = {}

        for col in columns:
            # Get a unique set of values from the dataframe slice
            categories = set(dataframe[col].values)
            cat_array = np.array(list(categories))
            categories_dict[col] = cat_array

        # Handle 'DEVUNITS' separately
        categories_dict['DEVUNITS'] = np.array(list(Transform().unit_dict.values()))

        return categories_dict


class VocabularyText:

    def __init__(self):
        return None

    @staticmethod
    def read_vocabulary_disc(file_name):
        """Read vocabulary from a file and aggregate to a list
        inputs
        ------
        none
        outputs
        -------
        vocabulary: (list) of str"""

        with open(file_name, 'rt') as f:
            vocabulary = f.read().splitlines()

        return vocabulary