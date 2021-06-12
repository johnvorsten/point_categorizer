# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 07:53:26 2019

This is a package of useful pipelines and transformation classes used for
processing data. This module contains :

 'DataFrameSelector' - Retrieve values of certain columns from pd.DataFrame
 'DuplicateRemover' - Remove duplicates from named columns of a pd.DataFrame
 'JVDBPipe' - A class with some useful pre-built pipelines
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

Example usage :

@author: z003vrzk
"""

# Python imports
from pathlib import Path
import re
from collections import Counter,namedtuple
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
import sqlalchemy
from sqlalchemy.sql import text as sqltext

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract.SQLAlchemyDataDefinition import TypesCorrection
from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev,
                          Customers, ClusteringHyperparameter, Labeling)


# Globals
config = configparser.ConfigParser()
config.read(r'../extract/sql_config.ini')
server_name = config['sql_server']['DEFAULT_SQL_SERVER_NAME']
driver_name = config['sql_server']['DEFAULT_SQL_DRIVER_NAME']
database_name = config['sql_server']['DEFAULT_DATABASE_NAME']
Insert = extract.Insert(server_name, driver_name, database_name)

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
        """columns : columns to clean text in. Must be list of column indicies
        replace_numbers : Boolean for wither to repalce numbers with empty string"""
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
        type_dict : (dict) where keys are column names and values are data types
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
        columns : (list) of string denoting column names in X"""
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
        seperator : (str or list of str) seperator between each text instance.
            Used in re.split()
        heirarchial_weight_word_pattern : (bool) setting this to True will
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
        vocabulary_size : int or 'all'. int will return specified size. 'all'
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
        dupe_cols : (list) of string where each string is a column name in
            the dataframe. Where the column has duplicated values in its index
            those rows / index will be removed
        remove_dup : (bool) to remove duplicates from the columns in dupe_cols

        outputs
        -------
        X : (pd.DataFrame) with duplicate"""
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
        column : (str) name of column
        X : (pd.DataFrame) dataframe to find duplicates in
        output
        -------
        duplicates : (list) of duplicate values in a column"""

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

class DepreciatedError(Exception):
    pass

class JVDBPipe():

    def __init__(self):
        """Depreciated
        """

        raise DepreciatedError('This class is depreciated. Use Transform instead')

        self._drop_attributes = ['CTSYSNAME', 'TMEMBER', 'ALARMHIGH', 'ALARMLOW',
                           'COMBOID', 'PROOFPRSNT', 'PROOFDELAY', 'NORMCLOSE', 'INVERTED',
                            'LAN', 'DROP', 'POINT', 'ADDRESSEXT', 'DEVNUMBER', 'CTSENSTYPE',
                             'CONTRLTYPE', 'UNITSTYPE', 'SIGUNITS', 'NUMBERWIRE', 'POWER',
                             'WIRESIZE', 'WIRELENGTH', 'S1000TYPE']

        self._maybe_drop_attr = ['SLOPE', 'INTERCEPT']

        self._text_attrs = ['NAME','DESCRIPTOR','TYPE','FUNCTION','SYSTEM',
                            'SENSORTYPE','DEVUNITS']

        self._type_dict = {'NAME':str, 'NETDEVID':str, 'DESCRIPTOR':str,
                           'INITVALUE':float,'ALARMTYPE':str,'FUNCTION':str,
                           'VIRTUAL':bool,'SYSTEM':str,'CS':str,
                           'SENSORTYPE':str, 'DEVUNITS':str}

        """How to deal with nan's -
        remove : remove the whole row
        empty : replace with empty string
        zero : replace with 0
        mode : replace with mode
        any other value : replace with that value"""
        self._nan_replace_dict = {'NETDEVID':'empty', 'NAME':'remove', 'DESCRIPTOR':'empty','TYPE':'mode','INITVALUE':'zero',
                             'ALARMTYPE':'mode','FUNCTION':'Value','VIRTUAL':'False','SYSTEM':'empty',
                             'CS':'empty','SENSORTYPE':'digital','DEVICEHI':'zero','DEVICELO':'zero',
                             'DEVUNITS':'empty','SIGNALHI':'zero','SIGNALLO':'zero',
                             'SLOPE':'zero','INTERCEPT':'zero'}
        self._text_clean_attrs = ['NAME','DESCRIPTOR','SYSTEM']
        _unit_df = pd.read_csv(r"D:\Z - Saved SQL Databases\clean_types_manual.csv", index_col=0)
        self._unit_dict = {}
        for _, _units in (_unit_df.iterrows()):
            _old_unit = _units['old']
            _new_unit = _units['new']
            if _new_unit == '0':
                _new_unit = ''
            self._unit_dict[_old_unit] = _new_unit
        self._dupe_cols = ['NAME']

        self._cat_attributes = ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS',
                           'SENSORTYPE', 'DEVUNITS']
        self._num_attributes = ['INITVALUE', 'DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO',
                           'SLOPE', 'INTERCEPT']
        self._text_attributes = ['NAME', 'DESCRIPTOR', 'SYSTEM']



    def import_raw_data(self,
                        path=r'D:\Z - Saved SQL Databases\master_pts_db.csv'):
        """Imports data from CSV File (Dirty Data)
        parameters
        -------
        path : path to raw data .csv"""
        _master_path = Path(path)
        df = pd.read_csv(_master_path, header=0, index_col=0, encoding='mac_roman')
        df.reset_index(drop=True, inplace=True)
        self.master_database = df
        return None

    def calc_pipelines(self,
                       remove_dupe=False,
                       replace_numbers=True,
                       remove_virtual=False,
                       seperator='.',
                       vocabulary_size='all'):
        """Runs all pipelines and saves outputs to .csv files
        parameters
        -------
        remove_dupe : whether or not to remove duplicates from the 'NAME' column
        replace_numbers : replace numbers in text features with empty strings (True)
        remove_virtual : remove "virtual" points
        seperator : seperator for text features in the'Name' column. Used in
        TextToWordDict transformer. Can be single character or iterable of characters
        vocabulary_size : number of featuers to encode in name, descriptor,
        and sysetm features text
        """

        cleaning_pipeline = self.cleaning_pipeline(remove_dupe=remove_dupe,
                                                   replace_numbers=replace_numbers,
                                                   remove_virtual=remove_virtual)

        cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(self._cat_attributes)),
                ('catEncoder', OneHotEncoder())
                ])

        num_pipeline = Pipeline([
                ('selector', DataFrameSelector(self._num_attributes)),
                ('std_scaler',StandardScaler())
                ])

        full_pipeline = FeatureUnion(transformer_list=[
                ('cleaning_pipe', cleaning_pipeline),
                ('numPipeline', num_pipeline),
                ('catPipeline', cat_pipeline)
                ])

        df2_cat_num_scr = full_pipeline.fit_transform(self.master_database)
        _cat_cols = cat_pipeline.named_steps.catEncoder.categories_
        _cat_columns = np.concatenate(_cat_cols).ravel().tolist()
        _df2_cat_num_cols = [self._num_attributes, _cat_columns]
        df2_cat_num = pd.DataFrame(df2_cat_num_scr.toarray(),
                                   columns=[item for sublist in _df2_cat_num_cols for item in sublist])
        _save_path_cat_num = r'D:\Z - Saved SQL Databases\master_pts_db_categorical_numeric.csv'
        if not Path(_save_path_cat_num).is_file():
            df2_cat_num.to_csv(_save_path_cat_num)
        return

    def text_pipeline(self,
                      vocab_size,
                      attributes,
                      heirarchial_weight_word_pattern,
                      seperator='.'):
        """Run raw data through the data and text pipelines. The point of this
        function is to control removing duplicates and removing numbers

        NOTE : To use this pipeline you should pass a pandas dataframe to its
        fit_transform() method
        NOTE : This pipeline returns a sparse metrix. To get values use
        result.toarray(). To create a dataframe use pd.DataFrame(text_prepared.toarray(),
        columns=word_vocab)
        word_vocab = name_pipeline.named_steps['WordDictToSparseTransformer'].vocabulary

        parameters
        -------
        vocab_size : (int or str) vocabulary size for one-hot text attributes.
        Pass int or string 'all' to include all vocabulary
        attributes : (string or list) column names to select. Should be list to
        return a 2D array, or a string to return a 1D array. Values should
        be one or multiple of ['NAME', 'SYSTEM','DESCRIPTOR']
        seperator : (string or list) seperator for text features in the'attribute'.
        Typical values are '.' for NAME, or ' ' for SYSTEM or DESCRIPTOR
        feature. Used in TextToWordDict transformer. Can be single character
        string or iterable of character strings
        heirarchial_weight_word_pattern : (bool) setting this to True will
        weight each word by the order that it appears in the input sequence.
        For example, the word phrase 'foo.bar.baz.fizz' will be given
        counts that relate inversely to their position in the sequence. The
        resulting word count will be a counter object of
        Counter({'foo': 4, 'bar': 2, 'baz': 1, 'fizz': 1}). If used with
        the text_pipeline or WordDictToSparseTransformer this will be
        encoded into the array [4,3,2,1].

        ouput
        -------
        A sklearn pipeline object containing modifier classes. To view the modifiers
        see Pipeline.named_steps attribute or Pipeline.__getitem__(ind)

        Example Usage

        database = pd.DataFrame

        # Create 'clean' data processing pipeline
        clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False,
                                              replace_numbers=False,
                                              remove_virtual=True)

        # 'clean' transformed pandas dataframe
        df_clean = clean_pipe.fit_transform(database)

        # Create pipeline specifically for clustering text features
        text_pipe = myDBPipe.text_pipeline(vocab_size='all',
                                           attributes='NAME',
                                           seperator='.')
        X = text_pipe.fit_transform(df_clean).toarray()
        _word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
        df_text = pd.DataFrame(X, columns=_word_vocab)
        """

        name_pipeline = Pipeline([
                ('dataframe_selector', DataFrameSelector(attributes)),
                ('text_to_dict', TextToWordDict(seperator=seperator,
                    heirarchial_weight_word_pattern=heirarchial_weight_word_pattern)),
                ('WordDictToSparseTransformer', WordDictToSparseTransformer(
                        vocabulary_size=vocab_size))
        ])

        return name_pipeline

    def text_pipeline_calc(self,
                           X,
                           vocab_size='all',
                           seperator='.'):
        """Run raw data through the data and text pipelines. The point of this
        function is to control removing duplicates and removing numbers
        parameters
        -------
        X : dataframe to clean and process text. Default only processes 'NAME'
        column
        vocab_size : vocabulary size for one-hot text attributes. Pass int() or
        string 'all'
        seperator : seperator for text features in the'Name' column. Used in
        TextToWordDict transformer. Can be single character or iterable of characters
        """
        assert type(X) == pd.DataFrame, 'X msut be dataframe'
        assert X.shape[0] < 20000, 'Input dataframe shape is large. May be an issue'

        name_pipeline = Pipeline([('text_to_dict', TextToWordDict(seperator=seperator)),
                          ('WordDictToSparseTransformer', WordDictToSparseTransformer(vocabulary_size=vocab_size))
        ])


        text_prepared = name_pipeline.fit_transform(X['NAME'])
#        word_dict = name_pipeline.named_steps['WordDictToSparseTransformer'].master_dict
        word_vocab = name_pipeline.named_steps['WordDictToSparseTransformer'].vocabulary
#        df2_name_cols = [word_vocab]
        name_onehot = pd.DataFrame(text_prepared.toarray(),
                                   columns=word_vocab)

        return name_onehot

    def cleaning_pipeline(self,
                          remove_dupe,
                          replace_numbers,
                          remove_virtual):
        """Cleaning pipeline.
        Remove attributes (self._drop_attributes)
        Remove nan (self._nan_replace_dict)
        Change data types (self._type_dict)
        Clean Text (self._text_clean_attrs, replace_numbers)
        Unit Cleaner (self._unit_dict)
        Remove Duplicates (self._dupe_cols, remove_dupe)
        Virtual Remove (remove_virtual)
        NOTE : To use this pipeline you should pass a pandas dataframe to its
        fit_transform() method
        parameters
        -------
        remove_dupe : whether or not to remove duplicates from the 'NAME' column.
        This happens with L2SL values.
        replace_numbers : whether or not to replace numbers in TextCleaner,
        see self._text_clean_attrs
        remove_virtual : whether or not to remove virtuals in VirtualRemover
        ouput
        -------
        A sklearn pipeline object containing modifier classes. To view the modifiers
        see Pipeline.named_steps attribute or Pipeline.__getitem__(ind) """

        cleaning_pipeline = Pipeline([
                ('dropper', RemoveAttribute(self._drop_attributes)),
                ('nan_remover', RemoveNan(self._nan_replace_dict)),
                ('set_dtypes', SetDtypes(self._type_dict)),
                ('text_clean', TextCleaner(self._text_clean_attrs, replace_numbers=replace_numbers)),
                ('unit_clean', UnitCleaner(self._unit_dict)),
                ('dupe_remove', DuplicateRemover(self._dupe_cols, remove_dupe=remove_dupe)),
                ('virtual_remover', VirtualRemover(remove_virtual=remove_virtual))
                ])

        return cleaning_pipeline

    def cleaning_pipeline_calc(self,
                               X,
                          remove_dupe,
                          replace_numbers,
                          remove_virtual):
        """Cleaning pipeline.
        Remove attributes (self._drop_attributes)
        Remove nan (self._nan_replace_dict)
        Change data types (self._type_dict)
        Clean Text (self._text_clean_attrs, replace_numbers)
        Unit Cleaner (self._unit_dict)
        Remove Duplicates (self._dupe_cols, remove_dupe)
        Virtual Remove (remove_virtual
        )
        parameters
        -------
        remove_dupe : whether or not to remove duplicates from the 'NAME' column
        replace_numbers : whether or not to replace numbers in TextCleaner,
        see self._text_clean_attrs
        remove_virtual : whether or not to remove virtuals in VirtualRemover

        return
        -------
        data passed through cleaning_pipeline"""
        assert type(X) == pd.DataFrame, 'X Must be DataFrame'

        cleaning_pipeline = Pipeline([
                ('dropper', RemoveAttribute(self._drop_attributes)),
                ('nan_remover', RemoveNan(self._nan_replace_dict)),
                ('set_dtypes', SetDtypes(self._type_dict)),
                ('text_clean', TextCleaner(self._text_clean_attrs, replace_numbers=replace_numbers)),
                ('unit_clean', UnitCleaner(self._unit_dict)),
                ('dupe_remove', DuplicateRemover(self._dupe_cols, remove_dupe=remove_dupe)),
                ('virtual_remover', VirtualRemover(remove_virtual=remove_virtual))
                ])

        df2 = cleaning_pipeline.fit_transform(X)

        return df2


#%%

class Transform():

    def __init__(self):
        """Transformation pipeline

        inputs
        -------
        None
        """

        self.drop_attributes = ['CTSYSNAME', 'TMEMBER', 'ALARMHIGH', 'ALARMLOW',
                'COMBOID', 'PROOFPRSNT', 'PROOFDELAY', 'NORMCLOSE', 'INVERTED',
                'LAN', 'DROP', 'POINT', 'ADDRESSEXT', 'DEVNUMBER', 'CTSENSTYPE',
                'CONTRLTYPE', 'UNITSTYPE', 'SIGUNITS', 'NUMBERWIRE', 'POWER',
                'WIRESIZE', 'WIRELENGTH', 'S1000TYPE','INITVALUE']

        self._maybe_drop_attr = ['SLOPE', 'INTERCEPT']

        self._text_attrs = ['NAME','DESCRIPTOR','TYPE','FUNCTION',
                            'SYSTEM','SENSORTYPE','DEVUNITS']

        self.type_dict = {'NAME':str, 'NETDEVID':str,
                          'DESCRIPTOR':str, 'ALARMTYPE':str,
                          'FUNCTION':str, 'VIRTUAL':str,
                          'SYSTEM':str, 'CS':str,
                          'SENSORTYPE':str, 'DEVUNITS':str}

        self.nan_replace_dict = {'NETDEVID':'empty', 'NAME':'remove',
                                  'DESCRIPTOR':'empty','TYPE':'mode',
                                  'ALARMTYPE':'mode',
                                  'FUNCTION':'Value','VIRTUAL':'False',
                                  'SYSTEM':'empty','CS':'empty',
                                  'SENSORTYPE':'digital','DEVICEHI':'zero',
                                  'DEVICELO':'zero','DEVUNITS':'empty',
                                  'SIGNALHI':'zero','SIGNALLO':'zero',
                                  'SLOPE':'zero','INTERCEPT':'zero'}

        self._text_clean_attrs = ['NAME','DESCRIPTOR','SYSTEM']

        # Modify units
        sel = sqlalchemy.select([TypesCorrection])
        units_df = Insert.pandas_select_execute(sel)
        self.unit_dict = {}
        for idx, unit in (units_df.iterrows()):
            depreciated_value = unit['depreciated_type']
            new_value = unit['new_type']
            if new_value == '0':
                new_value = ''
            self.unit_dict[depreciated_value] = new_value

        # Remove duplicates from the NAME column
        self.dupe_cols = ['NAME']

        # Categorical attributes in dataset that should be one-hot encoded
        self.cat_attributes = ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS',
                           'SENSORTYPE', 'DEVUNITS']
        self.num_attributes = ['DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO',
                               'SLOPE', 'INTERCEPT']
        self._text_attributes = ['NAME', 'DESCRIPTOR', 'SYSTEM']

        return None


    def full_pipeline_label(self,
                       remove_dupe=False,
                       replace_numbers=True,
                       remove_virtual=False,
                       categorical_attributes=None,
                       numeric_attributes=None,
                       seperator='.',
                       vocabulary_size='all'):
        """Return a full pipeline feature union
        inputs
        -------
        remove_dupe : (bool) whether or not to remove duplicates from the
            'NAME' column
        replace_numbers : (bool) replace numbers in text features with
            empty strings (True)
        remove_virtual : (bool) whether or not to remove virtuals in VirtualRemover.
            Virtual points are variables in PPCL programming or used for
            variables in the BAS database. They contrast logical points which
            relate to a physical field sensor, device, or reading
        categorical_attributes : (list) of str. Each string is a column name of
            a pandas dataframe. If None, then the classes standard categorical
            attributes will be used instead (see self.cat_attributes)
        numeric_attributes : (list) of str. Each string is a column name of
            a pandas dataframe. If None, then the classes standard numeric
            attributes will be used instead (see self.num_attributes)
        seperator : (str) seperator for text features in the'Name' column.
            Used in TextToWordDict transformer.
            Can be single character or iterable of characters
        vocabulary_size : (int) number of featuers to encode in name,
        descriptor, and sysetm features text

        Example usage
        #TODO

        """
        if categorical_attributes is None:
            categorical_attributes = self.cat_attributes

        if numeric_attributes is None:
            numeric_attributes = self.num_attributes

        # Cleaning pipeline
        clean_pipe = self.cleaning_pipeline(drop_attributes=None,
                                                 nan_replace_dict=None,
                                                 dtype_dict=None,
                                                 unit_dict=None,
                                                 remove_dupe=True,
                                                 replace_numbers=True,
                                                 remove_virtual=True)

        # Text feature encoders
        name_file = r'../data/vocab_name.txt'
        name_vocabulary = VocabularyText.read_vocabulary_disc(name_file)
        name_text_pipe = self.text_pipeline_label(attributes=['NAME'],
                                                  vocabulary=name_vocabulary)
        descriptor_file = r'../data/vocab_descriptor.txt'
        descriptor_vocabulary = VocabularyText.read_vocabulary_disc(descriptor_file)
        descriptor_text_pipe = self.text_pipeline_label(attributes=['DESCRIPTOR'],
                                                             vocabulary=descriptor_vocabulary)


        # Categorical Features
        categorical_pipe = self.categorical_pipeline(categorical_attributes=None,
                              handle_unknown='ignore',
                              categories_file=r'../data/categorical_categories.dat')

        # Numeric features
        numeric_pipe = self.numeric_pipeline(numeric_attributes=None)

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


    def numeric_pipeline(self, numeric_attributes=None):
        """Return a numeric pipeline
        The numeric pipeline will scale numeric attributes in your dataset
        using sklearn.preprocessing.StandardScaler
        inputs
        ------
        numeric_attributes : (list) of str. Each string is a column name of
            a pandas dataframe. If None, then the classes standard numeric
            attributes will be used instead (see self.num_attributes)
        outputs
        -------
        num_pipeline : (sklearn.pipeline.Pipeline) to transform data
            Call cat_pipeline.fit_transform(dataframe) to transform your data
        """
        if numeric_attributes is None:
            numeric_attributes = self.num_attributes

        num_pipeline = Pipeline([
                ('selector', DataFrameSelector(numeric_attributes)),
                ('std_scaler',StandardScaler())
                ])

        return num_pipeline


    def categorical_pipeline(self, categorical_attributes=None,
                             handle_unknown='ignore',
                             categories_file=r'../data/categorical_categories.dat'):
        """Return a categorical pipeline
        The categorical pipeline will one-hot encode categorical attributes
        in your dataset
        inputs
        ------
        categorical_attributes : (list) of str. Each string is a column name of
            a pandas dataframe. If None, then the classes standard categorical
            attributes will be used instead (see self.cat_attributes)
        handle_unknown = (str) how to handle categorical values in sklearn
            one of ['error' | 'ignore']
        outputs
        -------
        cat_pipeline : (sklearn.pipeline.Pipeline) to transform data
            Call cat_pipeline.fit_transform(dataframe) to transform your data
        """

        if categorical_attributes is None:
            categorical_attributes = self.cat_attributes
            categories = self._read_categories(categorical_attributes,
                                               categories_file)
        else:
            categories='auto'

        cat_pipeline = Pipeline([
                ('ReplaceNone', ReplaceNone(categorical_attributes)),
                ('DataFrameSelector', DataFrameSelector(categorical_attributes)),
                ('OneHotEncoder', OneHotEncoder(categories=categories,
                                                handle_unknown=handle_unknown)),
                ])

        return cat_pipeline

    def _read_categories(self, categorical_attributes, categories_file):
        """Read saved categories from disc - see EncodingCategories"""

        categories_dict = EncodingCategories.read_categories_from_disc(categories_file)
        categories_array = []

        for col_name in categorical_attributes:
            categories = categories_dict[col_name]
            categories_array.append(categories)

        return np.array(categories_array)


    def text_pipeline_cluster(self,
                      vocab_size,
                      attributes,
                      heirarchial_weight_word_pattern,
                      seperator='.'):
        """Return a pipeline for use only on text attributes. The point of this
        function is to control removing duplicates and removing numbers

        NOTE : To use this pipeline you should pass a pandas dataframe to its
        fit_transform() method
        NOTE : This pipeline returns a sparse metrix. To get values use
        result.toarray(). To create a dataframe use pd.DataFrame(text_prepared.toarray(),
        columns=word_vocab)
        To retrun the full word vocabulary use
        word_vocab = name_pipeline.named_steps['WordDictToSparseTransformer'].vocabulary

        parameters
        -------
        vocab_size : (int or str) vocabulary size for one-hot text attributes.
            Pass int or string 'all' to include all vocabulary
        attributes : (string or list) column names to select. Should be list to
            return a 2D array, or a string to return a 1D array. Values should
            be one or multiple of ['NAME', 'SYSTEM','DESCRIPTOR']
        seperator : (string or list) seperator for text features in the'attribute'.
            Typical values are '.' for NAME, or ' ' for SYSTEM or DESCRIPTOR
            feature. Used in TextToWordDict transformer. Can be single character
            string or iterable of character strings
        heirarchial_weight_word_pattern : (bool) setting this to True will
            weight each word by the order that it appears in the input sequence.
            For example, the word phrase 'foo.bar.baz.fizz' will be given
            counts that relate inversely to their position in the sequence. The
            resulting word count will be a counter object of
            Counter({'foo': 4, 'bar': 2, 'baz': 1, 'fizz': 1}). If used with
            the text_pipeline or WordDictToSparseTransformer this will be
            encoded into the array [4,3,2,1].

        ouput
        -------
        A sklearn pipeline object containing modifier classes. To view the modifiers
        see Pipeline.named_steps attribute or Pipeline.__getitem__(ind)

        Example Usage

        database = pd.DataFrame

        # Create 'clean' data processing pipeline
        clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False,
                                              replace_numbers=False,
                                              remove_virtual=True)

        # 'clean' transformed pandas dataframe
        df_clean = clean_pipe.fit_transform(database)

        # Create pipeline specifically for clustering text features
        text_pipe = myDBPipe.text_pipeline(vocab_size='all',
                                           attributes='NAME',
                                           heirarchial_weight_word_pattern=True,
                                           seperator='.')
        X = text_pipe.fit_transform(df_clean).toarray()
        _word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
        df_text = pd.DataFrame(X, columns=_word_vocab)
        """

        name_pipeline = Pipeline([
                ('DataFrameSelector', DataFrameSelector(attributes)),
                ('TextToWordDict', TextToWordDict(seperator=seperator,
                    heirarchial_weight_word_pattern=heirarchial_weight_word_pattern)),
                ('WordDictToSparseTransformer', WordDictToSparseTransformer(
                        vocabulary_size=vocab_size))
        ])

        return name_pipeline


    def text_pipeline_label(self, attributes,
                            token_pattern=r"(?u)\b\w\w+\b",
                            vocabulary=None):
        """Return a text vectorizer for multi instance labeling
        inputs
        -------
        attribute : (str) name of feature column to bianize
        token_pattern : (str) regex pattern used to tokenize text
        vocabulary : (list) of vocabulary in feature
        outputs
        -------
        text_pipe : (sklearn.Pipeline)

        Example
        """
        msg='attributes must be type list, not {}'.format(type(attributes))
        assert isinstance(attributes, list), msg
        msg='Only one attribute per text pipeline'
        assert len(attributes) == 1, msg

        # vocabulary = None means get vocabulary from passed data
        Count = CountVectorizer(input='content',
                                stop_words=None,
                                vocabulary=vocabulary,
                                token_pattern=token_pattern)

        text_pipeline = Pipeline([
            ('DataFrameSelector',DataFrameSelector(attributes)),
            ('ArrayResize', ArrayResize(shape=-1)),
            ('CountVectorizer', Count),
            ])

        return text_pipeline


    def cleaning_pipeline(self,
                          drop_attributes=None,
                          nan_replace_dict=None,
                          dtype_dict=None,
                          unit_dict=None,
                          remove_dupe=True,
                          replace_numbers=True,
                          remove_virtual=True):
        """Cleaning pipeline
        Remove attributes (self.drop_attributes)
        Remove nan (self.nan_replace_dict)
        Change data types (self._type_dict)
        Clean Text (self._text_clean_attrs, replace_numbers)
        Unit Cleaner (self.unit_dict)
        Remove Duplicates (self.dupe_cols, remove_dupe)
        Virtual Remove (remove_virtual)
        NOTE : To use this pipeline you should pass a pandas dataframe to its
        fit_transform() method
        inputs
        -------
        drop_attributes : (list) of str. Each str is a column name to drop
            in your dataset. If None then the class default of self.drop_attributes
            is used
        replace_nan : (dict) of key:value where key is the column name in your
            dataset that contains nan values, and value is the value to replace
            the nan with. If None then the lass default of self.nan_replace_dict
            is used. see self.nan_replace_dict
        mod_dtypes : (dict) of key:value where key is the column name in your
            dataset that contains mixed data types, and value is the dtype to
            cast all values in the dataset as
            If None then the lass default of self._type_dict is used
        mod_units : (dict) of key:value where key is the a value to replace
            in the 'DEVUNITS' column, and value is the value to replace key
            with. Only useful for my specific dataset :). See self.unit_dict
        remove_dupe : (bool) whether or not to remove duplicates from the
            'NAME' column. Duplicates happen with L2SL values.
        replace_numbers : (bool) whether or not to replace numbers in TextCleaner,
            see self._text_clean_attrs
        remove_virtual : (bool) whether or not to remove virtuals in VirtualRemover.
            Virtual points are variables in PPCL programming or used for
            variables in the BAS database. They contrast logical points which
            relate to a physical field sensor, device, or reading
        ouput
        -------
        A sklearn pipeline object containing modifier classes.
        To view the modifiers
        see Pipeline.named_steps attribute or Pipeline.__getitem__(ind)

        Example usage
        X = pd.DataFrame(data, columns, index)
        cleaning_pipeline = cleaning_pipeline(remove_dupe=True,
                                              replace_numbers=True,
                                              remove_virtual=True)
        dataframe = cleaning_pipeline.fit_transform(X)"""


        if drop_attributes is None:
            drop_attributes = self.drop_attributes
        if nan_replace_dict is None:
            nan_replace_dict = self.nan_replace_dict
        if dtype_dict is None:
            dtype_dict = self.type_dict
        if unit_dict is None:
            unit_dict = self.unit_dict

        cleaning_pipeline = Pipeline([
                ('RemoveAttribute', RemoveAttribute(drop_attributes)),
                ('RemoveNan', RemoveNan(nan_replace_dict)),
                ('SetDtypes', SetDtypes(dtype_dict)),
                ('TextCleaner', TextCleaner(self._text_clean_attrs, replace_numbers=replace_numbers)),
                ('UnitCleaner', UnitCleaner(unit_dict)),
                ('DuplicateRemover', DuplicateRemover(self.dupe_cols, remove_dupe=remove_dupe)),
                ('VirtualRemover', VirtualRemover(remove_virtual=remove_virtual))
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
        categories_dict : (dict) with key representing column name and
        value is an array representing unique valeus under column_name
        Example
        categories_dict = {'TYPE' : <class 'numpy.ndarray'>
                           'ALARMTYPE' : <class 'numpy.ndarray'>
                           'FUNCTION'' : <class 'numpy.ndarray'>
                           'VIRTUAL'' : <class 'numpy.ndarray'>
                           'CS'' : <class 'numpy.ndarray'>
                           'SENSORTYPE'' : <class 'numpy.ndarray'>
                           'DEVUNITS' : <class 'numpy.ndarray'>}"""

        with open(save_path, 'wb') as f:
            pickle.dump(categories_dict, f)

        return None

    @staticmethod
    def calc_categories_dict(dataframe, columns):
        """Get the categories of value from the columns of a dataframe
        inputs
        -------
        dataframe : (pd.DataFrame) data to extract categories from
        columns : (list) of strings representing column names of the passed
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
    def points_group_generator():
        """Iterate over points by common customer ID
        inputs
        -------
        outputs
        -------
        df_clean : (pd.DataFrame) of cleaned customer database points grouped
        by the common customer"""
        # Transform pipeline
        Transform_ = Transform()
        # Create 'clean' data processing pipeline
        clean_pipe = Transform_.cleaning_pipeline(drop_attributes=None,
                                                 nan_replace_dict=None,
                                                 dtype_dict=None,
                                                 unit_dict=None,
                                                 remove_dupe=True,
                                                 replace_numbers=True,
                                                 remove_virtual=True)

        sql = """SELECT id
        FROM {}""".format(Customers.__tablename__)
        sel = sqltext(sql)
        customer_ids = Insert.core_select_execute(sel)

        for row in customer_ids:
            customer_id = row.id

            sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
            dfraw = Insert.pandas_select_execute(sel)
            if dfraw.shape[0] <= 1:
                print('Customer ID {} has no associated points'.format(customer_id))
                print('Customer will be skipped')
                continue

            try:
                df_clean = clean_pipe.fit_transform(dfraw)
            except Exception as e:
                print('Transformation pipeline error at {}'.format(customer_id))
                print(e)
                continue

            yield df_clean

        return None

    @staticmethod
    def get_building_suffix(words):
        """Decide if a token is the building suffix of some dataset
        The building suffix is generally defined by this statistical property :
        1) it occurs in most of the word names
        2) it is the first token of the name

        A token will be categorized as a building suffix if
        a) 90% of points include it
        b) 90% of names incldue the suffix as their first token

        Example
        words = [acc.hm.chw.asvs, acc.hm.chw.muflow, acc.hm.chw.swt,
        acc.hm.chw.bpflow, acc.hm.chw.bpv, acc.hm.chw.flow]
        parts = [['acc', 'hm', 'chw', 'asvs'],
                 ['acc', 'hm', 'chw', 'muflow'],
                 ['acc', 'hm', 'chw', 'swt'],
                 ['acc', 'hm', 'chw', 'bpflow'],
                 ['acc', 'hm', 'chw', 'bpv'],
                 ['acc', 'hm', 'chw', 'flow']]"""

        msg = "parts must be type list not {}".format(type(words))
        assert isinstance(words, list), msg

        # For counting (duh)
        counter = Counter()

        # Keep track of all name suffixes
        suffixes = set([x[0] for x in words])

        # Number of names
        n_names = len(words)

        # Pre-compute counts of tokens
        for name in words:
            counter.update(name)

        for suffix in suffixes:
            n_occurences = counter[suffix]

            if (n_occurences / n_names) >= 0.9:
                return suffix

        return False


    def get_fobidden_vocabulary(self):
        """Iterate throguh all database points stored in SQL and retrieve
        building sufflix acronyms
        The building suffix acronyms will be excludied from the point naming
        text 'bag-of-words' feature encoding
        inputs
        -------
        None
        outputs
        -------
        forbidden_vocabulary : (list) of strings that are assumed to be building
        suffixes. Exclude these from feature name encodings"""
        # Prepare to tokenize words
        token_pattern = r'\.'
        tokenizer = re.compile(token_pattern)

        # Keep track of building suffixes
        forbidden_vocabulary = []

        # Get generator of points databases
        df_generator = self.points_group_generator()

        # iterate through word names and find building suffix (like acc, tfc, rgc)
        for df_clean in df_generator:

            # Keep track of words
            words = []

            # Split each name into tokens
            for idx, word in df_clean['NAME'].iteritems():
                parts = tokenizer.split(word)
                words.append(parts)

            suffix = self.get_building_suffix(words)
            if suffix: # Suffix is False if the words do not contain a building suffix
            # As determined in get_building_suffix()
                forbidden_vocabulary.append(suffix)

        return forbidden_vocabulary


    def get_text_vocabulary(self, X, col_name, remove_suffix=True, max_features=None):
        """Get the entire vocabulary of the feature col_name from X
        Use this method specifically to get the point name vocabulary
        from the 'NAME' attribute of my dataset. If remove_suffix is True
        then a list of forbidden building suffix acronyms will be found,
        and the final vocabulary will be the set difference of the total
        vocabulary and the building suffix acronyms
        inputs
        -------
        X : (pd.DataFrame)
        col_name : (str) should be 'NAME', or others if I want
        remove_suffix : (bool) True to calculate and exclude building suffix acronyms
        max_features : (int) If not None, build a vocabulary that only consider
            the top max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.
        outputs
        -------
        vocabulary : (list) of str representing total vocabulary of col_name
        feature from X

        Example Usage"""

        # Calculate total vocabulary
        # Default - 2 or more alphanumeric characters
        token_pattern = r"(?u)\b\w\w+\b"
        Count = CountVectorizer(input='content',
                                stop_words=None,
                                vocabulary=None,
                                token_pattern=token_pattern,
                                max_features=max_features)
        Count.fit(X[col_name])
        # Total vocabulary
        feature_names = Count.get_feature_names()

        # Remove building suffix if remove_suffix is True and col_name is NAME
        if remove_suffix and col_name=='NAME':
            forbidden_vocabulary = set(self.get_fobidden_vocabulary())
            vocabulary = list(set(feature_names).difference(forbidden_vocabulary))
        else:
            vocabulary = feature_names

        return vocabulary

    @staticmethod
    def save_vocabulary(vocabulary, file_name=r'../data/vocab_name.txt'):
        """Convenience to save a vocabulary to file_name
        inputs
        -------
        vocabulary : (list) of vocabulary strings
        file_name : (str) to save vocabulary to"""

        if os.path.isfile(file_name):
            x = input('File {} Already exists. Overwrite? [y/n]'.format(file_name))
            if x in ['y','yes','True','YES','Y','TRUE']:
                pass
            else:
                return None

        with open(file_name, 'wt') as f:
            for vocab in vocabulary:
                f.write(vocab + '\n')

        return None

    @staticmethod
    def read_vocabulary_disc(file_name):
        """Read vocabulary from a file and aggregate to a list
        inputs
        ------
        none
        outputs
        -------
        vocabulary : (list) of str"""

        with open(file_name, 'rt') as f:
            vocabulary = f.read().splitlines()

        return vocabulary