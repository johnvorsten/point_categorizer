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

#Local Imports
from JVWork_UnClusterAccuracy import AccuracyTest
from JVWork_UnsupervisedCluster import JVClusterTools
from JVWork_WholeDBPipeline import JVDBPipe

#Instantiate classes
myTest = AccuracyTest() #For performing unsupervised clustering
myClustering = JVClusterTools() #For retrieving data
myDBPipe = JVDBPipe() #Class in this module

# Optionally, create an iterator over databases on disc
_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
my_iter = myClustering.read_database_set(_master_pts_db)


sequence_tag = 'DBPath'
_, database = next(my_iter)
error_df = myTest.error_df

# Create a pipeline for cleaning the raw dataset
clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False,
                                      replace_numbers=False,
                                      remove_virtual=True)

# Apply cleaning pipeline
df_clean = clean_pipe.fit_transform(database)

# Create a pipeline for encoding text features
text_pipe = myDBPipe.text_pipeline(vocab_size='all', attributes='NAME',
                                   seperator='.')

# Extract text features
X = text_pipe.fit_transform(df_clean).toarray()

# Tip : examine the word vocabulary
word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary #Total dictionary of words in dataset

# Dataframe with each column naming the encoded word (feature).
# Instances are along (axis 0)
df_text = pd.DataFrame(X, columns=_word_vocab)


@author: z003vrzk
"""

# Python imports
from pathlib import Path
import re
from collections import Counter
import statistics

# Third party imports
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
import numpy as np


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

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attributeNames].values

class TextCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, columns, replace_numbers=True):
        """columns : columns to clean text in. Must be list of column indicies
        replace_numbers : Boolean for wither to repalce numbers with empty string"""
        self.REPLACE_BY_EMPTY_RE= re.compile('[/(){}\[\]\|@\\\,;]')
        self.BAD_SYMBOLS_RE = re.compile('[^a-zA-Z0-9 _.]')
        self.NUMBERS_RE = re.compile('[0-9]')
        self.columns = columns
        self.replace_numbers = replace_numbers

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
        self.type_dict = type_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        for idx in X['INITVALUE'].index:
            try:
                X.at[idx, 'INITVALUE'] = float(X.at[idx, 'INITVALUE'])
            except:
                X.loc[idx, 'INITVALUE'] = 0

        for col, my_type in self.type_dict.items():
            X[col].astype(my_type)

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
                X.dropna(axis=0, subset=[col], inplace=True)
            elif method == 'empty':
                X[col].fillna(value='', axis=0, inplace=True)
            elif method == 'zero':
                X[col].fillna(value=0, axis=0, inplace=True)
            elif method == 'mode':
                col_mode = X[col].mode()[0]

                X[col].fillna(value=col_mode, axis=0, inplace=True)
            else:
                X[col].fillna(value=method, axis=0, inplace=True)

        X.reset_index(drop=True, inplace=True)
        return X

class UnitCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, unit_dict):
        self.unit_dict = unit_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        for old_unit, new_unit in self.unit_dict.items():
            indicies = X.index[X['DEVUNITS'] == old_unit].tolist()
            X.loc[indicies, 'DEVUNITS'] = new_unit

        return X

#class OrderedCounter(Counter, dict):
#    'Counter that remembers the order elements are first encountered'
#    def __repr__(self):
#        return '%s(%r)' % (self.__class__.__name__, dict(self))
#
#    def __reduce__(self):
#        return self.__class__, (dict(self),)


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

        word_length_mode = statistics.mode(word_lengths)

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

    def __init__(self, vocabulary_size = 'all'):
        """parameters
        -------
        vocabulary_size : int or 'all'. int will return specified size. 'all'
        will return all unique words available"""
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        #Do stuff : get words in dictionary and save in master dictionary
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
        #Do stuff : iterate through each counter object in the [Counter({})] passed to this transformer
        #For each item in the list, output a sparse matrix based on # of words that match the master dict
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
        assert type(dupe_cols) == list, 'dupe_cols must be list'
        self.dupe_cols = dupe_cols
        self.remove_dupe = remove_dupe

    def fit(self, X, y=None):
        #Find all duplicates in column described
        self.master_dict = {}

        #Create Dictionary
        for col in self.dupe_cols:
            for word in X[col]:
                if self.master_dict.__contains__(word):
                    self.master_dict[word] += 1
                else:
                    self.master_dict[word] = 1

        self.repeats = []
        for key, count in self.master_dict.items():
            if count >= 2:
                self.repeats.append(key)
        return self

    def transform(self, X, y=None): #Delete repeats

        if self.remove_dupe:

            for col in self.dupe_cols:
                for repeat in self.repeats:
                    #TODO Should i prioritize indicies that dont have NETDEV to be deleted?
                    indicies = np.where(X[col] == repeat)[0]
                    X.drop(index=indicies[1:], axis=0, inplace=True)

            X.reset_index(drop=True, inplace=True)
        return X

class VirtualRemover(BaseEstimator, TransformerMixin):

    def __init__(self, remove_virtual=False):
        assert type(remove_virtual) ==  bool, 'remove_virtual must be boolean'
        self.remove_virtual = remove_virtual

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None): #Delete repeats

        if self.remove_virtual:

            indicies = np.where(X['VIRTUAL'] == True)[0]
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

        self._text_attrs = ['NAME','DESCRIPTOR','TYPE','FUNCTION','SYSTEM','SENSORTYPE','DEVUNITS']

        self._type_dict = {'NAME':str, 'NETDEVID':str, 'DESCRIPTOR':str, 'INITVALUE':float,'ALARMTYPE':str,
                      'FUNCTION':str, 'VIRTUAL':bool,'SYSTEM':str,'CS':str,'SENSORTYPE':str,
                      'DEVUNITS':str}

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

        Initialization attributes descriptions

        These attributes are not deemed useful for clustering and will be
        dropped
        self._drop_attributes = ['CTSYSNAME', 'TMEMBER', 'ALARMHIGH', 'ALARMLOW',
                           'COMBOID', 'PROOFPRSNT', 'PROOFDELAY', 'NORMCLOSE', 'INVERTED',
                            'LAN', 'DROP', 'POINT', 'ADDRESSEXT', 'DEVNUMBER', 'CTSENSTYPE',
                             'CONTRLTYPE', 'UNITSTYPE', 'SIGUNITS', 'NUMBERWIRE', 'POWER',
                             'WIRESIZE', 'WIRELENGTH', 'S1000TYPE']

        This is archived information. These attributes might not be useful
        self._maybe_drop_attr = ['SLOPE', 'INTERCEPT']

        Text attributes are used to indidcate features that are pure-text
        self._text_attrs = ['NAME','DESCRIPTOR','TYPE','FUNCTION','SYSTEM','SENSORTYPE','DEVUNITS']

        type_dict maps feature names to their desired data type
        self._type_dict = {'NAME':str, 'NETDEVID':str, 'DESCRIPTOR':str, 'INITVALUE':float,'ALARMTYPE':str,
                      'FUNCTION':str, 'VIRTUAL':bool,'SYSTEM':str,'CS':str,'SENSORTYPE':str,
                      'DEVUNITS':str}

        How to deal with nan's -
        remove : remove the whole row
        empty : replace with empty string
        zero : replace with 0
        mode : replace with mode
        any other value : replace with that value
        self._nan_replace_dict = {'NETDEVID':'empty', 'NAME':'remove',
                                  'DESCRIPTOR':'empty','TYPE':'mode',
                                  'INITVALUE':'zero', 'ALARMTYPE':'mode',
                                  'FUNCTION':'Value','VIRTUAL':'False',
                                  'SYSTEM':'empty','CS':'empty',
                                  'SENSORTYPE':'digital','DEVICEHI':'zero',
                                  'DEVICELO':'zero', 'DEVUNITS':'empty',
                                  'SIGNALHI':'zero','SIGNALLO':'zero',
                                  'SLOPE':'zero','INTERCEPT':'zero'}

        Attributes that will be used to cluster OR that should pass through
        the cleaning pipeline
        self._text_clean_attrs = ['NAME','DESCRIPTOR','SYSTEM']

        unit types have to do with the 'correct' type of units to use in raw
        data. UNITS is a data feature that needed cleaning up a lot
        _unit_df = pd.read_csv(r"D:\Z - Saved SQL Databases\clean_types_manual.csv", index_col=0)
        self._unit_dict = {}
        for _, _units in (_unit_df.iterrows()):
            _old_unit = _units['old']
            _new_unit = _units['new']
            if _new_unit == '0':
                _new_unit = ''
            self._unit_dict[_old_unit] = _new_unit

        Remove duplicate columns
        self._dupe_cols = ['NAME']

        These attributes should be on-hot encoded
        self._cat_attributes = ['TYPE','ALARMTYPE','FUNCTION',
                                'VIRTUAL','CS','SENSORTYPE','DEVUNITS']

        These are numerical attributes
        self._num_attributes = ['INITVALUE','DEVICEHI','DEVICELO',
                                'SIGNALHI','SIGNALLO','SLOPE','INTERCEPT']

        These are text attributes
        self._text_attributes = ['NAME', 'DESCRIPTOR', 'SYSTEM']
        """

        self._drop_attributes = ['CTSYSNAME', 'TMEMBER', 'ALARMHIGH', 'ALARMLOW',
                           'COMBOID', 'PROOFPRSNT', 'PROOFDELAY', 'NORMCLOSE', 'INVERTED',
                            'LAN', 'DROP', 'POINT', 'ADDRESSEXT', 'DEVNUMBER', 'CTSENSTYPE',
                             'CONTRLTYPE', 'UNITSTYPE', 'SIGUNITS', 'NUMBERWIRE', 'POWER',
                             'WIRESIZE', 'WIRELENGTH', 'S1000TYPE']

        self._maybe_drop_attr = ['SLOPE', 'INTERCEPT']

        self._text_attrs = ['NAME','DESCRIPTOR','TYPE','FUNCTION','SYSTEM','SENSORTYPE','DEVUNITS']

        self._type_dict = {'NAME':str, 'NETDEVID':str, 'DESCRIPTOR':str, 'INITVALUE':float,'ALARMTYPE':str,
                      'FUNCTION':str, 'VIRTUAL':bool,'SYSTEM':str,'CS':str,'SENSORTYPE':str,
                      'DEVUNITS':str}

        self._nan_replace_dict = {'NETDEVID':'empty', 'NAME':'remove',
                                  'DESCRIPTOR':'empty','TYPE':'mode',
                                  'INITVALUE':'zero','ALARMTYPE':'mode',
                                  'FUNCTION':'Value','VIRTUAL':'False',
                                  'SYSTEM':'empty','CS':'empty',
                                  'SENSORTYPE':'digital','DEVICEHI':'zero',
                                  'DEVICELO':'zero','DEVUNITS':'empty',
                                  'SIGNALHI':'zero','SIGNALLO':'zero',
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

        return None


    def full_pipeline(self,
                       remove_dupe=False,
                       replace_numbers=True,
                       remove_virtual=False,
                       seperator='.',
                       vocabulary_size='all'):
        """Return a full pipeline feature union
        inputs
        -------
        remove_dupe : (bool) whether or not to remove duplicates from the
        'NAME' column
        replace_numbers : (bool) replace numbers in text features with
        empty strings (True)
        remove_virtual : (bool) remove "virtual" points
        seperator : (str) seperator for text features in the'Name' column. Used in
        TextToWordDict transformer. Can be single character or iterable of characters
        vocabulary_size : (int) number of featuers to encode in name,
        descriptor, and sysetm features text

        Example usage
        # Import your data in a dataframe
        input_dataframe = dataframe

        # Transform data
        clean_dataframe = full_pipeline.fit_transform(input_dataframe)

        # Categorical columns
        cat_cols = cat_pipeline.named_steps.catEncoder.categories_
        categorical_columns = np.concatenate(_cat_cols).ravel().tolist()

        # TODO
        df2_cat_num_cols = [num_attributes, cat_columns]
        df2_cat_num = pd.DataFrame(df2_cat_num_scr.toarray(),
                                   columns=[item for sublist in _df2_cat_num_cols for item in sublist])

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


        return full_pipeline


    def text_pipeline(self,
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


    def cleaning_pipeline(self,
                          remove_dupe,
                          replace_numbers,
                          remove_virtual):
        """Cleaning pipeline
        Remove attributes (self._drop_attributes)
        Remove nan (self._nan_replace_dict)
        Change data types (self._type_dict)
        Clean Text (self._text_clean_attrs, replace_numbers)
        Unit Cleaner (self._unit_dict)
        Remove Duplicates (self._dupe_cols, remove_dupe)
        Virtual Remove (remove_virtual)
        NOTE : To use this pipeline you should pass a pandas dataframe to its
        fit_transform() method
        inputs
        -------
        remove_dupe : (bool) whether or not to remove duplicates from the
            'NAME' column.
            Duplicates happen with L2SL values.
        replace_numbers : (bool) whether or not to replace numbers in TextCleaner,
            see self._text_clean_attrs
        remove_virtual : (bool) whether or not to remove virtuals in VirtualRemover
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


