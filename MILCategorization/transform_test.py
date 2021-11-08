# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 18:10:48 2021

@author: vorst
"""

# Python imports
import sys
import os
import time, re, pickle

# Third party imports
import sqlalchemy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Local imports
from transform import Transform # Changed

if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

from extract import extract
from extract.SQLAlchemyDataDefinition import (Clustering, Points, Netdev,
                          Customers, ClusteringHyperparameter, Labeling,
                          TypesCorrection)


#%%

def test_cleaning_pipe():

    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    dataset_raw = Insert.pandas_select_execute(sel)

    # Transform pipeline
    Transform = transform_pipeline.Transform()
    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                             nan_replace_dict=None,
                                             dtype_dict=None,
                                             unit_dict=None,
                                             remove_dupe=True,
                                             replace_numbers=False,
                                             remove_virtual=True)

    df = clean_pipe.fit_transform(dataset_raw)

    return df

def test_categorical_pipe():

    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    dataset_raw = Insert.pandas_select_execute(sel)

    # Transform pipeline
    Transform = transform_pipeline.Transform()
    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                             nan_replace_dict=None,
                                             dtype_dict=None,
                                             unit_dict=None,
                                             remove_dupe=True,
                                             replace_numbers=False,
                                             remove_virtual=True)

    categorical_pipe = Transform.categorical_pipeline(categorical_attributes=None,
                          categories_file=r'../data/categorical_categories.dat')

    df_clean = clean_pipe.fit_transform(dataset_raw)
    ohe_array = categorical_pipe.fit_transform(df_clean).toarray()

    # Find more about categorical pipe
    ohe = categorical_pipe.named_steps['catEncoder']
    ohe.categories # ohe.categories_ when categories='auto'

    return ohe_array

def test_read_categories():

    # Ititialize
    Transform = transform_pipeline.Transform()
    categories_file = r'../data/categorical_categories.dat'
    categories = Transform._read_categories(Transform.cat_attributes, categories_file)
    categorical_attributes = Transform.cat_attributes

    ReplaceNone = transform_pipeline.ReplaceNone(categorical_attributes)
    DataFrameSelector = transform_pipeline.DataFrameSelector(categorical_attributes)
    OneHotEncoder = transform_pipeline.OneHotEncoder(categories=categories)

    # Get raw database
    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')
    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    dataset_raw = Insert.pandas_select_execute(sel)
    clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                             nan_replace_dict=None,
                                             dtype_dict=None,
                                             unit_dict=None,
                                             remove_dupe=True,
                                             replace_numbers=False,
                                             remove_virtual=True)
    df_clean1 = clean_pipe.fit_transform(dataset_raw)

    # Transform
    df0 = ReplaceNone.fit_transform(df_clean1)
    df1_array = DataFrameSelector.fit_transform(df0)
    ohearray = OneHotEncoder.fit_transform(df1_array).toarray()

    # Examine the transformers
    print(df0[categorical_attributes].iloc[:5])
    print(df1_array[:5])
    OneHotEncoder.categories

    return None


def test_numeric_pipe():

    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    dataset_raw = Insert.pandas_select_execute(sel)

    # Transform pipeline
    Transform = transform_pipeline.Transform()
    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                             nan_replace_dict=None,
                                             dtype_dict=None,
                                             unit_dict=None,
                                             remove_dupe=True,
                                             replace_numbers=True,
                                             remove_virtual=True)

    numeric_pipe = Transform.numeric_pipeline(numeric_attributes=None)

    df_clean = clean_pipe.fit_transform(dataset_raw)
    df_numeric = numeric_pipe.fit_transfor(df_clean)

    return df_numeric

def test_text_pipe():

    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    dataset_raw = Insert.pandas_select_execute(sel)

    # Transform pipeline
    Transform = transform_pipeline.Transform()
    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                             nan_replace_dict=None,
                                             dtype_dict=None,
                                             unit_dict=None,
                                             remove_dupe=True,
                                             replace_numbers=True,
                                             remove_virtual=True)


    # Create pipeline specifically for clustering text features
    text_pipe = Transform.text_pipeline(vocab_size='all',
                                       attributes='NAME',
                                       seperator='.',
                                       heirarchial_weight_word_pattern=True)

    full_pipeline = Pipeline([('clean_pipe', clean_pipe),
                              ('text_pipe', text_pipe),
                              ])

    dataset = full_pipeline.fit_transform(dataset_raw)

    return dataset


def test_time():

    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    customer_id = 15
    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
    dataset_raw = Insert.pandas_select_execute(sel)

    # Transform pipeline
    Transform = transform_pipeline.Transform()

    RemoveAttribute = transform_pipeline.RemoveAttribute(Transform.drop_attributes)
    RemoveNan = transform_pipeline.RemoveNan(Transform.nan_replace_dict)
    SetDtypes = transform_pipeline.SetDtypes(Transform.type_dict)
    TextCleaner = transform_pipeline.TextCleaner(Transform._text_clean_attrs, replace_numbers=True)
    UnitCleaner = transform_pipeline.UnitCleaner(Transform.unit_dict)
    DuplicateRemover = transform_pipeline.DuplicateRemover(Transform.dupe_cols, remove_dupe=True)
    VirtualRemover = transform_pipeline.VirtualRemover(remove_virtual=True)

    t0 = time.time()
    df0 = RemoveAttribute.fit_transform(dataset_raw)

    t1 = time.time()
    df1 = RemoveNan.fit_transform(df0)

    t2 = time.time()
    df2 = SetDtypes.fit_transform(df1)

    t3 = time.time()
    df3 = TextCleaner.fit_transform(df2)

    t4 = time.time()
    df4 = UnitCleaner.fit_transform(df3)

    t5 = time.time()
    indicies = DuplicateRemover.get_duplicate_indicies(df4, 'NAME')
    print('Duplicate names')
    print(df4['NAME'].iloc[indicies[:50]])
    df5 = DuplicateRemover.fit_transform(df4)

    t6 = time.time()
    df6 = VirtualRemover.fit_transform(df5)
    t7 = time.time()

    print('RemoveAttribute : {}'.format(t1 - t0))
    print('RemoveNan : {}'.format(t2 - t1))
    print('SetDtypes : {}'.format(t3 - t2))
    print('TextCleaner : {}'.format(t4 - t3))
    print('UnitCleaner : {}'.format(t5 - t4))
    print('DuplicateRemover : {}'.format(t6 - t5))
    print('VirtualRemover : {}'.format(t7 - t6))

    return None


def test_calc_categories_dict():

    """Generate data to find categories"""
    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    sel = sqlalchemy.select([Points])
    dataset_raw = Insert.pandas_select_execute(sel)

    # Transform pipeline
    Transform = transform_pipeline.Transform()
    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                             nan_replace_dict=None,
                                             dtype_dict=None,
                                             unit_dict=None,
                                             remove_dupe=True,
                                             replace_numbers=False,
                                             remove_virtual=True)
    string_pipe = transform_pipeline.SetDtypes(
        type_dict={'TYPE':str, 'ALARMTYPE':str,
                   'FUNCTION':str, 'VIRTUAL':str,
                   'CS':str, 'SENSORTYPE':str,
                   'DEVUNITS':str})

    categories_clean_pipe = Pipeline([
        ('clean_pipe',clean_pipe),
        ('string_pipe',string_pipe)
        ])

    df_clean = categories_clean_pipe.fit_transform(dataset_raw)

    """Calculate and save categories to be used later"""
    Encoding = transform_pipeline.EncodingCategories()
    columns = ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL',
               'CS', 'SENSORTYPE', 'DEVUNITS']
    categories_dict = Encoding.calc_categories_dict(df_clean, columns)
    save_path = r'../data/categorical_categories.dat'

    Encoding.save_categories_to_disc(categories_dict, save_path)
    categories_dict1 = Encoding.read_categories_from_disc(save_path)
    for key in set((*categories_dict.keys(), *categories_dict1.keys())):
        assert(np.array_equal(categories_dict[key], categories_dict1[key]))

    return None


def test_get_building_suffix():
    """Test whether a set of points is a building suffix word"""

    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(18))
    dataset_raw = Insert.pandas_select_execute(sel)

    # Split name variable
    token_pattern = r'\.'
    tokenizer = re.compile(token_pattern)

    # Keep track of words
    words = []

    # Split each name into tokens
    for idx, word in dataset_raw['NAME'].iteritems():
        parts = tokenizer.split(word)
        words.append(parts)

    # Get vocabulary
    VocabularyText = transform_pipeline.VocabularyText()
    suffix = VocabularyText.get_building_suffix(words)
    print("Suffix found : ", suffix)

    return None

def test_get_text_vocabulary():

    """Generate data to find Vocabulary"""
    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')

    sel = sqlalchemy.select([Points])
    dataset_raw = Insert.pandas_select_execute(sel)

    # Transform pipeline
    Transform = transform_pipeline.Transform()
    # Create 'clean' data processing pipeline
    clean_pipe = Transform.cleaning_pipeline(drop_attributes=None,
                                             nan_replace_dict=None,
                                             dtype_dict=None,
                                             unit_dict=None,
                                             remove_dupe=True,
                                             replace_numbers=True,
                                             remove_virtual=True)

    df_clean = clean_pipe.fit_transform(dataset_raw)

    # Get vocabulary for DESCRIPTOR feature - a text feature
    VocabularyText = transform_pipeline.VocabularyText()
    vocabulary = VocabularyText\
        .get_text_vocabulary(X=df_clean,
                             col_name='DESCRIPTOR',
                             remove_suffix=False,
                             max_features=80)

    # Sove vocabulary
    file_name = r'../data/vocab_descriptor.txt'
    transform_pipeline.VocabularyText.save_vocabulary(vocabulary, file_name)

    return None


def test_full_pipeline():

    Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                            driver_name='SQL Server Native Client 10.0',
                            database_name='Clustering')
    # group_id = 4
    group_id = 15
    sel = sqlalchemy.select([Points]).where(Points.group_id.__eq__(group_id))
    dfraw = Insert.pandas_select_execute(sel)

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
        ('CombinedFeatures',combined_features),
        ])

    combined_csr = full_pipeline.fit_transform(dfraw)
    combined_csr.shape

    CleaningPipe = full_pipeline.steps[0][1] # CleaningPipe
    RemoveAttribute = full_pipeline.steps[0][1][0] # RemoveAttribute
    RemoveNan = full_pipeline.steps[0][1][1]
    SetDtypes = full_pipeline.steps[0][1][2]
    TextCleaner = full_pipeline.steps[0][1][3]
    UnitCleaner = full_pipeline.steps[0][1][4]
    DuplicateRemover = full_pipeline.steps[0][1][5]
    VirtualRemover = full_pipeline.steps[0][1][6]


    df0 = RemoveAttribute.fit_transform(copy.deepcopy(dfraw))
    df1 = RemoveNan.fit_transform(copy.deepcopy(df0))
    df2 = SetDtypes.fit_transform(copy.deepcopy(df1))
    df3 = TextCleaner.fit_transform(copy.deepcopy(df2))
    df4 = UnitCleaner.fit_transform(copy.deepcopy(df3))
    df5 = DuplicateRemover.fit_transform(copy.deepcopy(df4))
    df6 = VirtualRemover.fit_transform(copy.deepcopy(df5))


    return None



# Hard to make work in feature unions
def flatten_pipeline(pipeline):
    """"""
    flat_transformers = []

    for name, step in pipeline.steps:

        # Recursive unpacking of pipeline
        if isinstance(step, Pipeline):
            transformers = flatten_pipeline(step)
            for x in transformers:
                flat_transformers.append(x)

        else:
            flat_transformers.append(step)


    return flat_transformers