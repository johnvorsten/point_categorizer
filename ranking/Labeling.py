# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 19:19:08 2019

This module is useful for calculating features and labels for a given dataset

Example Usage

#%% Calculate database Labels

#Local Imports
from JVWork_AccuracyVisual import import_error_dfs

# Calculate the best hyperparameter labels, and optionally retrieve the hyper_dict
# which shows all losses assosicated with each hyperparameter set
# Across all optimal k indexs

extract = ExtractLabels()

records = import_error_dfs()
db_tag = r'D:\Z - Saved SQL Databases\44OP-093324_Baylor_Bric_Bldg\JobDB.mdf'
labels, hyper_dict = extract.calc_labels(records, db_tag)

#%% Calculate database features Example Usage

from JVWork_UnClusterAccuracy import AccuracyTest
myTest = AccuracyTest()
text_pipe = myDBPipe.text_pipeline(vocab_size='all', attributes='NAME',
                           seperator='.')
clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False,
                              replace_numbers=False,
                              remove_virtual=True)
mypipe = Pipeline([('clean_pipe', clean_pipe),
                   ('text_pipe',text_pipe)
                   ])
correct_k = myTest.get_correct_k(db_name, df_clean, manual=True)


extract = ExtractLabels()
db_feat = extract.calc_features(database, mypipe, tag=db_name)

#%%

@author: z003vrzk
"""

# Python imports
import os
import sys
import pickle
import statistics
from collections import namedtuple
from collections import Counter

#Third party imports
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

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

# Local declarations
loss = namedtuple('loss',['clusters', 'l2error', 'variance', 'loss'])


#%%

class ExtractLabels():

    def __init__(self):
        pass



    def get_database_features(self, database, pipeline, instance_name):
        """Calculate the features of a dataset. These will be used to
        predict a good clustering algorithm.
        Inputs
        -------
        database : Your database. Its type not matter as long as the
            pipeline passed outputs a numpy array.
        pipeline : (sklearn.pipeline.Pipeline) Your finalized pipeline.
            See sklearn.Pipeline. The output of the pipelines .fit_transform()
            method should be your encoded array
            of instances and features of size (n,p) n=#instances, p=#features.
            Alternatively, you may pass a sequence of tuples containing names
            and pipeline objects : [('pipe1', myPipe1()), ('pipe2',myPipe2())]
        instance_name : (str | int) unique tag/name to identify each instance.
            The instance with feature vectors will be returned as a pandas
            dataframe with the tag on the index.
        Returns
        -------
        df : a pandas dataframe with the following features
            a. Number of points
    		b. Correct number of clusters
    		c. Typical cluster size (n_instances / n_clusters)
    		d. Word length
    		e. Variance of words lengths?
    		f. Total number of unique words
            g. n_instances / Total number of unique words

        #Example Usage
        from JVWork_UnClusterAccuracy import AccuracyTest
        myTest = AccuracyTest()
        text_pipe = myDBPipe.text_pipeline(vocab_size='all', attributes='NAME',
                                   seperator='.')
        clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False,
                                      replace_numbers=False,
                                      remove_virtual=True)
        mypipe = Pipeline([('clean_pipe', clean_pipe),
                           ('text_pipe',text_pipe)
                           ])
        correct_k = myTest.get_correct_k(db_name, df_clean, manual=True)

        data_transform = DataFeatures()
        db_feat = data_transform.calc_features(database, mypipe,
                                               tag=db_name, correct_k=correct_k)
        """

        if hasattr(pipeline, '__iter__'):
            pipeline = Pipeline(pipeline)
        else:
            pass

        data = pipeline.fit_transform(database)
        if isinstance(data, csr_matrix):
            data = data.toarray()

        #Number of points
        n_points = data.shape[0]

        #Number of features
        n_features = data.shape[1]

        #Word lengths
        count_dict_pct = self.get_word_dictionary(data, percent=True)

        #Variance of lengths
        lengths = []
        count_dict_whole = self.get_word_dictionary(data, percent=False)
        for key, value in count_dict_whole.items():
            lengths.extend([int(key[-1])] * value)
        len_var = np.array(lengths).var(axis=0)

        features_dict = {
                'instance':instance_name,
                'n_instance':n_points,
                'n_features':n_features,
                'len_var':len_var,
                'uniq_ratio':n_points/n_features,
                }
        features_dict = {**features_dict, **count_dict_pct}
        features_df = pd.DataFrame(features_dict, index=[instance_name])

        return features_df

    def get_word_dictionary(self, word_array, percent=True):

        count_dict = {}

        #Create a dictionary of word lengths
        for row in word_array:
            count = sum(row>0)
            try:
                count_dict[count] += 1
            except KeyError:
                count_dict[count] = 1

        #Return word lengths by percentage, or
        if percent:
            for key, label in count_dict.items():
                count_dict[key] = count_dict[key] / len(word_array) #Percentage
        else:
            pass #Dont change it, return number of each length


#        max_key = max(count_dict.keys())
        max_key = 7 #Static for later supervised training
        old_keys = list(count_dict.keys())
        new_keys = ['n_len' + str(old_key) for old_key in old_keys]

        for old_key, new_key in zip(old_keys, new_keys):
            count_dict[new_key] = count_dict.pop(old_key)

        required_keys = ['n_len' + str(key) for key in range(1,max_key+1)]
        for key in required_keys:
            count_dict.setdefault(key, 0)

        return count_dict



    def calc_labels(self, records, correct_k, var_scale=0.2, error_scale=0.8):
        """Given an instance_name (database name in this case) and set of records
        output the correct labels for that database.
        inputs
        -------
        records : (list | iterable) of records from the Record class. All records
            should be from the same customer/database
        var_scale : (float) contribution of prediction variance to total loss
        of a clustering hyperparameter set. The idea is less variance in predictions
        is better
        error_scale : (flaot) contribution of prediction error to total loss
        of a clustering hyperparameter set. error =

        output
        -------
        A list a labels in string form. The list includes :
            by_size : (bool) True to cluster by size, False otherwise
            distance : (str) 'euclidean' only
            clusterer : (str) clustering algorithm
            reduce : (str) 'MDS','TSNE','False' dimensionality reduction metric
            index : (str) clustering index to determine best number of clusters
            loss : (float) loss indicating error between predicted number
            of clusters and actual number of clusters, and variance of predictions
            n_components : (str) 8, 0, 2 number of dimensions reduced to. 0 if
            no dimensionality reduction was used

        Example Usage
        #Local Imports
        from JVWork_AccuracyVisual import import_error_dfs
        from JVWork_Labeling import ExtractLabels

        # Calculate the best hyperparameter labels, and optionally retrieve the hyper_dict
        # which shows all losses assosicated with each hyperparameter set
        # Across all optimal k indexs

        extract = ExtractLabels()

        records = import_error_dfs()
        db_tag = r'D:\Z - Saved SQL Databases\44OP-093324_Baylor_Bric_Bldg\JobDB.mdf'
        labels, hyper_dict = extract.calc_labels(records, db_tag)
        """

        # Customer records should all be the same
        ids = []
        for record in records:
            ids.append(record.indicies_dictionary['customer_id'])
        assert len(set(ids)).__eq__(1), "All customer IDs must be the same"


        indicies = ['KL','CH','Hartigan','CCC','Marriot','TrCovW',
                'TraceW','Friedman','Rubin','Cindex','DB','Silhouette',
                'Duda','PseudoT2','Beale','Ratkowsky','Ball','PtBiserial',
                'Frey','McClain','Dunn','Hubert','SDindex','Dindex','SDbw',
                'gap_tib','gap_star','gap_max','Scott']

        """keep track of all hyperparameter combinations. Each hyperparameter_set
        is a set of by_size, clusterer, distance, reduce, n_components, indicy
        hyperparameter_sets ~ [ frozenset({'8', False, 'MDS', 'euclidean', 'gap_max', 'ward.D'}),
                               frozenset({'8', False, 'MDS', 'Scott', 'euclidean', 'ward.D'}),
                               etc...]"""
        hyperparameter_sets = []
        for idx, record in enumerate(records):

            # Combine the indicy and hyperparameter dictionary in records
            for indicy in indicies:
                hyperparameter_set = frozenset((*record.hyper_dict.values(), indicy))
                hyperparameter_sets.append(hyperparameter_set)

            # Dont let hyperparameter_sets get too big
            if idx % 10 == 0:
                hyperparameter_sets = list(set(hyperparameter_sets))

        # Get the final collection of hyperparameter sets
        hyperparameter_sets = list(set(hyperparameter_sets))

        """Aggregate predicted number of clusters for each hyperparameter_set
        predicted_clusters ~ {frozenset({'8',False,'MDS','Dindex','euclidean','ward.D'}):[0 0,0,0],
                              frozenset({'8',False,'MDS','SDbw','euclidean','ward.D'}):[93,71,46,12]}"""
        predicted_clusters = {}
        for idx, record in enumerate(records):
            predicted_indicies = record.indicies_dictionary

            # Collect predicted number of clusters for each indicy and
            # aggregate to hyperparameter dictionary
            for indicy in indicies:
                predicted_k = predicted_indicies[indicy] # Integer or None
                if predicted_k is None:
                    # Dont count indicies that are None..
                    continue

                hyperparameter_set = frozenset((*record.hyper_dict.values(), indicy))
                # Aggregate predictions under hyperparameter dictionaries
                try:
                    predicted_clusters[hyperparameter_set].append(predicted_k)
                except KeyError:
                    # The dictionary key is not created yet..
                    predicted_clusters[hyperparameter_set] = [predicted_k]

        """Calculate loss metrics for each hyperparameter set"""
        x = namedtuple('related_items', ['hyperparameter_set','predictions','correct_k','loss'])
        best_losses = []
        for hyperparameter_set, predictions in predicted_clusters.items():
            # Calculate loss associated with
            loss = self.calculate_loss(correct_k,
                                       predictions,
                                       error_weight=0.8,
                                       variance_weight=0.2)
            tup = x(hyperparameter_set, predictions, correct_k, loss)
            best_losses.append(tup)

        best_losses = sorted(best_losses, key=lambda tup: tup.loss)

        return best_losses



    def calculate_loss(self, correct_k, predicted_ks, error_weight, variance_weight):
        """Calculate the custom loss metric
        This loss metric is a combination of error and variance of predictions
        Error is the difference between the correct number of clusters and the
        predicted 'optimal' number of clusters. More formally, it is the
        squarred error of the estimation vector and known vector
        Variance in predictions happens when the 'optimal' number of clusters
        predicted for a dataset changes (for example in kmeans clustering)
        The loss metric is the weighted sum or error and variance

        The loss metric is calculated as follows :
        1. Calculate the squared error across all predictions. The squared
        error is calculated as SE(A,θ)=∑ ||yn−fθ(xn)|| ** 2
        2. Calculate the variance of predictions. The variance is calculated as
        the average of squared deviations from the mean.
        S^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}
        3. Calculate the loss metric,
        loss = squarred_error * error_weight + variance * variance_weight

        inputs
        -------
        correct_k : (int) correct number of clusters for a dataset
        predicted_ks : (list | iterable) predicted number of clusters for a
        dataset. This argument can be iterable if there are multiple predictions
        available on a dataset
        error_weight : (float) between 0 and 1
        variance_weight : (float) between 0 and 1
        """

        # Normalize weight in case they dont pass values 0-1 that sum to 1
        weight = sum((error_weight, variance_weight))
        error_weight = error_weight / weight
        variance_weight = variance_weight / weight

        # Calculate l2 norm of absolute error between correct number of clusters
        # And predicted number of clusters
        predicted_ks = np.array(predicted_ks)
        squared_error = np.sum(np.square(predicted_ks - correct_k))

        # Calculate variance of predictions
        variance = np.var(predicted_ks)

        # Custom loss metric calculation
        custom_loss = (squared_error * error_weight + variance * variance_weight)

        return custom_loss


def choose_best_hyperparameter(labels, hyperparameter, top_pct=0.1, top_n=10, top_thresh=5):
    """Choose the best hyperparameter given a set of labels from ExtractLabels
    inputs
    ------
    labels : a dictionary of labels
    hyperparamter : (str) name of hyperparamter in labels. It must be a key
        in the labels nested dictionary
    top_pct : (float) percentage to choose from best labels
    top_n : (int) number to choose from best labels
    top_thresh : (float) error threshold for best labels
    output
    -------
    hyperparameter : (str) hyperparameter"""

    def _counts(data):
        # Generate a table of sorted (value, frequency) pairs.
        table = Counter(iter(data)).most_common()
        if not table:
            return table
        # Extract the values with the highest frequency.
        maxfreq = table[0][1]
        for i in range(1, len(table)):
            if table[i][1] != maxfreq:
                table = table[:i]
                break
        return table

    def get_top_n(labels, top_n, hyperparameter):
        # Top n
        top_n_obs = []

        if len(labels) / top_n <= 3:
            # Small Datasets
            top_n = max(3, int((len(labels)+1)*0.2))

            for key in range(1, top_n+1):
                top_n_obs.append(labels[str(key)][hyperparameter])

        else:
            for key in range(1, top_n+1):
                top_n_obs.append(labels[str(key)][hyperparameter])

        return top_n_obs

    def get_top_pct(labels, top_pct, hyperparameter):
        # 10% Percentile
        top_pct_obs = []

        top_pct_idx = max(3, int((len(labels)+1)*top_pct))
        for key in range(1, top_pct_idx+1):
            top_pct_obs.append(labels[str(key)][hyperparameter])

        return top_pct_obs

    def get_top_thresh(labels, top_thresh, hyperparameter):
        # Error Threshold
        top_thresh_obs = []

        for key, value in labels.items():
            if value['loss'] <= top_thresh:
                top_thresh_obs.append(value[hyperparameter])

        # Add top 5% in case none fall within threshold
        if len(top_thresh_obs) == 0:
            top_thresh_idx = max(3, int((len(labels)+1)*0.05))
            for key in range(1, top_thresh_idx+1):
                top_thresh_obs.append(labels[str(key)][hyperparameter])

        return top_thresh_obs

    # Calculate mode of each of percentile, threshold, and top_n
    top_pct_mode = _counts(
            get_top_pct(labels, top_pct, hyperparameter))[0][0]
    top_n_mode = _counts(get_top_n(
            labels, top_n, hyperparameter))[0][0]
    top_thresh_mode = _counts(
            get_top_thresh(labels, top_thresh, hyperparameter))[0][0]

    best_by_size = []
    best_by_size.append(top_pct_mode)
    best_by_size.append(top_n_mode)
    best_by_size.append(top_thresh_mode)

    if len(best_by_size) == 1:
        best_by_size = _counts(best_by_size)[0][0]
    else:
        return top_thresh_mode

    return best_by_size


def get_unique_labels():
    """ Retrieve all labels for later encoding
    Create a unique set of all labels on each hyperparameter
    input
    ------
    collection : a mongodb collection object
    output
    ------
    unique_labels : a dictionary containing hyperparameter fields
    and their unique labels for each field"""

    dat_file = r'..\data\unique_labels.dat'

    if not os.path.exists(dat_file):
    # If the file does not exist
        hyperparameters = {}

        # Set up connection to database
        Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                                driver_name='SQL Server Native Client 10.0',
                                database_name='Clustering')

        # Query database for unique values under hyperparameters table
        sql_bysize = """SELECT DISTINCT by_size from hyperparameter"""
        sql_clusterer = """SELECT DISTINCT clusterer from hyperparameter"""
        sql_distance = """SELECT DISTINCT distance from hyperparameter"""
        sql_reduce = """SELECT DISTINCT reduce from hyperparameter"""
        sql_ncomponents = """SELECT DISTINCT n_components from hyperparameter"""

        # Query database for unique indicies under clustering table
        sql_indicies = """SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'clustering'"""

        # Query
        bysize = Insert.core_select_execute(sql_bysize)
        bysize_vals = [x.by_size for x in bysize]
        clusterer = Insert.core_select_execute(sql_clusterer)
        clusterer_vals = [x.clusterer for x in clusterer]
        distance = Insert.core_select_execute(sql_distance)
        distance_vals = [x.distance for x in distance]
        reduce = Insert.core_select_execute(sql_reduce)
        reduce_vals = [x.reduce for x in reduce]
        ncomponents = Insert.core_select_execute(sql_ncomponents)
        ncomponents_vals = [x.n_components for x in ncomponents]
        indicies = Insert.core_select_execute(sql_indicies)
        indicies_vals = [ 'KL', 'CH', 'Hartigan', 'CCC', 'Marriot', 'TrCovW',
                         'TraceW', 'Friedman', 'Rubin', 'Cindex',
                         'DB', 'Silhouette', 'Duda', 'PseudoT2', 'Beale',
                         'Ratkowsky', 'Ball', 'PtBiserial', 'Frey', 'McClain',
                         'Dunn', 'Hubert', 'SDindex', 'Dindex', 'SDbw',
                         'gap_tib', 'gap_star', 'gap_max', 'Scott']


        # Construct a dicitonary of possible hyperparameters
        hyperparameters['by_size'] = bysize_vals
        hyperparameters['clusterer'] = clusterer_vals
        hyperparameters['reduce'] = reduce_vals
        hyperparameters['index'] = indicies_vals
        hyperparameters['n_components'] = ncomponents_vals

        with open(dat_file, 'wb') as f:
            pickle.dump(hyperparameters, f)

    else:
        with open(dat_file, 'rb') as f:
            hyperparameters = pickle.load(f)

    return hyperparameters


def save_unique_labels(unique_labels):
    """Save all labels to text files for use in tensorflow
    inputs
    -------
    unique_labels : (dict) with keys [by_size, clusterer, index, n_components,
                                      reduce]"""

    assert isinstance(unique_values, dict), 'unique_labels must be dictionary'

    file_name_bysize = r'../data/vocab_bysize.txt'
    file_name_clusterer = r'../data/vocab_clusterer.txt'
    file_name_index = r'../data/vocab_index.txt'
    file_name_n_components = r'../data/vocab_n_components.txt'
    file_name_reduce = r'../data/vocab_reduce.txt'
    file_name_all = r'../data/vocab_all.txt'

    vocab_all = []
    for key, value in unique_labels.items():
        for vocab in value: # value is list
            string_vocab.append(str(vocab))

    with open(file_name_all, 'w') as f:
        for vocab in vocab_all:
            f.write(vocab)
            f.write('\n')

    with open(file_name_bysize, 'w') as f:
        for vocab in unique_labels['by_size']:
            f.write(str(vocab))
            f.write('\n')

    with open(file_name_clusterer, 'w') as f:
        for vocab in unique_labels['clusterer']:
            f.write(str(vocab))
            f.write('\n')

    with open(file_name_index, 'w') as f:
        for vocab in unique_labels['index']:
            f.write(str(vocab))
            f.write('\n')

    with open(file_name_n_components, 'w') as f:
        for vocab in unique_labels['n_components']:
            f.write(str(vocab))
            f.write('\n')

    with open(file_name_reduce, 'w') as f:
        for vocab in unique_labels['reduce']:
            f.write(str(vocab))
            f.write('\n')

    return None







