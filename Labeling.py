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

#Third party imports
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from collections import namedtuple
from collections import Counter
import statistics
import numpy as np
import pandas as pd
import os
import pickle

loss = namedtuple('loss',['clusters', 'l2error', 'variance', 'loss'])



class ExtractLabels():
    
    def __init__(self):
        pass

    def calc_features(self, database, pipeline, tag):
        """Calculate the features of a dataset. These will be used to 
        predict a good clustering algorithm.
        Inputs
        -------
        database : Your database. Its input does not matter as long as the pipeline
        passed outputs a numpy array. 
        pipeline : Your finalized pipeline. See sklearn.Pipeline. The output of
        the pipelines .fit_transform() method should be your encoded array
        of instances and features of size (n,p) n=#instances, p=#features. 
        Alternatively, you may pass a sequence of tuples containing names and 
        pipeline objects : [('pipe1', myPipe1()), ('pipe2',myPipe2())]
        tag : unique tag to identify each instance. The instance with feature 
        vectors will be returned as a pandas dataframe with the tag on the index.
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
                'instance':tag,
                'n_instance':n_points,
                'n_features':n_features,
                'len_var':len_var,
                'uniq_ratio':n_points/n_features,
                }
        features_dict = {**features_dict, **count_dict_pct}
        features_df = pd.DataFrame(features_dict, index=[tag])
        
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

    def calc_labels(self, records, instance_name, var_scale=0.2, error_scale=0.8):
        """Given an instance_name (database name in this case) and set of records
        output the correct labels for that database.
        inputs
        -------
        records : a list of records from the Record class
        instance_name : The column/key of the database sequence you wish to 
        return labels for. In my case, it will be a path similar to D:\[...]\JobDB.mdf
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
        non_opt_cols = ['DBPath', 'correct_k','n_points', 'n_len1', 
                        'n_len2', 'n_len3', 'n_len4',
                        'n_len5', 'n_len6', 'n_len7']
        non_opt_cols2 = ['instance','correct_k','records']
        sequence_tag = 'DBPath'
        hyper_dict = {}
        
        # Loss object holds information for each clustering indicy
        # clusters is the number of predicted clusters. It is a list to hold the predicted value for each iteration
        # l2error is the l2 error of all predictions
        # variance is variance between the predicted number fo clusters
        # loss is a custom weighted loss used to find the best prediction index
        
        losses = []
        
        # Create a unique set of hyper_dict
        # The hyper_dict stores all unique combinations of hyperparameters
        # Used on a certain dataset
        for idx, record in enumerate(records): #Dictionary of unique dictionaries
            try:
                hyper_dict[self.dict2str(record.hyper_dict)]['records'].append(record)
            except KeyError:
                hyper_dict[self.dict2str(record.hyper_dict)] = {'records':[record]}

        # Keep track of each opt_k column in the error_dataframe
        # Create a loss object which keeps track of each instances error on each
        # opt_k. opt_k is a optimum metric returned from any of the cluustering 
        # Algorithms
        for hyper_set, subdict in hyper_dict.items():
            #Keep track of the instance of the user
            subdict['instance'] = instance_name
            
            for record in subdict['records']:
                dataframe = record.dataframe
                try:
                    # Extract the correct number of clusters
                    idx = dataframe[dataframe[sequence_tag] == instance_name].index
                    subdict['correct_k'] = dataframe.loc[idx[0], 'correct_k']
                    
                except IndexError:
                        continue
                
                
                # Add the number of clusters to the loss object
                for opt_k in set(dataframe.columns).difference(set(non_opt_cols)):
                    # Easier to access values corresponding to a database instance
                    col_name = str(opt_k)
                    
                    # Predicted number of clusters
                    clusters = dataframe.loc[idx[0], opt_k]
                    
                    # A failed/incomplete run should not be considered (aka 0)
                    if clusters == 0:
                        continue
                    
                    try:
                        _loss = loss(clusters, None, None, None)
                        hyper_dict[hyper_set][col_name].clusters.append(_loss.clusters)
                    except KeyError:
                        hyper_dict[hyper_set][col_name] = loss([_loss.clusters], None, None, None)
        
            # Calculate relevant metrics for each database on each hyperparameter set
            # Replace optimal_k calculations with errors (l2)
            for opt_k in set(subdict.keys()).difference(set(non_opt_cols2)):
                
                # Calculate the l2 norm (sum of squared error)
                abs_error = abs(np.array(subdict[opt_k].clusters) - subdict['correct_k'])
                l2norm = sum(abs_error**2)
                
                # If there is only one prediction, then the variance = l2 norm 
                # so we dont over-penalize Predictions with multiple predictions
                if len(subdict[opt_k].clusters) == 1:
                    # Variance of predictions
                    variance = l2norm 
                    
                else:
                    variance = np.var(np.array(subdict[opt_k].clusters))
                
                # Custom loss to find the best clustering index
                # basically, custom loss is a combination of the l2 norm error
                # and variance of predictions
                calc_loss = (l2norm * error_scale + variance * 
                             len(subdict[opt_k].clusters) * var_scale)
                subdict[opt_k] = loss(subdict[opt_k].clusters,
                        l2norm, 
                       variance*len(subdict[opt_k].clusters), 
                       calc_loss)
                losses.append(calc_loss)
        
        #Get ready to sort on this named typle, containing keys of nested dictionaries
        best_errors = []
        ErrorTuple = namedtuple('errors', ['hyper_key','opt_key','loss'])
        
        #Create tuples
        for hyper_set, subdict in hyper_dict.items():
            for opt_k in set(subdict.keys()).difference(set(non_opt_cols2)):
                error = ErrorTuple(hyper_set, opt_k, subdict[opt_k].loss)
                best_errors.append(error)
        #Sort on tuple objects
        best_errors = sorted(best_errors, key=lambda k: k.loss)
        
        #Return the best predicted clustering index based on loss in namedtuple loss.loss
        #Create a dictionary for returning
        
        best_labels = {}
        for i in range(1, len(best_errors)+1):
            best_labels[i] = {}
        
        for i in range(0, len(best_errors)):
            error = best_errors[i]
            
            # The "best" starts at 1, not 0
            position = i + 1
            
            by_size = bool(hyper_dict[error.hyper_key]['records'][0].hyper_dict['by_size'])
            distance = hyper_dict[error.hyper_key]['records'][0].hyper_dict['distance']
            clusterer = hyper_dict[error.hyper_key]['records'][0].hyper_dict['clusterer']
            n_components = hyper_dict[error.hyper_key]['records'][0].hyper_dict['n_components']
            reduce = hyper_dict[error.hyper_key]['records'][0].hyper_dict['reduce']
            index = error.opt_key
            loss = best_errors[i].loss
            
            # Enforce conversion to strings
            # The output should be strings for use in tensorflow conversion
            # Of features to indicator columns or embedding columns
            best_labels[position]['by_size'] = str(by_size)
            best_labels[position]['distance'] = str(distance)
            best_labels[position]['clusterer'] = str(clusterer)
            best_labels[position]['n_components'] = str(n_components)
            best_labels[position]['reduce'] = str(reduce)
            best_labels[position]['index'] = str(index)
            best_labels[position]['loss'] = str(loss)
        
        return best_labels, hyper_dict
    
    def dict2str(self, hyper_dict):
        x = str()
        for key, value in hyper_dict.items():
            x = x + str(key) + ':' + str(value) + '\n'
        return x
    

def choose_best_hyper(labels, hyperparameter, top_pct=0.1, top_n=10, top_thresh=5):
    """Choose the best hyperparameter given a set of labels from 
    JVWork_Labeling.ExtractLabels
    inputs
    ------
    labels : a dictionary of labels
    hyperparamter : name of hyperparamter in labels (must be a key in the labels
    nested dictionary)
    top_pct : percentage to choose from best labels
    top_n : number to choose from best labels
    top_thresh : error threshold for best labels
    output
    -------
    hyperparameter : hyperparameter"""
    
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


def get_unique_labels(collection=None):
    """ Retrieve all labels for later encoding
    Create a unique set of all labels on each hyperparameter
    input
    ------
    collection : a mongodb collection object
    output
    ------
    unique_labels : a dictionary containing hyperparameter fields 
    and their unique labels for each field"""
    
    
    dat_file = r'.\data\unique_labels.dat'
    
    if not os.path.exists(dat_file):
    # If the file does not exist
        hyperparameters = {}
        
        for document in collection.find():
            
            best_hyper = document['best_hyper']
            for key, value in best_hyper.items():
                
                if isinstance(value, dict):
                    try:
                        new_set = set(list(value.values()))
                        hyperparameters[key].update(new_set)
        
                    except KeyError:
                        new_set = set(list(value.values()))
                        hyperparameters[key] = new_set
                        
                elif isinstance(value, list): # Not dictionary
                    try:
                        new_set = set(value)
                        hyperparameters[key].update(new_set)
                        
                    except KeyError:
                        new_set = set(value)
                        hyperparameters[key] = new_set
                        
                elif isinstance(value, str):
                    try:
                        new_set = set([value])
                        hyperparameters[key].update(new_set)
                        
                    except KeyError:
                        new_set = set([value])
                        hyperparameters[key] = new_set
                        
                else:
                    try:
                        new_set = set([value])
                        hyperparameters[key].update(new_set)
                        
                    except KeyError:
                        new_set = set([value])
                        hyperparameters[key] = new_set
                        
        for key, value in hyperparameters.items():
            hyperparameters[key] = list(value)   
            
        with open(dat_file, 'wb') as f:
            pickle.dump(hyperparameters, f)
            
    else:
        with open(dat_file, 'rb') as f:
            hyperparameters = pickle.load(f)
            
    return hyperparameters





















