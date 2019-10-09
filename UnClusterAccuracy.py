# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:39:41 2019

@author: z003vrzk
"""

from JVWork_UnsupervisedCluster import JVClusterTools
from JVWork_WholeDBPipeline import JVDBPipe

import pandas as pd
import os
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from datetime import date

import matplotlib.pyplot as plt
from gap_statistic import OptimalK
import inspect
from JVrpy2 import nbclust_calc
pd.options.mode.chained_assignment = None  # default='warn'

#%% Iterate and track accuracy

class AccuracyTest():
    
    def __init__(self):
        #Load r'D:\Z - Saved SQL Databases\error_df.csv' into DF if not already
        #If file doesnt exist create file
        self.correct_k_dict_path = r".\\correct_k_dict.csv"
        self.correct_k_dict = pd.read_csv(self.correct_k_dict_path, index_col=0, header='infer')
        self.error_df_path = r'error_dfs\error_df.csv'
        self.sequence_tag = 'DBPath'
        if not os.path.isfile(self.error_df_path):
            self.error_df = pd.DataFrame(
                    columns=['DBPath','correct_k', 
                             'optk_MDS_gap_max', 'optk_MDS_gap_Tib',
                             'optk_MDS_gap*_max', 'optk_X_gap_max', 'optk_X_gap_Tib',
                             'optk_X_gap*_max', 
                             'n_points', 'n_len1','n_len2','n_len3','n_len4',
                             'n_len5', 'n_len6', 'n_len7'
                             ]
                    )
            self.error_df.to_csv(self.error_df_path)
        else:
            self.error_df = pd.read_csv(self.error_df_path, index_col=0)
        
        copy_cols = [self.sequence_tag,'correct_k','n_points', 'n_len1',
                     'n_len2','n_len3','n_len4','n_len5', 'n_len6', 'n_len7']
        self.error_df_copy = self.error_df[copy_cols]
    
    def get_correct_k(self, db_name, dataframe, manual=False, skip=False):
        """Finds the correct k based on database name from a csv file located
        in the local hard-disc.
        parameters
        -------
        db_name : name of the database
        dataframe : dataframe of current sequence (pass df_clean)
        manual : Display points for manual entry of correct K. True if you
        want the user to manually enter in K.
        skip : to skip the database set to True, else user will manually enter
        database correct_k or program will try to infer it"""
        sequence_tag = self.sequence_tag
        
        try:
            row = np.where(self.correct_k_dict[sequence_tag] == db_name)[0][0]
            correct_k = self.correct_k_dict.iloc[row, 1]
            
        except KeyError:
            
            if all([not(manual), not(skip)]): #False, False
                raise UserGuessException('Guess value for {}'.format(db_name))
            elif all([not(manual), skip]): #False, True
                raise UserSkipException('Skip {}'.format(db_name))
                
            else:
                _excel_path = ".\\user_input.csv"
                dataframe.sort_values(by=['NAME']).to_csv(_excel_path)
                os.startfile(_excel_path)
                
                num_k_user = int(input('User Defined Number of Clusters : '))
                new_pts = pd.DataFrame({'DBPath':[db_name],
                                        'Correct Number Clusters':[num_k_user]})
                
                new_pts.to_csv(self.correct_k_dict_path, mode='a', header=False)
                    
                self.correct_k_dict = self.correct_k_dict.append(new_pts)
                row = np.where(self.correct_k_dict['DBPath'] == db_name)[0][0]
                correct_k = self.correct_k_dict.iloc[row, 1]
            
        return correct_k
    
    def get_save_path(self):
        """Return location for saving a) hyper-dict and b) calculated values
        output
        -------
        (save_path_hyper, save_path_values)
        save_path_hyper : hyperparameter file name
        save_path_valeus : values file name"""
        mydate = date.today().strftime('%m-%d')
        base_dir = r'error_dfs\\'
        idx = 1
        save_path_values = 'error_df' + ' ' + mydate + ' ' + f'({idx})'
        
        def check_files(idx, save_path_values):
            if (save_path_values+'.csv' in os.listdir(base_dir)):
                return False
            else:
                return True
        
        def increment_file(idx, save_path_values):
            idx += 1
            save_path_values = 'error_df' + ' ' + mydate + ' ' + f'({idx})'
            return save_path_values
        
        while not check_files(idx, save_path_values):
            save_path_values = increment_file(idx, save_path_values)
        
        save_path_hyper = base_dir + save_path_values + ' ' + 'hyper.csv'
        save_path_values = base_dir + save_path_values + '.csv'
        return (save_path_hyper, save_path_values)
            
    
    @staticmethod
    def tib_optimal_k(n_clusters, gap, sk):
        """Finds the optimal K value based on Tibshirani's original paper on 
        the gap statistic. Yield the smallest k such that gap(k) >= gap(k+1) - s_(k+1).
        parameters
        -------
        n_clusters : array or Series of cluster counts (k)
        gap : array or Series of gap value at each n_clusters
        sk : array or Series of standard errors
        output
        -------
        optimalK
        optimalK : found using Tibshirani's method"""
        if type(n_clusters) == pd.Series:
            n_clusters = n_clusters.values
        if type(gap) == pd.Series:
            gap = gap.values
        if type(sk) == pd.Series:
            sk = sk.values
        
        diffs = gap[:-1] - gap[1:] + sk[:-1]
        
        
        for idx, dif in enumerate(diffs):
            if dif >=0:
                return n_clusters[idx]
        
        print('No value found that satisfies criteria')
        max_idx = np.argmax(gap)
        
        return n_clusters[max_idx]
    
    @staticmethod
    def get_word_dictionary(word_array):
    
        count_dict = {}
        
        for row in word_array:
            count = sum(row>0)
            try:
                count_dict[count] += 1
            except KeyError:
                count_dict[count] = 1
                
        return count_dict
    
    def get_max_iterations(self, X, X_partial, correct_k):
        """Return the max number of k_clusters to try in optimalK clustering
        parameters
        -------
        X : whole text bagged dataset
        X_partial : partial bagged dataset, based on word size
        correct_k : correct clusters for whole dataset"""
        #Set the upper bound
#        max_clusters = min(correct_k*(X_partial.shape[0]/X.shape[0])+5, 
#                           X_partial.shape[0])
#        if correct_k >= 500:
#            raise ClusterTooHigh()
        if correct_k >= 50: #Handle large datasets
            max_clusters = min(correct_k*(X_partial.shape[0]/X.shape[0])+9,
                               X_partial.shape[0])
        else: #Smaller datasets
            max_clusters = min(correct_k+9,
                               X_partial.shape[0])
        #Set the lower bound
        max_clusters = max(max_clusters, 
                           6)
        if max_clusters >= X_partial.shape[0]:
            max_clusters = X_partial.shape[0]
        return int(max_clusters)
    
    def get_count_df(self, X, index, db_name):
        """Make a dictionary that can be saved in dataframe
        parameters
        -------
        X : one-hot encoded text dataframe
        index : index to use on the returned dataframe. Included because the 
        returned dataframe is merged on another dataframe with a common index
        db_name : name of database havign count_dict generated"""
        
        sequence_tag = self.sequence_tag

        count_dict = self.get_word_dictionary(X)
        old_keys = list(count_dict.keys())
        new_keys = ['n_len' + str(old_key) for old_key in old_keys]
        
        for old_key, new_key in zip(old_keys, new_keys):
            count_dict[new_key] = count_dict.pop(old_key)
            
        required_keys = ['n_len1','n_len2','n_len3',
                         'n_len4','n_len5','n_len6','n_len7']
        for key in required_keys:
            count_dict.setdefault(key, 0)
        
        count_dict[sequence_tag] = db_name
        count_dict['n_points'] = X.shape[0]
        count_df = pd.DataFrame(count_dict, index=index)
        
        return count_df
        
    def recalc_count_df(self, iterator):
        """Recalculate the n_len1,[...],n_len7 values in the error_df 
        because of errors.
        parameters
        -------
        iterator : iterator : an iterator on your database. 
        See myClustering.read_database_set() 
        recalc : True if you want to recalculate error_df and update 
        the existing values for n_len1, [...], n_len7"""
        
        sequence_tag = self.sequence_tag
        myDBPipe = JVDBPipe()
        assert inspect.isgenerator(iterator),'iterable must not be None to recalc'
        for _, database in iterator:
            
            col = np.where(database.columns == sequence_tag)[0][0]
            db_name = database.iloc[0, col]
            
            if db_name in self.error_df[sequence_tag].values:
                
                print('\nUpdating : {}\n'.format(db_name))
                index = self.error_df.index[self.error_df[sequence_tag] == db_name].tolist()
                assert len(index)==1,'Multiple sequence_tag instances in self.error_df'
                
                df_clean = myDBPipe.cleaning_pipeline(database, 
                                                      remove_dupe=False, 
                                                      replace_numbers=False, 
                                                      remove_virtual=True)
                df_text = myDBPipe.text_pipeline(df_clean, vocab_size='all')
                X = df_text.values
                
                count_df = self.get_count_df(X, index, db_name)
                self.error_df.update(count_df)
            else:
                pass

        self.error_df.to_csv(self.error_df_path)
        return
    
    def iterate(self, database, manual=True, skip=False, by_size=True, 
                method='MDS', n_components=2, plot=False):
        """Iterate through a database and report optimalK for each sequence in
        the database. 
        parameters
        -------
        database : a database. See myClustering.read_database_set()
        manual : True to manually input number of correct clusters
        skip : skip the database if correct_k is not known and manual is False. 
        If manual and skip are False, algorithm will try to guess correct_k
        method : Dimensionality reduction technique ('MDS' or 'TSNE'). Default MDS
        n_components : number of dimensions reduced
        plot : plot the current datasets gap, gap*, and MDS sets (False to skip)
        output
        -------
        error_df : dataframe containing optimalK predicted and actual K"""
        myDBPipe = JVDBPipe()
        sequence_tag = self.sequence_tag

        col = np.where(database.columns == sequence_tag)[0][0]
        db_name = database.iloc[0, col]
        print('\n{}'.format(db_name))
        if db_name in self.error_df[sequence_tag].values:
            print('Database already calculated. Use iterate_recalc() to calculate again')
            return
        
        clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False, 
                                              replace_numbers=False, 
                                              remove_virtual=True)
        df_clean = clean_pipe.fit_transform(database)
        text_pipe = myDBPipe.text_pipeline(vocab_size='all',
                                           seperator='.')
        X = text_pipe.fit_transform(df_clean['NAME']).toarray()
#        _word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
#        df_text = pd.DataFrame(X, columns=_word_vocab)
        
        try:
            correct_k = self.get_correct_k(db_name, df_clean, 
                                           manual=manual, skip=skip)
        except UserSkipException:
            print('Database {} skipped, no known number of clusters.  \
                  Set manual = True to avoid this'.format(db_name))
            return
        except UserGuessException:
            correct_k = 60 #TODO, guess based on database size
            print('Database {} guess value of k'.format(db_name))
            
        index = [max(self.error_df.index)+1]
        
        #Do the clustering, returns the results of clustering
        new_vals_standard = self._optimalk_calc(X, 
                                                correct_k, 
                                                db_name, 
                                                by_size=by_size)
        new_vals_reduced = self._optimalk_calc_reduce(X, 
                                                      correct_k, 
                                                      db_name, 
                                                      by_size=by_size, 
                                                      method=method, 
                                                      n_components=n_components)
        
        new_vals = {**new_vals_standard, **new_vals_reduced}
        delete_keys = ['n_points','correct_k']
        for delete_key in delete_keys:
                new_vals.pop(delete_key)
        new_vals = pd.DataFrame(new_vals, 
            index=index)
        
        count_df = self.get_count_df(X, index, db_name)
        new_vals = new_vals.join(count_df.set_index(sequence_tag), on=sequence_tag)
        
        self.error_df = self.error_df.append(new_vals)
        
        #Save error DF for later reading
        new_vals.to_csv(self.error_df_path, mode='a', header=False)

        return self.error_df
    
    def iterate_recalc(self, database, by_size=True, standard=True, 
                       reduce=True, method='MDS', n_components=2,
                       nbclust=True, index_nb='alllong', clusterer='kmeans', 
                       distance='euclidean'):
        """Iterate through a database and report optimalK for each sequence in
        the database. 
        parameters
        -------
        database : a database. See myClustering.read_database_set()
        by_size : (True/False) cluster by word size (True), or the whole database
        at once (False)
        standard : True to calculate optimalK with standard dataset
        reduce : True to calculate optimalK with reduced dataset
        method : Dimensionality reduction technique ('MDS' or 'TSNE'). Default MDS
        n_components : number of dimensions reduced output
        nbclust : Calculate optimalk with NbClust R Package
        index_nb : index to use with NbClust. See 
        https://www.rdocumentation.org/packages/NbClust/versions/3.0/topics/NbClust
        clusterer : clusterer to use with NbClsut
        distance : distance metric for NbClust
        -------
        error_df : dataframe containing optimalK predicted and actual K"""
        myDBPipe = JVDBPipe()
        sequence_tag = self.sequence_tag

        col = np.where(database.columns == sequence_tag)[0][0]
        db_name = database.iloc[0, col]
        print('\n{}'.format(db_name))
        if not db_name in self.error_df[sequence_tag].values: #NOT
            print('Database : \n{}\n has not undergone initial calculation. Use\
                  self.iterate(). Database skipped reacalculation'.format(db_name))
            return None
        
        clean_pipe = myDBPipe.cleaning_pipeline(remove_dupe=False, 
                                              replace_numbers=False, 
                                              remove_virtual=True)
        df_clean = clean_pipe.fit_transform(database)
        text_pipe = myDBPipe.text_pipeline(attributes='NAME', 
                                           vocab_size='all',
                                           seperator='.')
        X = text_pipe.fit_transform(df_clean).toarray()
        
#        try:
        correct_k = self.get_correct_k(db_name, df_clean, manual=True, skip=False)
#        except UserSkipException:
#            print('Database {} skipped, no known number of clusters.  \
#                  Set manual = True to avoid this'.format(db_name))
#            return self.error_df_copy
#        except UserGuessException:
#            correct_k = 60 #TODO, guess based on database size
#            print('Database {} guess value of k'.format(db_name))
            
        index = self.error_df.index[self.error_df[sequence_tag] == db_name].tolist()
        
        #Do the clustering, returns the results of clustering
        new_vals_standard = {}
        new_vals_reduced = {}
        new_vals_nb = {}
        if standard:
            print('Standard')
            new_vals_standard = self._optimalk_calc(X,
                                                    correct_k, 
                                                    db_name, 
                                                    by_size=by_size)
        if reduce:
            print('Reduce')
            new_vals_reduced = self._optimalk_calc_reduce(X, 
                                                          correct_k, 
                                                          db_name, 
                                                          by_size=by_size, 
                                                          method=method, 
                                                          n_components=n_components)
        if nbclust:
            print('Nb')
            new_vals_nb = self._nbclust_calc(X, 
                                                correct_k, 
                                                db_name, 
                                                method=method, 
                                                n_components=n_components, 
                                                index=index_nb, 
                                                clusterer=clusterer, 
                                                distance=distance,
                                                by_size=False)

        delete_keys = ['n_points','correct_k']
        for delete_key in delete_keys:
            if standard:
                try:
                    new_vals_standard.pop(delete_key)
                except KeyError:
                    pass
            if reduce:
                try:
                    new_vals_reduced.pop(delete_key)
                except KeyError:
                    pass
            if nbclust:
                try:
                    new_vals_nb.pop(delete_key)
                except KeyError:
                    pass
        
        new_vals = {**new_vals_standard, **new_vals_reduced, **new_vals_nb}
        new_vals = pd.DataFrame(new_vals, 
            index=index)

        for key in new_vals.columns:
            if not key in self.error_df_copy.columns:
                loc=len(self.error_df_copy.columns)
                new_col=key
                value=np.zeros(self.error_df_copy.shape[0])
                self.error_df_copy.insert(loc=loc, column=new_col, value=value)

        self.error_df_copy.update(new_vals) #Should keep memory

        return self.error_df_copy
    
    def _optimalk_calc(self, X, correct_k, db_name, by_size=True):
        """Returns the calculated number of clusters with different methods
        parameters
        -------
        X : word encoded array
        correct_k : correct number of clusters
        db_name : name of database/index/sequence
        by_size : (True/False) cluster by word size (True), or the whole database
        at once (False)
        output
        -------
        new_vals (dictionary) : A dictionary that can be used to construct a dataframe.
        Contains the keys 
            {sequence_tag:db_name, #database name
             'correct_k':[correct_k], #Correct number of clusters
             'optk_X_gap_max':[sum(optK_X_gap_max)], #OptimalK from normal set w/ max result
             'optk_X_gap_Tib':[sum(optK_X_gap_Tib)], #OptimalK from normal set w/ Tib result
             'optk_X_gap*_max':[sum(optK_X_gapStar_max)], #OptimalK from normal set w/ gap* max result
             'n_points':[X.shape[0]] #Total number of instances
             }"""
        sequence_tag = self.sequence_tag
        global max_clusters, X_partial, global_indicies
        #Empty, sum results at saving
        
        optK_X_gap_max = []
        optK_X_gap_Tib = []
        optK_X_gapStar_max = []
        
        words_bool = X > 0
        lengths = np.sum(words_bool, axis=1)
        unique_lengths = list(set(lengths))
        
        global_indicies = []
        
        if by_size:
            for length in unique_lengths:
                indicies = np.where(lengths == length)
                global_indicies.append(indicies)
        else:
            indicies = np.arange(0,X.shape[0],1)
            global_indicies.append(indicies)
        
        for indicies in global_indicies: #Iterate over each word size
            X_partial = X[indicies]
            
            #Change max_clusters for correct_k
            max_clusters = self.get_max_iterations(X, X_partial, correct_k)
            
            optimalkX = OptimalK(parallel_backend='multiprocessing')
            num_k_gap1_X = optimalkX(X_partial.astype(np.float32), cluster_array=np.arange(1,max_clusters+1,1))
            gapdf1_X = optimalkX.gap_df
            
            #Different methods
            optK_X_gap_max.append(num_k_gap1_X)
            optK_X_gap_Tib.append(self.tib_optimal_k(gapdf1_X['n_clusters'], 
                                           gapdf1_X['gap_value'], 
                                           gapdf1_X['ref_dispersion_std']))
            optK_X_gapStar_max.append(gapdf1_X.iloc[np.argmax(gapdf1_X['gap*'].values)].n_clusters)
            
        new_vals = {sequence_tag:db_name,
             'correct_k':[correct_k],
             'optk_X_gap_max':[sum(optK_X_gap_max)],
             'optk_X_gap_Tib':[sum(optK_X_gap_Tib)],
             'optk_X_gap*_max':[sum(optK_X_gapStar_max)],
             'n_points':[X.shape[0]]
             }
        
        return new_vals
    
    def _optimalk_calc_reduce(self, X, correct_k, db_name, 
                              distance='euclidean', clusterer='kmeans',
                              by_size=True, method='MDS', n_components=2):
        """Returns the calculated number of clusters with different methods
        parameters
        -------
        X : word encoded array
        correct_k : correct number of clusters
        db_name : name of database/index/sequence
        distance : distance metreic for NbClust. See NbClust documents. 
        default 'euclidean'
        clusterer : cluster algorithm for NbClust. See NbClust documents, 
        default 'kmeans'
        by_size : (True/False) cluster by word size (True), or the whole database
        at once (False)
        method : Dimensionality reduction technique ('MDS' or 'TSNE'). Default MDS
        n_components : number of dimensions reduced
        output
        -------
        new_vals (dictionary) : A dictionary that can be used to construct a dataframe.
        Contains the keys 
            {sequence_tag:db_name, #database name
             'correct_k':[correct_k], #Correct number of clusters
             'optk_MDS_gap_max':[sum(optK_MDS_gap_max)], #OptimalK from MDS set w/ max result
             'optk_MDS_gap_Tib':[sum(optK_MDS_gap_Tib)], #OptimalK from MDS set w/ Tib result
             'optk_MDS_gap*_max':[sum(optK_MDS_gapStar_max)], #OptimalK from MDS set w/ gap* max result
             'optk_X_gap_max':[sum(optK_X_gap_max)], #OptimalK from normal set w/ max result
             'optk_X_gap_Tib':[sum(optK_X_gap_Tib)], #OptimalK from normal set w/ Tib result
             'optk_X_gap*_max':[sum(optK_X_gapStar_max)], #OptimalK from normal set w/ gap* max result
             'n_points':[X.shape[0]] #Total number of instances
             }"""
        sequence_tag = self.sequence_tag
        global max_clusters, X_partial, global_indicies
        #Empty, sum results at saving
        optK_MDS_gap_max = []
        optK_MDS_gap_Tib = []
        optK_MDS_gapStar_max = []
        
        words_bool = X > 0
        lengths = np.sum(words_bool, axis=1)
        unique_lengths = list(set(lengths))
        
        global_indicies = []
        
        if by_size:
            for length in unique_lengths:
                indicies = np.where(lengths == length)
                global_indicies.append(indicies)
        else:
            indicies = np.arange(0,X.shape[0],1)
            global_indicies.append(indicies)
        
        for indicies in global_indicies: #Iterate over each word size
            X_partial = X[indicies]
            
            if method=='MDS':
                mds = MDS(n_components = n_components)
                X_reduced = mds.fit_transform(X_partial)
            elif method=='TSNE':
                params = {'method':'barnes_hut', 
                          'n_components':n_components,
                          'metric':'euclidean',
                          'perplexity':12}
                if n_components >= 4:
                    params['method'] = 'exact' #Slower
                tsne = TSNE(method=params['method'],n_components=params['n_components'],
                            metric=params['metric'],perplexity=params['perplexity'])
                X_reduced = tsne.fit_transform(X_partial)

            #Change max_clusters for correct_k
            max_clusters = self.get_max_iterations(X, X_partial, correct_k)
            
            optimalkMDS = OptimalK(parallel_backend='multiprocessing')
            num_k_gap1_MDS = optimalkMDS(X_reduced, cluster_array=np.arange(1,max_clusters+1,1))
            gapdf1_MDS = optimalkMDS.gap_df
            
            #Different methods
            optK_MDS_gap_max.append(num_k_gap1_MDS)
            optK_MDS_gap_Tib.append(self.tib_optimal_k(gapdf1_MDS['n_clusters'], 
                                             gapdf1_MDS['gap_value'], 
                                             gapdf1_MDS['ref_dispersion_std']))
            optK_MDS_gapStar_max.append(gapdf1_MDS.iloc[np.argmax(gapdf1_MDS['gap*'].values)].n_clusters)
            
            gap_max_str = 'optk_'+str(method)+'_gap_max'
            gap_tib_str = 'optk_'+str(method)+'_gap_Tib'
            gap_max_star_str = 'optk_'+str(method)+'_gap*_max'
            
        new_vals = {sequence_tag:db_name,
             'correct_k':[correct_k],
             gap_max_str:[sum(optK_MDS_gap_max)],
             gap_tib_str:[sum(optK_MDS_gap_Tib)],
             gap_max_star_str:[sum(optK_MDS_gapStar_max)],
             'n_points':[X.shape[0]]
             }
        
        return new_vals
    
    def _nbclust_calc(self, X, correct_k, db_name, 
                      method='MDS', n_components=3, 
                      index='all', clusterer='kmeans', distance='euclidean',
                      by_size=False):
        """Returns the calculated optimal number of clusters with the NbClust
        R Package.
        parameters
        -------
        X : word encoded array (n,p) n=# instances, p=#features/words
        correct_k : correct number of clusters
        db_name : name of database/index/sequence
        method : dimensionality reduction method. Must be either 'TSNE' or 'MDS'
        n_components : number of dimensions of reduced dataset (p->p_reduced)
        index : Index to use with NbClust package. See NbClust documentation.
        Typical values can be 'all' or 'alllong'
        by_size : (True/False) cluster by word size (True), or the whole database
        at once (False). Recommended False with the NbClust implementation
        output
        -------
        new_vals (dictionary) : A dictionary that can be used to construct a dataframe.
        The keys returned are variable based on 'index'"""
        sequence_tag = self.sequence_tag
        
        words_bool = X > 0
        lengths = np.sum(words_bool, axis=1)
        unique_lengths = list(set(lengths))
        
        global_indicies = []
        
        if by_size:
            for length in unique_lengths:
                indicies = np.where(lengths == length)
                global_indicies.append(indicies)
        else:
            indicies = np.arange(0,X.shape[0],1)
            global_indicies.append(indicies)
        
        for indicies in global_indicies: #Iterate over each word size
            X_partial = X[indicies]
            if X_partial.shape[0] <= 4:
                continue #Problem with R indexing and NbClust requirements
            
            if method=='MDS':
                mds = MDS(n_components = n_components)
                X_reduced = mds.fit_transform(X_partial)
            elif method=='TSNE':
                params = {'method':'barnes_hut', 
                          'n_components':n_components,
                          'metric':'euclidean',
                          'perplexity':12}
                if n_components >= 4:
                    params['method'] = 'exact' #Slower
                tsne = TSNE(method=params['method'],n_components=params['n_components'],
                            metric=params['metric'],perplexity=params['perplexity'])
                
#                mds = MDS(n_components = 20)
#                X_partial = mds.fit_transform(X_partial) #Preprocess to speed computation
                X_reduced = tsne.fit_transform(X_partial)
                X_reduced = X_reduced/100
                
            if X_reduced.shape[0] >= 1000:
                print('{} Skipped | size {}'.format(db_name, X_reduced.shape[0]))
                return {}
            
            max_clusters = self.get_max_iterations(X, X_partial, correct_k)
            
            #Calculate with NbClust method
            min_nc = 2 #Must be at least 2
            max_nc = max_clusters-1 #R Indexing lol
            print(f'min {min_nc} max {max_nc}')
            print('X.shape {}'.format(X_reduced.shape))
            Nb_result = nbclust_calc(X_reduced, 
                         min_nc=min_nc, max_nc=max_nc,
                         distance=distance, method=clusterer,
                         index=index)
#            df_index = Nb_result.index_df
            df_bestnc = Nb_result.best_nc_df

        new_vals = {sequence_tag:db_name,
             'correct_k':[correct_k],
             'n_points':[X.shape[0]]
             }
        try:
            for key, value in df_bestnc.to_dict().items():
                new_vals[key] = value['Number_clusters'] #value is a dictionary
        except NameError:
            pass
        return new_vals
    
    @staticmethod
    def plt_MDS(x, y, classes, artist):
        """parameters
        -------
        x : array (1D) of values
        y : array (1D) of values
        classes : array of classes for each (x,y)"""
    
        uniques = list(set(classes))
        colors = [np.array(plt.cm.viridis(i/float(len(uniques)))).reshape(1,-1) for i in range(len(uniques)+1)]
        for idx, uniq in enumerate(uniques):
            xi = [x[j] for j in range(len(x)) if classes[j] == uniq]
            yi = [y[j] for j in range(len(x)) if classes[j] == uniq]
            artist.scatter(xi, yi, c=colors[idx], label=str(uniq))
            
        artist.set_title('MDS Reduction')
        artist.legend()
        artist.set_xlabel('$z_1$')
        artist.set_ylabel('$z_2$')
        artist.grid(True)
    
    @staticmethod
    def plt_gap(k_vals, gap_vals, optimal_k, artist, label='Gap1', correct_k=None):
    
        artist.plot(k_vals, gap_vals, linewidth=2, label=label)
        artist.scatter(optimal_k, gap_vals[optimal_k - 1], s=200, c='r')
        if correct_k:
            artist.axvline(x=correct_k, ymin=0.05, ymax=0.95, c='g', label='Correct k', linestyle='--')
        
        artist.grid(True)
        artist.set_xlabel('Cluster Count')
        artist.set_ylabel('Gap Values (mean(log(refDisps)) - log(origDisp))')
        artist.set_title('Gap Value v. Cluster Count')
        artist.legend()
        
    def plot_gaps(self, gapdf1_X, gapdf1_MDS, k_X, k_MDS, correct_k):
        gap_fig = plt.figure(2)
        ax = gap_fig.subplots(1,1)
        self.plt_gap(gapdf1_MDS['n_clusters'], gapdf1_MDS['gap_value'], k_MDS, ax, label='Gap1_MDS', correct_k=correct_k)
        self.plt_gap(gapdf1_X['n_clusters'], gapdf1_X['gap_value'], k_X, ax, label='Gap1_X')

        gap_fig = plt.figure(3)
        ax = gap_fig.subplots(1,1)
        self.plt_gap(gapdf1_MDS['n_clusters'], gapdf1_MDS['gap*'], k_MDS, ax, label='Gap*1_MDS', correct_k=correct_k)
        self.plt_gap(gapdf1_X['n_clusters'], gapdf1_X['gap*'], k_X, ax, label='Gap*1_X')

    def plot_reduced_dataset(self, X_reduced_mds):
        mds_fig = plt.figure(1)
        ax = mds_fig.subplots(1,1)
        self.plt_MDS(X_reduced_mds[:,0], X_reduced_mds[:,1], np.zeros(X_reduced_mds.shape[0]), ax)
    
    
class UserGuessException(Exception):
    #Database correct # clusters is not known, and we want to guess
    pass
class UserSkipException(Exception):
    #Database correct # clusters is not known, and we dont want to guess
    #AKA skip the database
    pass
class ClusterTooHigh(Exception):
    pass






