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
import matplotlib.pyplot as plt
from gap_statistic import OptimalK


#%% Iterate and track accuracy

class AccuracyTest():
    
    def __init__(self):
        #Load r'D:\Z - Saved SQL Databases\error_df.csv' into DF if not already
        #If file doesnt exist create file
        self.correct_k_dict_path = r"D:\Z - Saved SQL Databases\correct_k_dict.csv"
        self.correct_k_dict = pd.read_csv(self.correct_k_dict_path, index_col=0, header='infer')
        self.error_df_path = r'D:\Z - Saved SQL Databases\error_df.csv'
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
    
    def get_correct_k(self, db_name, dataframe, manual=False):
        """Finds the correct k based on database name from a csv file located
        in the local hard-disc.
        parameters
        -------
        db_name : name of the database
        manual : Display points for manual entry of correct K. True if you
        want the user to manually enter in K.
        dataframe : dataframe of current sequence (pass df_clean)"""
        
        try:
            col = np.where(dataframe.columns == 'DBPath')[0][0]
            row = np.where(self.correct_k_dict['DBPath'] == db_name)[0][0]
            correct_k = self.correct_k_dict.iloc[row, 1]
            
        except:
            
            if not manual:
                raise NoValueFound('The correct K value is not known')
                
            else:   
                _excel_path = ".\\user_input.csv"
                dataframe.sort_values(by=['NAME']).to_csv(_excel_path)
                os.startfile(_excel_path)
                
                num_k_user = int(input('User Defined Number of Clusters : '))
                new_pts = pd.DataFrame({'DBPath':[dataframe.iloc[0, col]],
                                        'Correct Number Clusters':[num_k_user]})
                
                with open(self.correct_k_dict_path, 'a') as f:
                    new_pts.to_csv(f, header=False)
                    
                self.correct_k_dict = self.correct_k_dict.append(new_pts)
                row = np.where(self.correct_k_dict['DBPath'] == db_name)[0][0]
                correct_k = self.correct_k_dict.iloc[row, 1]
            
        return correct_k
    
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
        if correct_k >= 500:
            raise ClusterTooHigh()
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
    
    def iterate(self, iterator, manual=True, plot=False):
        """Iterate through a database and report optimalK for each sequence in
        the database. 
        parameters
        -------
        iterator : an iterator on your database
        manual : True to manually input number of correct clusters
        plot : plot the current datasets gap, gap*, and MDS sets (False to skip)
        output
        -------
        error_df : dataframe containing optimalK predicted and actual K"""
        global database, df_clean, error_df, new_vals, count_df, new_vals, X_partial
        global indicies
        myDBPipe = JVDBPipe()
        sequence_tag = 'DBPath'
        
        for _, database in iterator:

            col = np.where(database.columns == sequence_tag)[0][0]
            db_name = database.iloc[0, col]
            print('\n## {}\n'.format(db_name))
            if db_name in self.error_df[sequence_tag].values:
                continue
            
            df_clean = myDBPipe.cleaning_pipeline(database, 
                                                  remove_dupe=False, 
                                                  replace_numbers=False, 
                                                  remove_virtual=True)
    
            df_text = myDBPipe.text_pipeline(df_clean, vocab_size='all')
            #_vocabulary = df_text.columns.tolist()
            X = df_text.values
            
            try:
                correct_k = self.get_correct_k(db_name, df_clean, manual=manual)
            except NoValueFound:
                print('Database {} skipped, no known number of clusters.  \
                      Set manual = True to avoid this error'.format(db_name))
                continue
            
            words_bool = X > 0
            lengths = np.sum(words_bool, axis=1)
            unique_lengths = list(set(lengths))
            
            #Empty, sum results at saving
            optK_MDS_gap_max = []
            optK_MDS_gap_Tib = []
            optK_MDS_gapStar_max = []
            
            optK_X_gap_max = []
            optK_X_gap_Tib = []
            optK_X_gapStar_max = []
            
            #Iterate over each word size
            for length in unique_lengths: 
                indicies = np.where(lengths == length)
                X_partial = X[indicies]
                print('Trying word count : {}'.format(length))
                if X_partial.shape[0] <= 1: #Skip 1-length words, not useful
                    continue
                
                mds = MDS(n_components = 2)
                X_reduced_mds = mds.fit_transform(X_partial)
                #Change max_clusters for correct_k
                max_clusters = self.get_max_iterations(X, X_partial, correct_k)
                
                optimalkMDS = OptimalK(parallel_backend='multiprocessing')
                num_k_gap1_MDS = optimalkMDS(X_reduced_mds, cluster_array=np.arange(1,max_clusters,1))
                gapdf1_MDS = optimalkMDS.gap_df
                optimalkX = OptimalK(parallel_backend='multiprocessing')
                num_k_gap1_X = optimalkX(X_partial.astype(np.float32), cluster_array=np.arange(1,max_clusters,1))
                gapdf1_X = optimalkX.gap_df
                
                #Maybe if other one is too slow?
        #        num_k_gap2_X, gapdf2_X = myClustering.optimalK2(X, nrefs=5, maxClusters=_max_clusters)
        #        num_k_gap2_MDS, gapdf2_MDS = myClustering.optimalK2(X_reduced_mds, nrefs=5, maxClusters=_max_clusters)
                
                #Different methods
                optK_MDS_gap_max.append(num_k_gap1_MDS)
                optK_MDS_gap_Tib.append(self.tib_optimal_k(gapdf1_MDS['n_clusters'], 
                                                 gapdf1_MDS['gap_value'], 
                                                 gapdf1_MDS['ref_dispersion_std']))
                optK_MDS_gapStar_max.append(gapdf1_MDS.iloc[np.argmax(gapdf1_MDS['gap*'].values)].n_clusters)
                
                optK_X_gap_max.append(num_k_gap1_X)
                optK_X_gap_Tib.append(self.tib_optimal_k(gapdf1_X['n_clusters'], 
                                               gapdf1_X['gap_value'], 
                                               gapdf1_X['ref_dispersion_std']))
                optK_X_gapStar_max.append(gapdf1_X.iloc[np.argmax(gapdf1_X['gap*'].values)].n_clusters)
                
            new_vals = pd.DataFrame({sequence_tag:db_name,
                     'correct_k':[correct_k],
                     'optk_MDS_gap_max':[sum(optK_MDS_gap_max)],
                     'optk_MDS_gap_Tib':[sum(optK_MDS_gap_Tib)],
                     'optk_MDS_gap*_max':[sum(optK_MDS_gapStar_max)],
                     'optk_X_gap_max':[sum(optK_X_gap_max)],
                     'optk_X_gap_Tib':[sum(optK_X_gap_Tib)],
                     'optk_X_gap*_max':[sum(optK_X_gapStar_max)],
                     'n_points':[X.shape[0]]
                     })
            
            #Make a dictionary that can be saved in dataframe
            count_dict = self.get_word_dictionary(X)
            old_keys = list(count_dict.keys())
            new_keys = ['n_len' + str(old_key) for old_key in old_keys]
            for old_key, new_key in zip(old_keys, new_keys):
                count_dict[new_key] = count_dict.pop(old_key)
            
            count_dict[sequence_tag] = db_name
            count_df = pd.DataFrame(count_dict, index=[0])
            new_vals = new_vals.join(count_df.set_index(sequence_tag), on=sequence_tag)
            
            self.error_df = self.error_df.append(new_vals)
            error_df = self.error_df
        
            if plot:
                self.plot_gaps(gapdf1_X, 
                               gapdf1_MDS, 
                               optK_X_gap_max, 
                               optK_MDS_gap_max, 
                               correct_k)
                self.plot_reduced_dataset(X_reduced_mds)
                input('Press <ENTER> to continue loop. Suggested close the figurs')
            
            #Save error DF for later reading
            with open(self.error_df_path, 'a') as f:
                new_vals.to_csv(f, header=False)
            
        return self.error_df
    
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
        #plt_gap(gapdf2_MDS['n_clusters'], gapdf2_MDS['gap_value'], num_k_gap2_MDS, ax, label='Gap2_MDS')
        #plt_gap(gapdf2_X['n_clusters'], gapdf2_X['gap_value'], num_k_gap2_X, ax, label='Gap2_X')
        
        gap_fig = plt.figure(3)
        ax = gap_fig.subplots(1,1)
        self.plt_gap(gapdf1_MDS['n_clusters'], gapdf1_MDS['gap*'], k_MDS, ax, label='Gap*1_MDS', correct_k=correct_k)
        self.plt_gap(gapdf1_X['n_clusters'], gapdf1_X['gap*'], k_X, ax, label='Gap*1_X')
        #plt_gap(gapdf2_MDS['n_clusters'], gapdf2_MDS['gap*'], num_k_gap2_MDS, ax, label='Gap*2_MDS')
        #plt_gap(gapdf2_X['n_clusters'], gapdf2_X['gap*'], num_k_gap2_X, ax, label='Gap*2_X')
    
    def plot_reduced_dataset(self, X_reduced_mds):
        mds_fig = plt.figure(1)
        ax = mds_fig.subplots(1,1)
        self.plt_MDS(X_reduced_mds[:,0], X_reduced_mds[:,1], np.zeros(X_reduced_mds.shape[0]), ax)
    
    
class NoValueFound(Exception):
    pass
class ClusterTooHigh(Exception):
    pass



myTest = AccuracyTest()
_master_pts_db = r"D:\Z - Saved SQL Databases\master_pts_db.csv"
myClustering = JVClusterTools()
my_iter = myClustering.read_database_set(_master_pts_db)
error_df = myTest.error_df

myTest.iterate(my_iter, manual=True, plot=False)
b=myTest.error_df



    






