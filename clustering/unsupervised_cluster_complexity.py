# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:54:21 2019

@author: z003vrzk
"""

# Python imports
import time
import sys
import os

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
from concurrent.futures.process import BrokenProcessPool
from rpy2.rinterface import RRuntimeError

# Local imports
if __name__ == '__main__':
    # Remove the drive letter on windows
    _CWD = os.path.splitdrive(os.getcwd())[1]
    _PARTS = _CWD.split(os.sep)
    # Project dir is one level above cwd
    _PROJECT_DIR = os.path.join(os.sep, *_PARTS[:-1])
    if _PROJECT_DIR not in sys.path:
        sys.path.insert(0, _PROJECT_DIR)

    from clustering import unsupervised_cluster, nbclust_rpy2
    from transform import transform_pipeline
    from extract import extract
    from extract.SQLAlchemyDataDefinition import (Customers, Points, Netdev,
                                              ClusteringHyperparameter, Clustering,
                                              TypesCorrection)
else:
    from clustering import unsupervised_cluster, nbclust_rpy2
    from transform import transform_pipeline
    from extract import extract
    from extract.SQLAlchemyDataDefinition import (Customers, Points, Netdev,
                                              ClusteringHyperparameter, Clustering,
                                              TypesCorrection)


UnsupervisedCluster = unsupervised_cluster.UnsupervisedClusterPoints()

#%% NbClust operation complexity

# Initialize local objects
calculate_time = []
reduction_time = []

# Hyperparameters
hyperparams = {
    'by_size':False,
    'n_components':8,
    'reduce':'MDS',
    'clusterer':'ward.D',
    'distance':'euclidean',
    'index':'all'}


# Iterate through 20 databases. For each database, perform dimensinoality
# Reduction and clustering. Record calculation times for dim-reduction
# And clustering

# Instantiate local classes
Transform = transform_pipeline.Transform()
# Create 'clean' data processing pipeline
clean_pipe = Transform.cleaning_pipeline(remove_dupe=False,
                                      replace_numbers=False,
                                      remove_virtual=True)

# Create pipeline specifically for clustering text features
text_pipe = Transform.text_pipeline(vocab_size='all',
                                   attributes='NAME',
                                   seperator='.',
                                   heirarchial_weight_word_pattern=True)

# Set up connection to SQL
Insert = extract.Insert(server_name='.\\DT_SQLEXPR2008',
                        driver_name='SQL Server Native Client 10.0',
                        database_name='Clustering')

# Get a points dataframe
customer_ids = [ 44,  91,  97, 146, 104,  70,  31, 175, 164,  72,  76, 148,
                155, 109,  55,  47, 101, 156,  78,  62]

def iterate_points(customer_ids):
    for customer_id in customer_ids:
        sel = sqlalchemy.select([Points]).where(Points.customer_id.__eq__(customer_id))
        with Insert.engine.connect() as connection:
            connection = Insert.engine.connect()
            database = pd.read_sql(sel, connection)

        yield database, customer_id

points_iterator = iterate_points(customer_ids)

for database, customer_id in points_iterator:

    if database.shape[0] == 0:
        continue

    # Transform data
    df_clean = clean_pipe.fit_transform(database)
    X = text_pipe.fit_transform(df_clean).toarray()
    word_vocab = text_pipe.named_steps['WordDictToSparseTransformer'].vocabulary
    df_text = pd.DataFrame(X, columns=word_vocab)

    # Get correct k
    with Insert.engine.connect() as connection:
        sel = sqlalchemy.select([Customers]).where(Customers.id.__eq__(customer_id))
        res = connection.execute(sel).fetchone()
        correct_k = res.correct_k

    # Keep track of dimensionality reduction times
    _t0 = time.time()

    # Dimensionality reduce
    X_reduced = UnsupervisedCluster._dimensionality_reduction(X,
                                                              hyperparams['reduce'],
                                                              hyperparams['n_components'])
    _t01 = time.time()
    reduction_time.append((_t01-_t0, X.shape))

    # NbClust clustering
    print('Customer ID {}\nDB Size : {}'.format(customer_id, X_reduced.shape))

    try:
        print('Starting NbClust')
        # Perform clustering with NbClust package
        result = UnsupervisedCluster.cluster_with_hyperparameters(hyperparams, X)
    except RRuntimeError as e:
        if str(e).__contains__('computationally singular'):
            # The eigenvalue matrix is singular. Reduce the number of dimensions
            _hyperparams = hyperparams
            _hyperparams['n_components'] = int(_hyperparams['n_components'] / 2)
            result = UnsupervisedCluster.cluster_with_hyperparameters(hyperparams, X)
        else:
            print(e)
            continue
            # raise(e)

    _t02 = time.time()
    calculate_time.append((_t02 - _t01 - (_t01 - _t0), X_reduced.shape))

#%% Plotting

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

xs = [shape[0] for (_time, shape) in calculate_time]
ys = [_time for (_time, shape) in calculate_time]
xs_reduced = [shape[0] for (_time, shape) in reduction_time]
ys_reduced = [_time for (_time, shape) in reduction_time]

#Get Data
Xs = np.array(xs)
Ys = np.array(ys)
Xs_reduced = np.array(xs_reduced)
Ys_reduced = np.array(ys_reduced)

#Fit Data
poly_features = PolynomialFeatures(degree=4, include_bias=True)
x_poly = poly_features.fit_transform(Xs.reshape(-1,1))
lin_reg = LinearRegression()
lin_reg.fit(x_poly, Ys)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x_poly, Ys)

x_poly_reduce = poly_features.fit_transform(Xs_reduced.reshape(-1,1))
lin_reg_reduce = LinearRegression()
lin_reg_reduce.fit(x_poly_reduce, Ys_reduced)
lasso_reg_reduce = Lasso(alpha=0.3)
lasso_reg_reduce.fit(x_poly_reduce, Ys_reduced)

#Predict instances
_x = np.arange(10,4000,100)
x_test = poly_features.fit_transform(_x.reshape(-1,1))
_time_poly = lin_reg.predict(x_test)
_time_lasso = lasso_reg.predict(x_test)
_time_poly_reduce = lin_reg_reduce.predict(x_test)
_time_lasso_reduce = lasso_reg_reduce.predict(x_test)

#Plot data
fig, ax = plt.subplots(1)
ax.plot(_x, _time_poly, label='poly_Nb')
ax.plot(_x, _time_lasso, label='lasso_Nb')
ax.plot(_x, _time_poly_reduce, label='poly reduce')
ax.plot(_x, _time_lasso_reduce, label='lasso reduce')
ax.scatter(xs, ys, c='r', label='Nb Compute')
ax.scatter(Xs_reduced, Ys_reduced, label='Reduce')

ax.set_title('Predicted and measure computation time')
ax.set_xlabel('Database Size (indstances)')
ax.set_ylabel('Time (seconds)')
ax.set_xlim(0, max(_x) + 100)
ax.set_ylim(ax.get_xlim())
ax.legend()







