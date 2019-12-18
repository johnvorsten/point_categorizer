# point_categorizer
Automatically label BAS <br/>
The goal of this repository is to create a framework that can automatically label BAS points into known classes. <br/>
***
The implementation steps are : <br/>
1) Collect data from remote servers <br/>
2) Build pipelines for data <br/>
3) Save data in .csv files (Actually, a noSQL system would suit this application well b/c we are dealing with sequential data <br/>
4) Automatically cluster data (unsupervised clustering) <br/>

## Files : <br/>
1. It would be a bad idea to describe all files before I'm finished with this repository
---

## Packages : <br/>
1. clustering <br/>
Clustering algorithms and modules to cluster points based on similarity.  After the points are clustered, they are loaded into mongodb to be used in the MIL categorization section of this project.  TODO : Move some of the database pipeline modules OUT of clustering and INTO pipelines & mongo <br/>
---
2. data
Includes .csv files for clustering, manual cleaning of some data, tensorflow TF record protos, vocabulary files, and a master .csv file of extracted databases from .mdf to .csv
---
3. error_dfs
After predicting the optimal number of clusters for databases, I saved the results in .csv files with their corresponding clustering hyperparameters here.  It is another form of saved data, and it will eventually be used for tensorflow ranking models. See the clustering package for transforming this data. TODO : Move the transformation to the pipeline package...
---
4. MILCategorization
Multiple instance categorization for bags of clustered data.  Optionally, use this on databases segmented based on controller instead of clustered databases.
---
5. pipelines & mongo
extract, transform, and load pipelines.  Currently, there exists a pipeline for extract .mdf -> transform -> load .csv.  Add pipelines for extract .csv -> transform -> load mongo.  Add pipelines for extract .csv -> transform error_df data -> load mongo
---
6. ranking
Holds models for tensorflow ranking models.  Includes rankign of optimal hyperparameters for clustering databases based on points similarity
