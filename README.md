## Packages: <br/>
1. clustering <br/>
Clustering algorithms and modules to cluster points based on similarity.  After the points are clustered, they are loaded into mongodb to be used in the MIL categorization section of this project.  TODO : Move some of the database pipeline modules OUT of clustering, pipelines & mongo
---
2. data
Includes .csv files for clustering, manual cleaning of some data, tensorflow TF record protos, vocabulary files, and a master .csv file of extracted databases from .mdf to .csv
---
4. MILCategorization
Multiple instance categorization for bags of clustered data.  Optionally, use this on databases segmented based on controller instead of clustered databases.
---
6. ranking
Holds models for tensorflow ranking models.  Includes rankign of optimal hyperparameters for clustering databases based on points similarity
