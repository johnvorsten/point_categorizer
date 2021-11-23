## MIL - Multiple instance learning on BMS datasets

## TODO
Estimators (Pickling, saving)
    # Re-train estimator
    # Pickle SkLearn estimator
    # Save (training and validation) data for modular testing and validation
    # Record optimal hyperparameters
    # Document validation results (.txt)

Create HTTP API for accessing and serving predictions
    Document raw data input fields (what specifically is required?)
    Create input data structure for raw data (dataframe, dictionary, tuple, array...)
    Deal with incomplete data (Error handling - see 8 cases of failure mode)
    Transform raw data in estimator
    Input data to estimator
    Predict results
    Send results
    ** Use FastAPI (https://fastapi.tiangolo.com/)

Other - Low priority
    Rate limiting

Web page
    HTML Form fields
    Javascript - Template data input
    Javascript - POST form
    Javascript - Serialize data to JSON
    
Containerize service
Deploy container

Vocabulary
* MILES - Multiple instance learning via embedded instance selection

Important files
* si_mil_sklearn - Contains a KNN, Multinomial Native Bayes, and Complement Native Bayes estimator which operate on single-instances of bags. Bag labels are predicted using single-instance inference of the bag label. In this case the bag label is predicted from the mode of labels assigned to instances within a bag
* svm_miles - Contains a linear L1 regularized SVM classifier and a RBF distance kernel SVM classifier which estimates bag labels based on an embedded feature vector which represents a set of instances comprising a bag
* bag_cross_validate - A module which allows for cross-validation on the bag-level using single-instance inference with sklearn estimators.
* mil_load - A set of functions for loading and saving MIL datasets


Optimal hyperparameters

For single instance estimators:
KNN - _knn_nneighbors = 3 # Number of neighbors to infer instance label. Best result of cross validation for KNN estimator.
multiNB - _multinb_alpha = 0.5 # Regularization parameter for Multinomial Native Bayes estimator. Best result of cross validation
compNB - _compnb_alpha = 0.9 # Regularization parameter for Complement native bayes estimator. Best result of cross validation
svmc_l1 - _svm_c = 5 # Regularization paramter for linear SVM classifier with L1 norm penalization on single-instance. Best result of cross validation
svmc - _svm_c = 5 # Regularization paramter for SVM classifier with RBF distance function on single-instance. Best result of cross validation

For MILES estimators:
SVMC L1 - L1 norm penalized, squared hinge loss,  C = 2 regularized
SVMC RBF - RBF Kenerel, degree=3, gamma=kenel coefficient=5, , C=10 regularization

Saving models for prediction in a service:
* Save Training data for each estimator
* Save source code for generating the model
* Save requirements file for sklearn version and python version
* Save cross validation score and results

Saved classifiers:
svmc_l1_miles.clf - Linear L1 penalized SVM trained on embedded bags
svmc_rbf_miles.clf - RBF distance L2 penalized SVM trained on embedded bags
compNB_si.clf - Complement native bayes estimator trained on single instances which inherit the bag label
knn_si.clf - KNN estimator trained on single instances which inherit the bag label
multiNB_si - Multinomial Native Bayes estimator, trained on single instances which inherit the bag label
svmc_rbf_si.clf - RBF L2 penalized SVM trained on single instances which inherit the bag label
svmc_l1_si.clf - Linear L1 penalized SVM trained on single instances which inherit the bag label

Saved data:
data_milesembedded_test - A tuple of (data, labels) where data is a dense array of size (5035, 907) and labels are an array of size (907). Each slice along axis=1 of data is related to a label along axis=0 in labels

data_milesembedded_trian - A tuple of (data, labels) where data is a dense array of size (5035, 907) and labels are an array of size (907). Each slice along axis=1 of data is related to a label along axis=0 in labels

data_sicat_test - A tuple of (data, labels) where data is a numpy array of size (440) containing numpy arrays of shape (n,3236) where n is the number of instances within a bag. The test set contains 440 bags. Labels are an array of size (440). Each bag along axis=0 of data is related to a label along axis=0 in labels. Instances within bags are categorically encoded, and contain sparse feature vectors. Use this set for traning and prediction using ONLY the multinomial native bayes and complement native bayes estimators

data_sicat_trian - A tuple of (data, labels) where data is a numpy array of size (1689) containing numpy arrays of shape (n,3236) where n is the number of instances within a bag. The test set contains 1689 bags. Labels are an array of size (1689). Each bag along axis=0 of data is related to a label along axis=0 in labels. Instances within bags are categorically encoded, and contain sparse feature vectors. Use this set for traning and prediction using ONLY the multinomial native bayes and complement native bayes estimators

data_sidense_test - A tuple of (data, labels) where data is a numpy array of size (440) containing numpy arrays of shape (n,3236) where n is the number of instances within a bag. The test set contains 440 bags. Labels are an array of size (440). Each bag along axis=0 of data is related to a label along axis=0 in labels

data_sidense_trian - A tuple of (data, labels) where data is a numpy array of size (1689) containing numpy arrays of shape (n,3236) where n is the number of instances within a bag. The test set contains 1689 bags. Labels are an array of size (1689). Each bag along axis=0 of data is related to a label along axis=0 in labels

History
Old data had 3236 features..
New data has...

