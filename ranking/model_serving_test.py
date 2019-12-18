# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:02:51 2019

@author: z003vrzk
"""

# Third party imports
from pymongo import MongoClient

# Local imports
from model_serving import LoadSerializedAndServe

# Instantiate classes
ServingClass = LoadSerializedAndServe()

client = MongoClient('localhost', 27017)
db = client['master_points']
collection = db['raw_databases']




#%% Demo LoadSerializedAndServe

_cursor = collection.find()
document = next(_cursor)

# Make predictions with model2 and model4
prediction2 = ServingClass.load_serialized_and_serve_model2(document)
prediction4 = ServingClass.load_serialized_and_serve_model4(document, 
                                                            list_size=None,
                                                            peritem_source='default')
# Sorting
a = sorted(prediction4, key = lambda output : output.score)
b = sorted(prediction2, key = lambda output : output.score)

# Print all scores
print('\nPREDICTED Best Scores \n')
for output in reversed(a):
    print(output.hyperparameter_dict)

print('\nMEASURED Best Scores \n')
for _measured_hparam in list(document['hyper_labels'].values()):
    print(_measured_hparam)

# Print best scores for reference
for _idx in range(-1, -6, -1):
    print(a[_idx].hyperparameter_dict, '|', b[_idx].hyperparameter_dict)
    print(a[_idx].score, '|', b[_idx].score)

# Create list of 5 best hyperparam dicts
best_hyperparam_list = []
for _idx in range(-1, -6, -1):
    best_hyperparam_list.append(a[_idx].hyperparameter_dict)
    
    
#%% Clustering using the predicted best hyperparameters


