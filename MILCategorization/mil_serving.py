# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 07:48:59 2021

API server for several estimators, including 
* 

@author: jvorsten
"""

# Python imports
from typing import Union, List
import configparser
import warnings

# Third party imports
from fastapi import FastAPI
import uvicorn

# Local imports
from si_mil_sklearn_predict import (CompNBPredictor, KNNPredictor, 
                                    MultiNBPredictor, SVMCL1SIPredictor, 
                                    SVMCRBFSIPredictor)
from svm_miles_predict import (SVMCL1MILESPredictor, SVMCRBFMILESPredictor)
from dataclass_serving import (RawInputDataPydantic, 
                               prediction_map, PredictorOutput)

# Declarations
config = configparser.ConfigParser()
config.read(r'./serving_config.ini')
warnings.filterwarnings(action='ignore', message='elementwise comparison failed')

# Application level
app = FastAPI(title=config['FastAPI']['title'],
              description=config['FastAPI']['description'],
              version=config['FastAPI']['version'],
              )

# Configuration for classifiers
SVMC_l1_classifier_filename = config['SVMCMILES']['SVMC_l1_classifier_filename']
SVMC_rbf_classifier_filename = config['SVMCMILES']['SVMC_rbf_classifier_filename']
concept_class_filename = config['SVMCMILES']['concept_class_filename']
CompNB_classifier_filename = config['SISklearn']['CompNB_classifier_filename']
# KNN_classifier_filename = config['SISklearn']['KNN_classifier_filename']
MultiNB_classifier_filename = config['SISklearn']['MultiNB_classifier_filename']
SVMCL1SI_classifier_filename = config['SISklearn']['SVMCL1SI_classifier_filename']
SVMCRBFSI_classifier_filename = config['SISklearn']['SVMCRBFSI_classifier_filename']

# Global objects
#TODO is this bad practice?
compNBPredictor = CompNBPredictor(
    CompNB_classifier_filename, 
    pipeline_type='categorical')
multiNBPredictor = MultiNBPredictor(
    MultiNB_classifier_filename, 
    pipeline_type='categorical')
# knnPredictor = KNNPredictor(
#     KNN_classifier_filename, 
#     pipeline_type='numeric')
svmcL1SIPredictor = SVMCL1SIPredictor(
    SVMCL1SI_classifier_filename, 
    pipeline_type='numeric')
svmcRBFSIPredictor = SVMCRBFSIPredictor(
    SVMCRBFSI_classifier_filename, 
    pipeline_type='numeric')
svmcL1MILESPredictor = SVMCL1MILESPredictor(
    SVMC_l1_classifier_filename)
svmcRBFMILESPredictor = SVMCRBFMILESPredictor(
    SVMC_rbf_classifier_filename)

#%%

def _determine_unpack(data:Union[List[RawInputDataPydantic], 
                                 RawInputDataPydantic]) -> Union[dict, List[dict]]:
    if isinstance(data, list):
        return [x.__dict__ for x in data]
    elif isinstance(data, RawInputDataPydantic):
        return data.__dict__
    else:
        msg="Expected list of [RawInputDataPydatic,], instead got {}."
        raise ValueError(msg.format(type(data)))
    return None

@app.on_event("startup")
async def startup():
    """How could I use this to add objects to the application state?
    app.state.PredictionObject = load_all_of_my_predictors()
    Then
    use app.state.PredictionObject in all of the endpoints?
    Access the app object by importing Request from fastapi, and adding
    it as a parameter in each endpoint like
    async def endpoint(request: Request, data_model:DataModel):"""
    pass

@app.get("/")
async def root():
    msg=("See API endpoints at /CompNBPredictor, MultiNBPredictor, "+
        "KNNPredictor, SVMCL1SIPredictor, SVMCRBFSIPredictor, "+
        "SVMCL1MILESPredictor, SVMCRBFMILESPredictor")
    return {"message":msg}

@app.post("/mil-prediction/CompNBPredictor/", response_model=PredictorOutput)
async def CompNB_server(data:List[RawInputDataPydantic]):
    """Serve predictions from the CompNBPredictor"""
    predictor_input = _determine_unpack(data)
    prediction = compNBPredictor.predict(predictor_input)
    return PredictorOutput(prediction=prediction_map[prediction[0]])

@app.post("/mil-prediction/MultiNBPredictor/", response_model=PredictorOutput)
async def MultiNB_server(data:List[RawInputDataPydantic]):
    """Serve predictions from the MultiNBPredictor"""
    predictor_input = _determine_unpack(data)
    prediction = multiNBPredictor.predict(predictor_input)
    return PredictorOutput(prediction=prediction_map[prediction[0]])

# @app.post("/mil-serving/KNNPredictor/", response_model=PredictorOutput)
# async def KNN_server(data:List[RawInputDataPydantic]):
#     """Serve predictions from the KNNPredictor"""
#     predictor_input = _determine_unpack(data)
#     prediction = knnPredictor.predict(predictor_input)
#     return PredictorOutput(prediction=prediction_map[prediction[0]])

@app.post("/mil-prediction/SVMCL1SIPredictor/", response_model=PredictorOutput)
async def SVMCL1SI_server(data:List[RawInputDataPydantic]):
    """Serve predictions from the SVMCL1SIPredictor"""
    predictor_input = _determine_unpack(data)
    prediction = svmcL1SIPredictor.predict(predictor_input)
    return PredictorOutput(prediction=prediction_map[prediction[0]])

@app.post("/mil-prediction/SVMCRBFSIPredictor/", response_model=PredictorOutput)
async def SVMCRBFSI_server(data:List[RawInputDataPydantic]):
    """Serve predictions from the SVMCRBFSIPredictor"""
    predictor_input = _determine_unpack(data)
    prediction = svmcRBFSIPredictor.predict(predictor_input)
    return PredictorOutput(prediction=prediction_map[prediction[0]])

@app.post("/mil-prediction/SVMCL1MILESPredictor/", response_model=PredictorOutput)
async def SVMCL1MILES_server(data:List[RawInputDataPydantic]):
    """Serve predictions from the SVMCL1MILESPredictor"""
    predictor_input = _determine_unpack(data)
    prediction = svmcL1MILESPredictor.predict(predictor_input)
    return PredictorOutput(prediction=prediction_map[prediction[0]])

@app.post("/mil-prediction/SVMCRBFMILESPredictor/", response_model=PredictorOutput)
async def SVMCRBFMILES_server(data:List[RawInputDataPydantic]):
    """Serve predictions from the SVMCRBFMILESPredictor"""
    predictor_input = _determine_unpack(data)
    prediction = svmcRBFMILESPredictor.predict(predictor_input)
    return PredictorOutput(prediction=prediction_map[prediction[0]])


#%%
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=config['FastAPI']['SERVE_PORT']
        )