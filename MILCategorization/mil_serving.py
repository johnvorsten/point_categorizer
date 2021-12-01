# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 07:48:59 2021

API server for several estimators, including 
* 

@author: jvorsten
"""

# Python imports
from typing import Optional
from dataclasses import dataclass
import configparser

# Third party imports
from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

# Local imports
from si_mil_sklearn_predict import (CompNBPredictor, KNNPredictor, 
                                    MultiNBPredictor, SVMCL1SIPredictor, 
                                    SVMCRBFSIPredictor)
from svm_miles_predict import (SVMCL1MILESPredictor, SVMCRBFMILESPredictor, 
                                )

# Declarations
config = configparser.ConfigParser()
config.read(r'./serving_config.ini')

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
KNN_classifier_filename = config['SISklearn']['KNN_classifier_filename']
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
knnPredictor = CompNBPredictor(
    KNN_classifier_filename, 
    pipeline_type='numeric')
svmcL1SIPredictor = SVMCL1SIPredictor(
    SVMCL1SI_classifier_filename, 
    pipeline_type='numeric')
svmcRBFSIPredictor = SVMCRBFSIPredictor(
    SVMCRBFSI_classifier_filename, 
    pipeline_type='numeric')
svmcL1MILESPredictor = SVMCL1MILESPredictor(
    SVMC_l1_classifier_filename)
svmcRBFMILESPredictor = SVMCL1MILESPredictor(
    SVMC_rbf_classifier_filename)

#%%

class RawInputData(BaseModel):
    """Raw input data from HTTP Web form
    Note, any default values are not required by the estimator, and are 
    removed by the data cleaning pipeline
    
    Required numeric attributes
    ['DEVICEHI', 'DEVICELO', 'SIGNALHI', 'SIGNALLO', 'SLOPE', 'INTERCEPT']
    
    Required categorical attributes
    ['TYPE', 'ALARMTYPE', 'FUNCTION', 'VIRTUAL', 'CS','SENSORTYPE', 'DEVUNITS']
    
    Requried text attributes
    ['NAME', 'DESCRIPTOR']
    """
    # Required numeric attributes
    DEVICEHI: float
    DEVICELO: float
    SIGNALHI: float
    SIGNALLO: float
    SLOPE: float
    INTERCEPT: float
    # Required categorical attributes
    TYPE: str
    ALARMTYPE: str
    FUNCTION: str
    VIRTUAL: bool
    CS: str
    SENSORTYPE: str
    DEVUNITS: str
    # Requried text attributes
    NAME: str
    DESCRIPTOR: str
    NETDEVID: str
    SYSTEM: str


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

@app.post("/CompNBPredictor/")
async def CompNB_server(data:RawInputData):
    """Serve predictions from the CompNBPredictor"""
    return {"prediction":compNBPredictor.predict(data)}

@app.post("/MultiNBPredictor/")
async def MultiNB_server(data:RawInputData):
    """Serve predictions from the MultiNBPredictor"""
    return {"prediction":multiNBPredictor.predict(data)}

@app.post("/KNNPredictor/")
async def KNN_server(data:RawInputData):
    """Serve predictions from the KNNPredictor"""
    return {"prediction":knnPredictor.predict(data)}

@app.post("/SVMCL1SIPredictor/")
async def SVMCL1SI_server(data:RawInputData):
    """Serve predictions from the SVMCL1SIPredictor"""
    return {"prediction":svmcL1SIPredictor.predict(data)}

@app.post("/SVMCRBFSIPredictor/")
async def SVMCRBFSI_server(data:RawInputData):
    """Serve predictions from the SVMCRBFSIPredictor"""
    return {"prediction":svmcRBFSIPredictor.predict(data)}

@app.post("/SVMCL1MILESPredictor/")
async def SVMCL1MILES_server(data:RawInputData):
    """Serve predictions from the SVMCL1MILESPredictor"""
    return {"prediction":svmcL1MILESPredictor.predict(data)}

@app.post("/SVMCRBFMILESPredictor/")
async def SVMCRBFMILES_server(data:RawInputData):
    """Serve predictions from the SVMCRBFMILESPredictor"""
    return {"prediction":svmcRBFMILESPredictor.predict(data)}


#%%
if __name__ == "__main__":
    uvicorn.run(app, 
                host="127.0.0.1", 
                port=config['FastAPI']['SERVE_PORT']
                )