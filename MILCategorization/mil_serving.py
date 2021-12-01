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
app = FastAPI()
SERVE_PORT = 8003

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


@app.get("/")
async def root():
    msg=("See API endpoints at /CompNBPredictor, MultiNBPredictor, "+
        "KNNPredictor, SVMCL1SIPredictor, SVMCRBFSIPredictor, "+
        "SVMCL1MILESPredictor, SVMCRBFMILESPredictor")
    
    return {"message":msg}

@app.post("/CompNBPredictor/")
async def CompNB_server(data:RawInputData):
    """Serve predictions from the CompNBPredictor"""
    msg="Not Implemented"
    return {"message":msg}


@app.post("/MultiNBPredictor/")
async def MultiNB_server(data:RawInputData):
    """Serve predictions from the MultiNBPredictor"""
    msg="Not Implemented"
    return {"message":msg}

@app.post("/KNNPredictor/")
async def KNN_server(data:RawInputData):
    """Serve predictions from the KNNPredictor"""
    msg="Not Implemented"
    return {"message":msg}

@app.post("/SVMCL1SIPredictor/")
async def SVMCL1SI_server(data:RawInputData):
    """Serve predictions from the SVMCL1SIPredictor"""
    msg="Not Implemented"
    return {"message":msg}

@app.post("/SVMCRBFSIPredictor/")
async def SVMCRBFSI_server(data:RawInputData):
    """Serve predictions from the SVMCRBFSIPredictor"""
    msg="Not Implemented"
    return {"message":msg}

@app.post("/SVMCL1MILESPredictor/")
async def SVMCL1MILES_server(data:RawInputData):
    """Serve predictions from the SVMCL1MILESPredictor"""
    msg="Not Implemented"
    return {"message":msg}

@app.post("/SVMCRBFMILESPredictor/")
async def SVMCRBFMILES_server(data:RawInputData):
    """Serve predictions from the SVMCRBFMILESPredictor"""
    msg="Not Implemented"
    return {"message":msg}


#%%
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=SERVE_PORT)