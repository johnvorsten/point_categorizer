# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 07:48:59 2021

API server for several estimators, including 
* 

@author: jvorsten
"""

# Python imports


# Third party imports
from fastapi import FastAPI

# Local imports
from si_mil_sklearn_predict import (CompNBPredictor, KNNPredictor, 
                                    MultiNBPredictor, SVMCL1SIPredictor, 
                                    SVMCRBFSIPredictor)
from svm_miles_predict import (SVMCL1MILESPredictor, SVMCRBFMILESPredictor, 
                               RawInputData)

# Declarations
app = FastAPI()
SERVE_PORT = 8003

#%%

@app.get("/")
async def root():
    msg="""
    See API endpoints at /CompNBPredictor, MultiNBPredictor, 
    KNNPredictor, SVMCL1SIPredictor, SVMCRBFSIPredictor, SVMCL1MILESPredictor, 
    SVMCRBFMILESPredictor"""
    
    return {"message":msg}

@app.post("/CompNBPredictor")
async def CompNBServer():
    """Serve predictions from the CompNBPredictor"""
    msg="Not Implemented"
    return {"message":msg}


@app.post("/MultiNBPredictor")
async def MultiNBServer():
    """Serve predictions from the MultiNBPredictor"""
    msg="Not Implemented"
    return {"message":msg}


@app.post("/KNNPredictor")
async def KNNServer():
    """Serve predictions from the KNNPredictor"""
    msg="Not Implemented"
    return {"message":msg}


@app.post("/SVMCL1SIPredictor")
async def SVMCL1SIServer():
    """Serve predictions from the SVMCL1SIPredictor"""
    msg="Not Implemented"
    return {"message":msg}


@app.post("/SVMCRBFSIPredictor")
async def SVMCRBFSIServer():
    """Serve predictions from the SVMCRBFSIPredictor"""
    msg="Not Implemented"
    return {"message":msg}


@app.post("/SVMCL1MILESPredictor")
async def SVMCL1MILESServer():
    """Serve predictions from the SVMCL1MILESPredictor"""
    msg="Not Implemented"
    return {"message":msg}


@app.post("/SVMCRBFMILESPredictor")
async def SVMCRBFMILESServer():
    """Serve predictions from the SVMCRBFMILESPredictor"""
    msg="Not Implemented"
    return {"message":msg}
