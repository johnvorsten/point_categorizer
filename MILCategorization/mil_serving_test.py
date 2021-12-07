# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:54:57 2021

@author: jvorsten
"""

# Python imports
import json

# Third party imports
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Local imports
from mil_serving import SVMCL1MILES_server
from mil_serving import app

# Declarations
client = TestClient(app)

#%% 

def test_SVMCL1MILES_server():
    data = [
            {
            "DEVICEHI": 0,
            "DEVICELO": 0,
            "SIGNALHI": 0,
            "SIGNALLO": 0,
            "SLOPE": 0,
            "INTERCEPT": 0,
            "TYPE": "string",
            "ALARMTYPE": "string",
            "FUNCTION": "string",
            "VIRTUAL": False,
            "CS": "string",
            "SENSORTYPE": "string",
            "DEVUNITS": "string",
            "NAME": "First.Name01",
            "DESCRIPTOR": "string",
            "NETDEVID": "string",
            "SYSTEM": "string"
            },
            {
            "DEVICEHI": 0,
            "DEVICELO": 0,
            "SIGNALHI": 0,
            "SIGNALLO": 0,
            "SLOPE": 0,
            "INTERCEPT": 0,
            "TYPE": "string",
            "ALARMTYPE": "string",
            "FUNCTION": "string",
            "VIRTUAL": False,
            "CS": "string",
            "SENSORTYPE": "string",
            "DEVUNITS": "string",
            "NAME": "Another.Name",
            "DESCRIPTOR": "string",
            "NETDEVID": "string",
            "SYSTEM": "string"
            },
    ]
    header = {"accept": "application/json",
              "Content-Type": "application/json"}
    response = client.post("/CompNBPredictor/",
                           headers=header,
                           data=json.dumps(data))
    assert response.status_code == 200
    print(response.json())
    
if __name__ == "__main__":
    test_SVMCL1MILES_server()