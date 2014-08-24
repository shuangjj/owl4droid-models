#!/usr/bin/env python
# File: light_features.py
# Light feature extraction from LightSensorProbe data

import ast
import datetime as dt
import sqlite3

class LightFeatures():
    def __init__(self, rec):
        self.data_id = rec[0]
        self.dev_id = rec[1]
        self.probe = rec[2]
        self.timestamp = rec[3]
        self.values = rec[4]
        self.features = ast.literal_eval(self.values)

#-------------------------------------------------------------------------------
# Light feature accessors
#-------------------------------------------------------------------------------
    def getTime(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S') 

    def getLux(self):
        return self.features["lux"]
