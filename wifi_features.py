#!/usr/bin/env python
# File: wifi_features.py
# Wifi feature extraction from WifiProbe data

import ast
import datetime as dt

class WifiFeatures():
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
    def getSSID(self):
        return self.features["SSID"]

    def getBSSID(self):
        return self.features["BSSID"]
 
