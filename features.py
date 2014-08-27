#!/usr/bin/env python
# File: features.py
# Extract formatted features form database

import os
import ast
import datetime as dt

#-------------------------------------------------------------------------------
# Map feature database row columns to corresponding variables
#-------------------------------------------------------------------------------
class FeatureRecord:
    def __init__(self, rec):
        self.data_id = rec[0]
        #self.dev_id = rec[1]
        self.probe = rec[1]
        self.timestamp = rec[2]
        self.values = rec[3]
        # Convert string values to python map style feature set
        self.features = ast.literal_eval(self.values)

    def getDate(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d')

    def getHour(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%H')

    def getTime(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S') 

#-------------------------------------------------------------------------------
# Light feature accessors
#-------------------------------------------------------------------------------
class LightFeatures(FeatureRecord):
    def __init__(self, rec):
        FeatureRecord.__init__(self, rec)

    def getLux(self):
        return self.features["lux"]
#-------------------------------------------------------------------------------
# Audio features getters
#-------------------------------------------------------------------------------
class AudioFeatures(FeatureRecord):
    
    def __init__(self, rec):
        FeatureRecord.__init__(self, rec)
    def getFeatures(self):
        return self.features

    def getMfccs(self):
        return self.features["mfccs"]

    def getDiffSecs():
        return self.features["diffSecs"]

    def getL1Norm(self):
        return self.features["l1Norm"]

    def getL2Norm(self):
        return self.features["l2Norm"]

    def getLinfNorm(self):
        return self.features["linfNorm"]

    def getPsdAcrossFrequencyBands(self):
        return self.featurns["psdAcrossFrequencyBands"]

#-------------------------------------------------------------------------------
# Wifi feature accessors
#-------------------------------------------------------------------------------
class WifiFeatures(FeatureRecord):
    def __init__(self, rec):
        FeatureRecord.__init__(self, rec)

    def getSSID(self):
        return self.features["SSID"]

    def getBSSID(self):
        return self.features["BSSID"]

#-------------------------------------------------------------------------------
# Bluetooth feature accessors
#-------------------------------------------------------------------------------
class BluetoothFeatures(FeatureRecord):
    def __init__(self, rec):
        FeatureRecord.__init__(self, rec)

    def getDeviceName(self):
        return self.features["android.bluetooth.device.extra.NAME"]

    def getDeviceAddress(self):
        return self.features["android.bluetooth.device.extra.DEVICE"]["mAddress"]

    def getRSSI(self):
        return self.features["android.bluetooth.device.extra.RSSI"]



