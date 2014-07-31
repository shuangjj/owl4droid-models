#!/usr/bin/env python
# File: bluetooth_features.py
# Bluetooth feature extraction from BluetoothProbe data

import ast
import datetime as dt

class BluetoothFeatures():
    def __init__(self, rec):
        self.data_id = rec[0]
        self.dev_id = rec[1]
        self.probe = rec[2]
        self.timestamp = rec[3]
        self.values = rec[4]
        self.features = ast.literal_eval(self.values)

#-------------------------------------------------------------------------------
# Bluetooth feature accessors
#-------------------------------------------------------------------------------
    def getTime(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S') 

    def getDeviceName(self):
        return self.features["android.bluetooth.device.extra.NAME"]

    def getDeviceAddress(self):
        return self.features["android.bluetooth.device.extra.DEVICE"]["mAddress"]

    def getRSSI(self):
        return self.features["android.bluetooth.device.extra.RSSI"]


