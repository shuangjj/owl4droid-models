#!/usr/bin/env python
# File: features.py
# Extract formatted features form database
from db_helper import DBHelper

import audio_features as af
import light_features as lf
import bluetooth_features as bf
import wifi_features as wf

import ast
import datetime as dt

from constants import *

import numpy as np
def getFeatureVector(usage, scene, features):
    V = []  # Feature vector
    #  Audio
    if('audio' in features):
        mfccs = readAudioFeature(usage, scene)
        # Take the average of mfccs of audio frames as features
        mfcc = np.mean(mfccs, axis=0)
        V = V + mfcc.tolist()

    # Light
    if('light' in features):
        luxs = readLightFeature(usage, scene)
        lux = np.mean(luxs, axis=0)
        V = V + [lux]

    #TODO: Bluetooth
    if('bluetooth' in features):
        mAddresses = readBluetoothFeature(usage, scene)
        print mAddresses

    #TODO: Wifi feature
    if('wifi' in features):
        bssids = readWifiFeature(usage, scene)
        print bssids
    return V

class FeatureRecord:
    def __init__(self, rec):
        self.data_id = rec[0]
        #self.dev_id = rec[1]
        self.probe = rec[1]
        self.timestamp = rec[2]
        self.values = rec[3]
        # Convert string values to python map style feature set
        self.features = ast.literal_eval(self.values)

    def getDateHour(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H')

    def getHour(self):
        print ''

    def getTime(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S') 

#-------------------------------------------------------------------------------
# Read Bluetooth features from DB
#-------------------------------------------------------------------------------
def readBluetoothFeature(usage, scene):
    db = DBHelper(usage, 'bluetooth', scene)
    db.fetchDB()
    # Extract bluetooth features
    bluetooth_features = []
    for rec in db.fetchData():
        bluetooth = bf.BluetoothFeatures(rec)
        feature = (bluetooth.getDeviceName(), bluetooth.getDeviceAddress())
        bluetooth_features.append(feature[1])
    db.closeDB()

    #print bluetooth_features
    print INDENT_L4, "Total # of bluetooth features extracted from %s: %d" % (scene, len(bluetooth_features))
    return list(set(bluetooth_features))

#-------------------------------------------------------------------------------
# Read Wifi features from DB
#-------------------------------------------------------------------------------
def readWifiFeature(usage, scene):
    db = DBHelper(usage, 'wifi', scene)
    ## Extract Wifi feature
    wifi_features = []
    for rec in db.fetchData():
        wifi = wf.WifiFeatures(rec)
        feature = (wifi.getSSID(), wifi.getBSSID())
        wifi_features.append(feature[1])
    db.closeDB()
    #print wifi_features
    print INDENT_L4, "Total # of wifi features extracted from %s: %d" % (scene, len(wifi_features))
    return list(set(wifi_features))

#-------------------------------------------------------------------------------
# Read light feature from DB
#-------------------------------------------------------------------------------
def readLightFeature(usage, scene):
    db = DBHelper(usage, 'light', scene)
    db.fetchDB()
    ## Extract light feature
    audio_features = []
    for rec in db.fetchData():
        light = lf.LightFeatures(rec)
        #print light.getLux()
        audio_features.append(light.getLux())
    db.closeDB()
    print INDENT_L4, "Total # of light features extracted from %s: %d" % (scene, len(audio_features))

    return audio_features

#-------------------------------------------------------------------------------
# Read audio feature from DB
#-------------------------------------------------------------------------------
def readAudioFeature(usage, scene):
    db = DBHelper(usage, 'audio', scene)
    db.fetchDB()
    ## Extract audio features
    train_seqs = []
    for rec in db.fetchData():
       audio = af.AudioFeatures(rec) 
       train_seqs.append(audio.getMfccs())

    db.closeDB()

    print INDENT_L4, "Total # of mfcc features extracted from %s: %d" % (scene, len(train_seqs))
    return train_seqs

