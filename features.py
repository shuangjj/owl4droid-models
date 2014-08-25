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
# 
#-------------------------------------------------------------------------------
def getFeatureVector(usage, scene, features):
    V = []  # Feature vector

    # Light
    if('light' in features):
        luxs = readLightFeature(usage, scene)
        lux = np.mean(luxs, axis=0)
        V = V + [lux]
    '''
    #  Audio
    if('audio' in features):
        mfccs = readAudioFeature(usage, scene)
        # Take the average of mfccs of audio frames as features
        mfcc = np.mean(mfccs, axis=0)
        V = V + mfcc.tolist()

    #TODO: Bluetooth
    if('bluetooth' in features):
        mAddresses = readBluetoothFeature(usage, scene)
        print mAddresses

    #TODO: Wifi feature
    if('wifi' in features):
        bssids = readWifiFeature(usage, scene)
        print bssids
'''
    return V

def getSampleDBPaths(usage, scene, sensor='%', location='%', datehour='%'):
    db = db_helper.DBHelper(constants.FUNFDB, '.')
    query_sql = "SELECT sensor, dbpath FROM %s WHERE usage LIKE ? AND scene LIKE ? \
            AND sensor LIKE ? AND location LIKE ? \
            AND datehour LIKE ?" % (constants.FUNFTBL)
    rows = db.query_db(query_sql, (usage, scene, sensor, location, datehour))
    return rows

def getSampleFeatureSets(usage, scene, sensor='%', location='%', datehour='%'):
    S = []
    paths = getSampleDBPaths(usage, scene, 'light')
    if len(paths) == 0:
        print 'No samples available for %s->%s->%s' % (usage, scene, sensor)
        return
    print 'Got %d samples for %s->%s->sensor' % (len(paths), usage, scene, sensor)
    for sensor, dbpath in paths:
        if sensor == 'light':
            V = getLightFeatureVector(dbpath)
            if len(V) > 0:
                S.append(V)
        elif sensor == 'audio':

        else:
            print 'Sensor type %s not supported' % sensor

    return S

#-------------------------------------------------------------------------------
# Read light feature vector from DB
#-------------------------------------------------------------------------------
def getLightFeatureVector(dbdir):
    dbname = os.path.split(dbdir)[1]
    dbpath = os.path.split(dbdir)[0]
    db = db_helper.DBHelper(dbname, dbpath)
    ## Extract light feature
    vector = []
    query_sql = "SELECT * FROM %s" % constants.DATA_TABLE_NAME
    for rec in db.query_db(query_sql, None):
        light = lf.LightFeatures(rec)
        #print light.getLux()
        vector.append(light.getLux())
    print INDENT_L4, "Total # of light values extracted from %s: %d" % (dbname, len(vector))
    return vector
'''
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
'''
