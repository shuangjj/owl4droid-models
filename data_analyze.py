#!/usr/bin/env python
# File: audio_analyze.py
# Read audio data from db and analyze/classify audio feature

import os
from optparse import OptionParser

from constants import *
import models
import features
import db_helper

def main():
    ## Test
    # readFeatureSequence('test', 'train')
    # return

    ## Command argument parsing
    parser = OptionParser()
    parser.add_option("-t", "--test", dest="testscene", default="test",
            help="test scene", metavar="TEST_SCENE")

    parser.add_option("-n", "--train", dest="trainscenes", default="office",
            help="train scenes", metavar="TRAIN_SCENES")

    parser.add_option("-s", "--sensors", dest="sensors", default="audio light",
            help="", metavar="SENSORS")

    (options, args) = parser.parse_args()
    

    print INDENT_L1, '+--------------------------------------------------------------------+'
    print INDENT_L1, '| AMSC : Automatic Mobile Scene Classification                       |' 
    print INDENT_L1, '| Probes: audio                                                      |'
    print INDENT_L1, '| Author: Shuang Liang <shuang.liang2012@temple.edu>                 |'
    print INDENT_L1, '+--------------------------------------------------------------------+'
    ## Training
    # Classes
    # y = ['office', 'home', 'cafe', 'station', 'gym', 'test']
    scenes = options.trainscenes.split() #['office', 'cafe', 'home']
    sensors = options.sensors.split()

    X = []      # Observations
    light_obs = [] # (scene, observations)
    audio_obs = []
    wifi_obs = []
    bluetooth_obs = []
    for scene in scenes:
        for sensor in sensors:
            S = getSampleFeatureSet('train', scene, sensor)
            # Add observation set for scene
            if len(S) > 0:
                if sensor == 'light':
                    light_obs.append((scene, S))
                elif sensor == 'audio':
                    audio_obs.append((scene, S))
                elif sensor == 'wifi':
                    wifi_obs.append((scene, S))
                elif sensor == 'bluetooth':
                    bluetooth_obs.append((scene, S)) 
                else:
                    print 'Unsupported sensor %s' % (sensor)

                print 'Got %d train samples for %s->%s' % (len(S), scene, sensor)



    test_light = [];    test_audio = []
    test_wifi = [];     test_bluetooth = [] 
    testscene = options.testscene
    for sensor in sensors:
        S = getSampleFeatureSet('test', testscene, sensor)
        # Add observation set for test
        if len(S) > 0:
            if sensor == 'light':
                test_light.append((testscene, S))
            elif sensor == 'audio':
                test_audio.append((testscene, S))
            elif sensor == 'wifi':
                test_wifi.append((testscene, S))
            elif sensor == 'bluetooth':
                test_bluetooth.append((testscene, S)) 
            else:
                print 'Unsupported sensor %s' % (sensor)

            print 'Got %d test samples for %s->%s' % (len(S), scene, sensor)   
    # Train and test using light model
    if 'light' in sensors:
        light = models.LightModel(scenes)
        print INDENT_L4, '=' * 20 + ' Light Classifier ' + '=' * 20
        if light.setTrainVector(light_obs) > 0:
            model = light.trainNB()
            if light.setTestVector(test_light) > 0:
                light.testNB(model)

    # Train and test using audio model
    if 'audio' in sensors:
        audio = models.AudioModel(scenes)
        print INDENT_L4, '=' * 20 + ' Audio Classifier ' + '=' * 20
        if audio.setTrainVector(audio_obs) > 0:
            model = audio.trainNB()
            if audio.setTestVector(test_audio) > 0:
                audio.testNB(model)
  


def getSampleDBPaths(usage, scene, sensor='%', location='%', date='%', hour='%'):
    db = db_helper.DBHelper(FUNFDB, SERVER_PATH)
    query_sql = "SELECT sensor, dbpath FROM %s WHERE usage LIKE ? AND scene LIKE ? \
            AND sensor LIKE ? AND location LIKE ? \
            AND date LIKE ? AND strftime('%%H', time) LIKE ?" % (FUNFTBL)
    rows = db.query_db(query_sql, (usage, scene, sensor, location, date, hour))
    return rows

def getSampleFeatureSet(usage, scene, sensor):
    S = []
    paths = getSampleDBPaths(usage, scene, sensor)
    if len(paths) == 0:
        print 'No samples available for %s->%s->%s' % (usage, scene, sensor)
        return S
    #print 'Got %d samples for %s->%s->%s' % (len(paths), usage, scene, sensor)
    for sensor, dbpath in paths:
        if sensor == 'light':
            V = getLightSampleValues(dbpath)
        elif sensor == 'audio':
            V = getAudioSampleValues(dbpath)
        elif sensor == 'wifi':
            V = getWifiSampleValues(dbpath)
        elif sensor == 'bluetooth':
            V = getBluetoothSampleValues(dbpath)
        else:
            print 'Sensor type %s not supported' % sensor
        if len(V) > 0:
            S.append(V)
    return S

def getDBDataByPath(dbdir):
    dbname = os.path.split(dbdir)[1]
    dbpath = os.path.join(SERVER_PATH, os.path.split(dbdir)[0])
    db = db_helper.DBHelper(dbname, dbpath)
    query_sql = "SELECT * FROM %s" % DATA_TABLE_NAME
    return db.query_db(query_sql, ())
#-------------------------------------------------------------------------------
# Read light sample values into set
#-------------------------------------------------------------------------------
def getLightSampleValues(dbdir):
    rows = getDBDataByPath(dbdir)
    S = []
    for rec in rows:
        light = features.LightFeatures(rec)
        S.append(light.getLux())
    print INDENT_L4, "Extract %d light values from %s" % (len(S), os.path.basename(dbdir))
    return S

#-------------------------------------------------------------------------------
# Read audio sample values into set
#-------------------------------------------------------------------------------
def getAudioSampleValues(dbdir):
    rows = getDBDataByPath(dbdir)
    S = []
    for rec in rows:
        audio = features.AudioFeatures(rec)
        S.append(audio.getMfccs())
    print INDENT_L4, "Extract %d audio values from %s" % (len(S), os.path.basename(dbdir))
    return S
#-------------------------------------------------------------------------------
# Read Wifi sample values into set
#-------------------------------------------------------------------------------
def getWifiSampleValues(dbdir):
    rows = getDBDataByPath(dbdir)
    S = []
    for rec in rows:
        wifi = features.WifiFeatures(rec)
        feature = (wifi.getSSID(), wifi.getBSSID())
        S.append(feature[1])
    print INDENT_L4, "Extract %d wifi values from %s" % (len(S), os.path.basename(dbdir))
    return S
#-------------------------------------------------------------------------------
# Read bluetooth sample values into set
#-------------------------------------------------------------------------------
def getBluetoothSampleValues(dbdir):
    rows = getDBDataByPath(dbdir)
    S = []
    for rec in rows:
        bluetooth = features.BluetoothFeatures(rec)
        feature = (bluetooth.getDeviceName(), bluetooth.getDeviceAddress())
        S.append(feature[1])
    print INDENT_L4, "Extract %d bluetooth values from %s" % (len(S), os.path.basename(dbdir))
    return S

if __name__ == "__main__":
    main()
