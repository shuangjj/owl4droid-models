#!/usr/bin/env python
# File: features_from_db.py
# Wrapper for fetching sensor features from db
import db_helper
from constants import *
import features

import os
##----------------------------------------------------------------------------------------
# Enumerate all samples of all kinds of sensors
# The order of the returned sample tuples is the same with (base, others)
##----------------------------------------------------------------------------------------
def enumerateAllSamples(usage, scenes, base, others):
    sample_tuples = []
    db = db_helper.DBHelper(FUNFDB, SERVER_PATH)
    query_sql = "SELECT scene, location, date, SUBSTR(time, 1, 2) 'hour', \
            dbpath FROM %s WHERE usage=? AND sensor=?" % FUNFTBL
    rows = db.query_db(query_sql, (usage, base))
    for row in rows:
        sample_tuple = []
        sample_good = True
        # Base information
        scene = row[0]; 
        location = row[1]; 
        date =row[2]; hour = row[3];
        dbpath = row[4]
        
        if scene not in scenes:
            continue
        # Add base sensor values
        sample_tuple.append(scene)
        sample_tuple.append( getSamplesBySensor(base, dbpath) )

        ## Add values of the other sensors
        query_sensor_data = "SELECT dbpath FROM %s WHERE usage=? AND scene=? AND location=? AND date=? \
                AND SUBSTR(time, 1, 2)=? AND sensor=?" % (FUNFTBL)
        for sensor in others:
            rows = db.query_db(query_sensor_data, (usage, scene, location, date, hour, sensor))
            if len(rows) > 0:
                sample_tuple.append( getSamplesBySensor(sensor, rows[0][0]) )
            else:
                default_values = getSamplesBySensor(sensor, None)
                if default_values is None:
                    sample_good = False
                    print dbpath
                    break;
                else:
                    sample_tuple.append(default_values)
        if sample_good:
            sample_tuples.append(tuple(sample_tuple))

    return sample_tuples

def getSamplesBySensor(sensor, dbpath):
    values = None
    if sensor == 'audio':
        if dbpath is None:
            print 'No default values for empty audio'
        else:
            values =  getAudioSampleValues(dbpath)
    elif sensor == 'light':
        if dbpath is None:
            print 'No default values for empty light'
        else:
            values = getLightSampleValues(dbpath)
    elif sensor == 'wifi':
        if dbpath is None:
            values = [[0]]
        else:
            values = getWifiSampleValues(dbpath)
    elif sensor == 'bluetooth':
        if dbpath is None:
            values = [[0]]
        else:
            values = getBluetoothSampleValues(dbpath)
    else:
        print 'Unsupported sensor type ', sensor

    return values

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
        S.append([light.getLux()])
    #print INDENT_L4, "Extract %d light values from %s" % (len(S), os.path.basename(dbdir))
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
    #print INDENT_L4, "Extract %d audio values from %s" % (len(S), os.path.basename(dbdir))
    return S
#-------------------------------------------------------------------------------
# Read Wifi sample values into set
#-------------------------------------------------------------------------------
def getWifiSampleValues(dbdir):
    rows = getDBDataByPath(dbdir)
    bss_level = {};
    for rec in rows:
        wifi = features.WifiFeatures(rec)
        #feature = (wifi.getSSID(), wifi.getBSSID())
        bss_level[wifi.getBSSID()] = wifi.getLevel()
    # Construct vector <num_wifi, rssi_level> for each sample
    S = []
    wifi_num = len(bss_level)
    for level in bss_level.values():
        S.append([wifi_num])
    #print S
    #print INDENT_L4, "Extract %d wifi values from %s" % (len(S), os.path.basename(dbdir))
    return S
#-------------------------------------------------------------------------------
# Read bluetooth sample values into set
#-------------------------------------------------------------------------------
def getBluetoothSampleValues(dbdir):
    rows = getDBDataByPath(dbdir)
    S = []
    addr_rssi = {}
    for rec in rows:
        bluetooth = features.BluetoothFeatures(rec)
        #feature = (bluetooth.getDeviceName(), bluetooth.getDeviceAddress())
        addr_rssi[bluetooth.getDeviceAddress()] = bluetooth.getRSSI()
    # Bluetooth feature vector
    bluetooth_num = len(addr_rssi)
    for rssi in addr_rssi.values():
        S.append([bluetooth_num])
    #print INDENT_L4, "Extract %d bluetooth values from %s" % (len(S), os.path.basename(dbdir))
    return S

