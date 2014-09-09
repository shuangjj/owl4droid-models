#!/usr/bin/env python
# File: audio_analyze.py
# Read audio data from db and analyze/classify audio feature

import os
from optparse import OptionParser

from constants import *
import models
import features
import db_helper
import numpy as np

def main():
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

    ## Configurations
    train_scenes = options.trainscenes.split() #['office', 'cafe', 'home']
    test_scenes = options.testscene.split()
    sensors = options.sensors.split()

    ##------------------------------------------------------------------------------------
    #                  Collect training and testing dataset
    # 
    # classes = ['bar', 'cafe', 'elevator', 'library', 'office', 'subwaystation']
    ##------------------------------------------------------------------------------------

    ## Collect training dataset
    train_light = [];    train_audio = []
    train_wifi = [];     train_bluetooth = [] 

    train_tuples = enumerateAllSamples('train', train_scenes, sensors[0], sensors[1:])
    print "# of train tuples: ", len(train_tuples)

    for sample_tuple in train_tuples:
        scene = sample_tuple[0]
        idx = 1
        for sensor in sensors:
            if sensor == 'light':
                train_light.append((scene, sample_tuple[idx]))
            elif sensor == 'audio':
                train_audio.append((scene, sample_tuple[idx]))
            elif sensor == 'wifi':
                train_wifi.append((scene, sample_tuple[idx]))
            elif sensor == 'bluetooth':
                train_bluetooth.append((scene, sample_tuple[idx])) 
            else:
                print 'Unsupported sensor %s' % (sensor)
            idx = idx + 1

    ## Collect testing dataset
    test_light = [];    test_audio = []
    test_wifi = [];     test_bluetooth = [] 
    test_tuples = enumerateAllSamples('test', test_scenes, sensors[0], sensors[1:])
    print "# of test tuples: ", len(test_tuples)

    ## Assign test tuples to profile and ensemble tuples
    profile_tuples = []; profile_scenes = []
    ensemble_tuples = []; ensemble_scenes = []

    for sample_tuple in test_tuples:
        if profile_scenes.count(sample_tuple[0]) > ensemble_scenes.count(sample_tuple[0]):
            ensemble_tuples.append(sample_tuple)
            ensemble_scenes.append(sample_tuple[0])
        else:
            profile_tuples.append(sample_tuple)
            profile_scenes.append(sample_tuple[0])
    print "# of profile tuples", len(profile_tuples)
    print "# of ensemble tuples: ", len(ensemble_tuples)

    for sample_tuple in profile_tuples:
        scene = sample_tuple[0]
        idx = 1
        for sensor in sensors:
            if sensor == 'light':
                test_light.append((scene, sample_tuple[idx]))
            elif sensor == 'audio':
                test_audio.append((scene, sample_tuple[idx]))
            elif sensor == 'wifi':
                test_wifi.append((scene, sample_tuple[idx]))
            elif sensor == 'bluetooth':
                test_bluetooth.append((scene, sample_tuple[idx])) 
            else:
                print 'Unsupported sensor %s' % (sensor)
            idx = idx + 1

    ##------------------------------------------------------------------------------------
    #                  Testing and profiling for individual models
    ##------------------------------------------------------------------------------------
   
    ## Audio model 
    if 'audio' in sensors:
        audio = models.AudioModel(train_scenes)
        print INDENT_L4, '=' * 20 + ' Audio Classifier ' + '=' * 20
        if audio.setTrainVector(train_audio) > 0:
            audio_model = audio.trainNB()
            if audio.setTestVector(test_audio) > 0:
                audio_score = audio.scoreNB(audio_model)
                audio.predict_profile(audio_model)
                #audio.testNB(audio_model)

    ## Light model
    if 'light' in sensors:
        light = models.LightModel(train_scenes)
        print INDENT_L4, '=' * 20 + ' Light Classifier ' + '=' * 20
        if light.setTrainVector(train_light) > 0:
            light_model = light.trainNB()
            if light.setTestVector(test_light) > 0:
                light_score = light.scoreNB(light_model)
                light.predict_profile(light_model)
                #light.testNB(light_model)

    ## Wifi model
    if 'wifi' in sensors:
        wifi = models.WifiModel(train_scenes)
        print INDENT_L4, '=' * 20 + ' Wifi Classifier ' + '=' * 20
        if wifi.setTrainVector(train_wifi) > 0:
            wifi_model = wifi.trainNB()
            if wifi.setTestVector(test_wifi) > 0:
                wifi_score = wifi.scoreNB(wifi_model)
                wifi.predict_profile(wifi_model)
                #wifi.testNB(wifi_model)

    ## Bluetooth model
    if 'bluetooth' in sensors:
        bluetooth = models.BluetoothModel(train_scenes)
        print INDENT_L4, '=' * 20 + ' Bluetooth Classifier ' + '=' * 20
        if bluetooth.setTrainVector(train_bluetooth) > 0:
            bluetooth_model = bluetooth.trainNB()
            if bluetooth.setTestVector(test_bluetooth) > 0:
                bluetooth_score = bluetooth.scoreNB(bluetooth_model)
                bluetooth.predict_profile(bluetooth_model)
                #bluetooth.testNB(bluetooth_model)

    ##------------------------------------------------------------------------------------
    #                     Mixed model / majority voting
    ##------------------------------------------------------------------------------------
    print 
    print INDENT_L4, "=" * 40 + " Majority Voting " + "=" * 40
    ## 
    total = 0; correct = 0
    target_classes = sorted(train_scenes)

    ## weights for sensors
    audio_fsize = 12
    light_fsize = 1
    wifi_fsize = 2
    bluetooth_fsize = 2

    ## Predict and ensemble 
    #sample_tuples = enumerateAllSamples('test', test_scenes, sensors[0], sensors[1:])

    for sample_tuple in ensemble_tuples:

        result_vector = np.zeros(len(target_classes))
        audio_vector = []; light_vector = []; 
        bluetooth_vector = []; wifi_vector = []

        scene = sample_tuple[0]
        idx = 1
        for sensor in sensors:
            testset = [(scene, sample_tuple[idx])]
            if sensor == 'audio':
                audio.setTestVector(testset)
                audio_predict = audio.predictNB(audio_model)[0]
                #audio.testNB(audio_model)
                ## Construct predict vector
                for t in target_classes:
                    if t == audio_predict:
                        audio_vector.append(audio.vote(audio_predict))
                    else:
                        audio_vector.append(0)
                result_vector = result_vector + np.array(audio_vector)
     
            elif sensor == 'light':
                light.setTestVector(testset)
                light_predict = light.predictNB(light_model)[0]
                #light.testNB(light_model)
                for t in target_classes:
                    if t == light_predict:
                        light_vector.append(light.vote(light_predict))
                    else:
                        light_vector.append(0)
                result_vector = result_vector + np.array(light_vector)

            elif sensor == 'wifi':
                wifi.setTestVector(testset)
                wifi_predict = wifi.predictNB(wifi_model)[0]
                #wifi.testNB(wifi_model)
                for t in target_classes:
                    if t == wifi_predict:
                        wifi_vector.append(wifi.vote(wifi_predict))
                    else:
                        wifi_vector.append(0)
                result_vector = result_vector + np.array(wifi_vector)

            elif sensor == 'bluetooth':
                bluetooth.setTestVector(testset)
                bluetooth_predict = bluetooth.predictNB(bluetooth_model)[0]
                #bluetooth.testNB(bluetooth_model)
                for t in target_classes:
                    if t == bluetooth_predict:
                        bluetooth_vector.append(bluetooth.vote(bluetooth_predict))
                    else:
                        bluetooth_vector.append(0)
                result_vector = result_vector + np.array(bluetooth_vector)
            idx = idx + 1

        ## Ensemble
        result_idx = result_vector.argmax()
        result_class = target_classes[result_idx]
        print 
        print "Ensemble predict %s to %s" % (scene, result_class), result_vector
        print audio_vector, light_vector, bluetooth_vector, wifi_vector
        total = total + 1
        # Correct & Reward 
        if result_class == scene:
            correct = correct + 1
            if 'audio' in sensors and audio_predict == result_class:
                audio.weight = audio.weight * 1.1

            if 'light' in sensors and light_predict == result_class:
                light.weight = light.weight * 1.1

            if 'wifi' in sensors and wifi_predict == result_class:
                wifi.weight = wifi.weight * 1.1

            if 'bluetooth' in sensors and bluetooth_predict == result_class:
                bluetooth.weight = bluetooth.weight * 1.1
        # Wrong & Punish
        else:
            if 'audio' in sensors and audio_predict == result_class:
                audio.weight = audio.weight * 0.9

            if 'light' in sensors and light_predict == result_class:
                light.weight = light.weight * 0.9

            if 'wifi' in sensors and wifi_predict == result_class:
                wifi.weight = wifi.weight * 0.9

            if 'bluetooth' in sensors and bluetooth_predict == result_class:
                bluetooth.weight = bluetooth.weight * 0.9

            print "wrong for predict target %s to %s" % (scene, result_class)
        print '-' * 80

    print
    
    print "Score of individual audio, light, wifi and bluetooth: ", \
        audio_score if 'audio' in sensors else 'Unavailable', \
        light_score if 'light' in sensors else 'Unavailable', \
        wifi_score if 'wifi' in sensors else 'Unavailable', \
        bluetooth_score if 'bluetooth' in sensors else 'Unavailable'

    print "Voting weights of audio, light, wifi, and bluetooth: ", \
        audio.weight if 'audio' in sensors else 'Unavailable', \
        light.weight if 'light' in sensors else 'Unavailable', \
        wifi.weight if 'wifi' in sensors else 'Unavailable', \
        bluetooth.weight if 'bluetooth' in sensors else 'Unavailable'

    print "Score of majority voting (%d in %d): %f" % (correct, total, float(correct)/total)
        
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

if __name__ == "__main__":
    main()
