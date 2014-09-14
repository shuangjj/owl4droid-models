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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import matplotlib as mpl
import pandas

def main():
    ## Command argument parsing
    parser = OptionParser()
    parser.add_option("-t", "--test", dest="testscene", \
            default="bar cafe elevator library office subwaystation",
            help="test scene", metavar="TEST_SCENE")

    parser.add_option("-n", "--train", dest="trainscenes", 
            default="bar cafe elevator library office subwaystation",
            help="train scenes", metavar="TRAIN_SCENES")

    parser.add_option("-s", "--sensors", dest="sensors", default="audio light wifi bluetooth",
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
    learners = []
    for sensor in sensors:
        if 'audio' == sensor:
            learner = models.AudioModel(train_scenes)
            model_profiling(learner, train_audio, test_audio)
        elif 'light' == sensor:
            learner = models.LightModel(train_scenes)
            model_profiling(learner, train_light, test_light)
        elif 'wifi' == sensor:
            learner = models.WifiModel(train_scenes)
            model_profiling(learner, train_wifi, test_wifi)
        elif 'bluetooth' == sensor:
            learner = models.BluetoothModel(train_scenes)
            model_profiling(learner, train_bluetooth, test_bluetooth)
        else:
            print "Unsupported sensor %s, unable to create learners for that" % (sensor)

        learners.append(learner)
        #draw_confusion_matrix(learner.cm, abbreviate_names(train_scenes, scene_abbr_dict), \
                #abbreviate_names(train_scenes, scene_abbr_dict))


    ## Setup data table for precision
    dt_precision = {}
    dt_recall = {}
    for learner in learners:
        precisions = []
        recalls = []
        for scene in train_scenes: 
            precisions.append(learner.scene_precision[scene])
            recalls.append(learner.scene_recall[scene])

        dt_precision[learner.getName()]  = precisions
        dt_recall[learner.getName()] = recalls


    #draw_barh(learners, dt_precision, abbreviate_names(train_scenes, scene_abbr_dict), 'Precision')
    #draw_barh(learners, dt_recall, abbreviate_names(train_scenes, scene_abbr_dict), 'Recall')

    xtick_labels = abbreviate_names(train_scenes, scene_abbr_dict)
    ytick_labels = abbreviate_names(train_scenes, scene_abbr_dict)
    ## Draw confusion matrix altogether
    #draw_confusion_matrixes(learners, xtick_labels, ytick_labels)
    ##------------------------------------------------------------------------------------
    #                     Mixed model / majority voting
    ##------------------------------------------------------------------------------------
    print 
    print INDENT_L4, "=" * 40 + " Majority Voting " + "=" * 40
    ensemble = models.EnsembleModel(train_scenes, learners)
    ensemble.recognize(ensemble_tuples)
    draw_confusion_matrix(ensemble.cm, xtick_labels, ytick_labels)

def draw_barh(learners, dt, ytick_label, xlabel):
    ## Draw precision bar chart
    fontsize_labels = 15
    mpl.rc('figure.subplot', left=0.1, right=0.97, top=0.95)
    mpl.rc('figure', figsize=(6.12, 5.14))
    mpl.rc('xtick', labelsize=fontsize_labels)
    mpl.rc('ytick', labelsize=fontsize_labels)
    # Add colors
    mpl.colors.ColorConverter.colors.update(dict(
            atomictangerine = (1.0, 0.6, 0.4),
            babyblueeyes = (0.63, 0.79, 0.95),
            babypink = (0.96, 0.76, 0.76),
            bluebell = (0.64, 0.64, 0.82),
            brass = (0.71, 0.65, 0.26)
        ))
    colors = ['babyblueeyes', 'babypink', 'atomictangerine', 'brass', 'blue', 'red', 'green', 'yellow', 'cyan']
    patterns = ['//', '\\\\', 'xx', '||']
    # Set bar values
    # ticks and labels
    step = 1.7
    pos = np.arange(0, len(ytick_label)*step, step)
    width = 0.3
    # Draw bar charts for each sensor 
    fig, ax = plt.subplots()
    cnt = 0
    for learner in learners:
        ax.barh(pos+cnt*width, dt[learner.getName()], width, color=colors[cnt], hatch=patterns[cnt], \
                label=learner.getName())
        cnt = cnt + 1

    ax.set(yticks=pos+(cnt/2)*width, yticklabels=ytick_label, \
            ylim=[-width, len(ytick_label)*step+step])
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.legend(ncol=2, columnspacing=0.1)
    plt.show()

def model_profiling(learner, train_data, test_data):
    print 
    print INDENT_L4, '=' * 20 + learner.getName() + ' model profiling ' + '=' * 20
    print
    if learner.setTrainVector(train_data) > 0:
        model = learner.trainNB()
        if learner.setTestVector(test_data) > 0:
            learner.scoreNB(model)
            learner.predict_profile(model)
            ## Confusion matrix
            predicts = learner.predictNB(model)
            cm = confusion_matrix(learner.targets, predicts)
            print cm
            learner.setConfusionMatrix(cm)

scene_abbr_dict = {
        'bar': 'bar',
        'cafe': 'cafe',
        'elevator': 'elev',
        'library': 'lib',
        'office': 'offi', 
        'subwaystation': 'subw'
}

def abbreviate_names(names, abbrdict):
    abbrs = []
    for name in names:
        abbrs.append(abbrdict[name])
    return abbrs

def draw_confusion_matrix(cm, xtick_labels, ytick_labels):
    fontsize_labels = 12
    mpl.rc('xtick', labelsize=fontsize_labels)
    mpl.rc('ytick', labelsize=fontsize_labels)
    # Vertical configs
    mpl.rc('figure.subplot', wspace=0, top=1, bottom=0)   

    # Horizontal configs
    mpl.rc('figure.subplot', left=0.16, right=0.97, hspace=0)

    mpl.rc('figure', figsize=(4.97, 4.04))
    fig, axes = plt.subplots()
    ## Show confusion matrix in a separate window

    im = axes.matshow(cm, cmap=plt.cm.GnBu)
    #plt.title('Confusion Matrix', fontsize=12)
    axes.set_xticklabels([''] + xtick_labels, fontsize=fontsize_labels)
    axes.set_yticklabels([''] + ytick_labels, fontsize=fontsize_labels)
    axes.set_ylabel('Actual Scene', fontsize=fontsize_labels)
    axes.set_xlabel('Predicted Scene', fontsize=fontsize_labels)

    fig.colorbar(im, ax=axes, shrink=0.8)
    ## Show figure
    plt.show()

def draw_confusion_matrixes(learners, xtick_labels, ytick_labels):
    fontsize_labels = 12
    mpl.rc('xtick', labelsize=fontsize_labels)
    mpl.rc('ytick', labelsize=fontsize_labels)
    #mpl.rc('axes', hold=False)
    mpl.rc('figure.subplot', left=0, right=1, bottom=0.08, \
            hspace=0.25, wspace=0)
    #mpl.rc('figure', figsize=(3.5, 3.5))
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    #fig.suptitle('Confusion Matrix of Sensing Models', fontsize=12)

    idx = 0
    for ax in axes.flat:
        im = ax.matshow(learners[idx].cm, cmap=plt.cm.GnBu)
        ax.set_xticklabels([""]+xtick_labels)
        ax.set_yticklabels([""]+ytick_labels)
        # Title
        ax.text(0.5, -0.1, learners[idx].getName(), fontsize=fontsize_labels, fontweight='bold',\
                transform=ax.transAxes)
        idx = idx + 1


    #cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], location='right')
    #plt.colorbar(im, cax=cax, **kw)
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


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
