#!/usr/bin/env python
# File: audio_analyze.py
# Read audio data from db and analyze/classify audio feature

import sqlite3
import ast
import datetime as dt
import audio_features as af
import light_features as lf
import bluetooth_features as bf
import wifi_features as wf

#import nltk : needs tagged token, which combine a basic token value with a tag.
# refer to [http://docs.huihoo.com/nltk/0.9.5/guides/tag.html] for details
import numpy as np
from sklearn import hmm 
from sklearn.naive_bayes import GaussianNB
#: removed since 0.17 release of scikit-learn
#from hmmlearn import hmm
import math
import os
from optparse import OptionParser
import ast
 
## Funf framework 
FUNFSENS_ROOT = "/home/shuang/workspace/funfsens/"
SCRIPT_PATH = FUNFSENS_ROOT + "scripts-0.2.3"
DATA_PATH = FUNFSENS_ROOT + "myserver/uploads"

import sys
sys.path.insert(0, SCRIPT_PATH + "/data_processing")
import decrypt, dbdecrypt, dbmerge


# funf database table 
DES_key = 'changeme'

# Parameters
INDENT_L1 = ' ' * 1
INDENT_L2 = ' ' * 2
INDENT_L4 = ' ' * 4

def main():
    ## Test
    # readFeatureSequence('test', 'train')
    # return

    ## Command argument parsing
    parser = OptionParser()
    parser.add_option("-t", "--test", dest="testscene", default="test",
            help="test scene", metavar="TEST_SCENE")

    parser.add_option("-s", "--scenes", dest="scenes", default="office",
            help="train scenes", metavar="TRAIN_SCENES")

    parser.add_option("-f", "--features", dest="features", default="audio light",
            help="", metavar="FEATURES")

    (options, args) = parser.parse_args()
    

    print INDENT_L1, '+--------------------------------------------------------------------+'
    print INDENT_L1, '| AMSC : Automatic Mobile Scene Classification                       |' 
    print INDENT_L1, '| Probes: audio                                                      |'
    print INDENT_L1, '| Author: Shuang Liang <shuang.liang2012@temple.edu>                 |'
    print INDENT_L1, '+--------------------------------------------------------------------+'
    ## Training
    X = []      # Observations
    # Classes
    # y = ['office', 'home', 'cafe', 'station', 'gym', 'test']
    y = options.scenes.split() #['office', 'cafe', 'home']

    ## Obtain features from the training samples
    features = options.features.split()
    for scene in y:
        V = []  # Feature vector
        #  Audio
        if('audio' in features):
            mfccs = readAudioFeature('train', scene)
            # Take the average of mfccs of audio frames as features
            mfcc = np.mean(mfccs, axis=0)
            V = V + mfcc.tolist()

        # Light
        if('light' in features):
            luxs = readLightFeature('train', scene)
            lux = np.mean(luxs, axis=0)
            V = V + [lux]

        # Bluetooth
        if('bluetooth' in features):
            mAddresses = readBluetoothFeature('train', scene)
            print mAddresses

        # Wifi feature
        if('wifi' in features):
            bssids = readWifiFeature('train', scene)
            print bssids

        # Add observation for scene
        if len(V) > 0:
            X.append(np.array(V))       

    if len(X) == 0:
        print 'No feature selected'
    else:
        print X

    # Gaussian Naive Bayes
    print INDENT_L2, ">> Training Gaussian Naive Bayes <<"
    model = trainNB(np.array(X), y)

    ## Test ##
    print INDENT_L2, ">> Testing Gaussian Naive Bayes Model <<"
    usage = 'train'
    scene = options.testscene
    V = []  # Feature vector
    X = []
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

    # Add observation for scene
    if len(V) > 0:
        X.append(np.array(V))
        print X
        X = np.array(X)
        print INDENT_L4, "Shape of data feed to model: ", X.shape

    print INDENT_L4, '>> Test %s with model for [%s], probability: [%s]' % (scene, \
            ', '.join(sorted(y)), ', '.join(str(x) for x in model.predict_proba(X)[0]))

    #score = model.score(X, [scene])
    #print INDENT_L1, "With Gaussian Naive Bayes, the score is %f" % (score)

    ## Gaussian Hidden Markov Model
    '''
    mfccs = readFeatureSequence("home")
    X = np.array([mfccs])
    print X.shape
    trainHMM(X)
    '''
    print
#-------------------------------------------------------------------------------
# Decrypt and merge DB segments
# Return the path of merged DB
#-------------------------------------------------------------------------------
TABLE_NAME = 'data'
def fetchDB(usage='train', feature='audio', scene='office'):
    print INDENT_L4, ">> Fetching DB for %s/%s/%s" % (usage, feature, scene)
    TRAIN_DB_PATH = DATA_PATH + '/' +  usage + '/' + feature + '/' + scene + '/'
    TRAIN_DB_NAME = "merged_" + usage + "_" + feature + '_' + scene + ".db"
    if not os.path.exists(TRAIN_DB_PATH + TRAIN_DB_NAME):
        ## Decrypt database segments 
        for db_seg in os.listdir(TRAIN_DB_PATH):
            if db_seg.endswith('.db') and not db_seg.startswith('merged'):
                dbdecrypt.decrypt_if_not_db_file(TRAIN_DB_PATH+db_seg, DES_key)
        ## Merge db segments
        db_files = [TRAIN_DB_PATH+file for file in os.listdir(TRAIN_DB_PATH) if file.endswith('.db') and not file.startswith('merged')]
        dbmerge.merge(db_files, TRAIN_DB_PATH+TRAIN_DB_NAME)
        #print db_files
    #else:
        #print "%s already exists!" % (TRAIN_DB_OFFICE)

    db = TRAIN_DB_PATH + TRAIN_DB_NAME
    return db

#-------------------------------------------------------------------------------
# Read Bluetooth features from DB
#-------------------------------------------------------------------------------
def readBluetoothFeature(usage, scene):
    ## Connect to Sqlite3 DB
    conn = sqlite3.connect(fetchDB(usage, 'bluetooth', scene))
    cur = conn.cursor()
    ## Extract Bluetoth feature
    cur.execute("SELECT * FROM " + TABLE_NAME)
    bluetooth_features = []
    for rec in cur:
        bluetooth = bf.BluetoothFeatures(rec)
        feature = (bluetooth.getDeviceName(), bluetooth.getDeviceAddress())
        bluetooth_features.append(feature[1])
    conn.close()
    #print bluetooth_features
    print INDENT_L4, "Total # of bluetooth features extracted from %s: %d" % (scene, len(bluetooth_features))
    return list(set(bluetooth_features))
    
#-------------------------------------------------------------------------------
# Read Wifi features from DB
#-------------------------------------------------------------------------------
def readWifiFeature(usage, scene):
    ## Connect to Sqlite3 DB
    conn = sqlite3.connect(fetchDB(usage, 'wifi', scene))
    cur = conn.cursor()
    ## Extract light feature
    cur.execute("SELECT * FROM " + TABLE_NAME)
    wifi_features = []
    for rec in cur:
        wifi = wf.WifiFeatures(rec)
        feature = (wifi.getSSID(), wifi.getBSSID())
        wifi_features.append(feature[1])
    conn.close()
    #print wifi_features
    print INDENT_L4, "Total # of wifi features extracted from %s: %d" % (scene, len(wifi_features))
    return list(set(wifi_features))
#-------------------------------------------------------------------------------
# Read light feature from DB
#-------------------------------------------------------------------------------
def readLightFeature(usage, scene):
    ## Connect to Sqlite3 DB
    conn = sqlite3.connect(fetchDB(usage, 'light', scene))
    cur = conn.cursor()
    ## Extract light feature
    cur.execute("SELECT * FROM " + TABLE_NAME)
    audio_features = []
    for rec in cur:
        light = lf.LightFeatures(rec)
        #print light.getLux()
        audio_features.append(light.getLux())
    conn.close()
    print INDENT_L4, "Total # of light features extracted from %s: %d" % (scene, len(audio_features))

    return audio_features
#-------------------------------------------------------------------------------
# Read audio feature from DB
#-------------------------------------------------------------------------------
def readAudioFeature(usage, scene):
    ## SQLite3 database
    conn = sqlite3.connect(fetchDB(usage, 'audio', scene))
    cur = conn.cursor()

    ## Fetch and extract features
    train_seqs = []
    cur.execute("SELECT * FROM " + TABLE_NAME)
    for rec in cur:
       audio = af.AudioFeatures(rec) 
       train_seqs.append(audio.getMfccs())

    conn.close()

    print INDENT_L4, "Total # of mfcc features extracted from %s: %d" % (scene, len(train_seqs))
    return train_seqs

#-------------------------------------------------------------------------------
# Train Naive Bayes
#-------------------------------------------------------------------------------
def trainNB(X, y):
    model = GaussianNB()
    print INDENT_L4, "Shape of data feed to classifier: ", X.shape
    model.fit(X, y)
    return model

#-------------------------------------------------------------------------------
# Train Hiden Markov Model
#-------------------------------------------------------------------------------
def trainHMM(X):
    ## HMM
    # Use unsupervised Baum-Welch to train HMM for different locations/scenes
    # 3-states, full covariance 
    model = hmm.GaussianHMM(3, "full")
    # One obervation with length len(X[0]), n_feature = 12
    #print train_seqs
    #print X[0].shape
    model.fit(X)
    #states = model.predict(X[0])
    #print "Most likely state sequences: \n%s" % states
    score = model.score(X[0])
    print "Observation score for the same data: %f" % score#math.exp(score*math.pow(10, -6))
    return
    #--------------------------------------------------------------------------- 
    # Test
    #---------------------------------------------------------------------------
    ## Model for 'office'
    TRAIN_PATH_OFFICE = DATA_PATH + '/train/test/'
    TRAIN_DB_OFFICE = "merged_train_test.db"
    if not os.path.exists(TRAIN_PATH_OFFICE + TRAIN_DB_OFFICE):
        ## Decrypt database 
        for db_seg in os.listdir(TRAIN_PATH_OFFICE):
            if db_seg.endswith('.db') and not db_seg.startswith('merged'):
                dbdecrypt.decrypt_if_not_db_file(TRAIN_PATH_OFFICE+db_seg, DES_key)
        ## Merge db segments
        db_files = [TRAIN_PATH_OFFICE+file for file in os.listdir(TRAIN_PATH_OFFICE) if file.endswith('.db') and not file.startswith('merged')]
        dbmerge.merge(db_files, TRAIN_PATH_OFFICE+TRAIN_DB_OFFICE)
        print db_files
    else:
        print "%s already exists!" % (TRAIN_DB_OFFICE)

    ## SQLite3 database
    db_path = TRAIN_PATH_OFFICE + TRAIN_DB_OFFICE
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    ## Fetch and extract data
    cur.execute("SELECT * FROM " + TABLE_NAME)

    train_seqs = []
    #rec = cur.fetchone()
    for rec in cur:
        audio = af.AudioFeatures(rec) 
        train_seqs.append(audio.getMfccs())
    score = model.score(np.array(train_seqs))
    #print "Observation score: ", score
    print "Observation score: %f" % math.exp(score*math.pow(10, -6))

    print 'Train finished'


if __name__ == "__main__":
    main()
