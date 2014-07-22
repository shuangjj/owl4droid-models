#!/usr/bin/env python
# File: audio_analyze.py
# Read audio data from db and analyze/classify audio feature

import sqlite3
import ast
import datetime as dt
import audio_features as af
#import nltk : needs tagged token, which combine a basic token value with a tag.
# refer to [http://docs.huihoo.com/nltk/0.9.5/guides/tag.html] for details
import numpy as np
from sklearn import hmm 
from sklearn.naive_bayes import GaussianNB
#: removed since 0.17 release of scikit-learn
#from hmmlearn import hmm
import math
import os
 
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
INDENT_T1 = '\t\t'
INDENT_L1 = '\t'

def main():
    ## Training
    X = []
    y = ['office', 'cafe']
    mfccs_office = readFeatureSequence(y[0])
    mfccs_cafe = readFeatureSequence(y[1])
    X.append(np.mean(mfccs_office, axis=0))
    X.append(np.mean(mfccs_cafe, axis=0))
    # Gaussian Naive Bayes
    print INDENT_T1, ">> Training Gaussian Naive Bayes <<"
    model = trainNB(np.array(X), y)
    # Test
    print INDENT_T1, ">> Testing NB model <<"
    scene = 'home'
    mfccs = readFeatureSequence(scene)
    X = np.array([np.mean(mfccs, axis=0)])
    print INDENT_L1, 'Test %s with result: ' % scene, sorted(y), model.predict_proba(X)

    #score = model.score(X, [scene])
    #print INDENT_L1, "With Gaussian Naive Bayes, the score is %f" % (score)

def readFeatureSequence(scene):
    print INDENT_L1, ">> Extracting data from ", scene
    TABLE_NAME = 'data'
    TRAIN_DB_PATH = DATA_PATH + '/train/' + scene + "/"
    TRAIN_DB_NAME = "merged_train_" + scene + ".db"
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

    ## SQLite3 database
    db = TRAIN_DB_PATH + TRAIN_DB_NAME
    conn = sqlite3.connect(db)
    cur = conn.cursor()

    ## Fetch and extract data
    train_seqs = []
    cur.execute("SELECT * FROM " + TABLE_NAME)
    for rec in cur:
        audio = af.AudioFeatures(rec) 
        train_seqs.append(audio.getMfccs())
    print INDENT_L1, "Total # of mfcc features extracted from %s: %d" % (scene, len(train_seqs))
    return train_seqs

#-------------------------------------------------------------------------------
# Train Naive Bayes
#-------------------------------------------------------------------------------
def trainNB(X, y):
    model = GaussianNB()
    print INDENT_L1, "Shape of data feed to classifier: ", X.shape
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
    '''
    X = np.array(train_seqs)
    X = np.rot90(X, 1)
    X = np.flipud(X)
    X = np.array([X])
    print X[0].shape
    '''
    # One obervation with length len(X[0]), n_feature = 12
    #print train_seqs
    #X = np.array([train_seqs])
    print X.shape
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
