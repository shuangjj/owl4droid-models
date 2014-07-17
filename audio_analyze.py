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
#: removed since 0.17 release of scikit-learn
#from hmmlearn import hmm
# funf database table 
TABLE_NAME = 'data'
 

def main():
    ## SQLite3 database
    db_path = '/home/shuang/workspace/funfsens/simple_server/uploads/merged_1405366637.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    ## Fetch and extract data
    cur.execute("SELECT * FROM " + TABLE_NAME)

    train_seqs = []
    #rec = cur.fetchone()
    for rec in cur:
        audio = af.AudioFeatures(rec) 
        train_seqs.append(audio.getMfccs())
    #print len(train_seqs)
    ## HMM
    # Use unsupervised Baum-Welch to train HMM for different locations/scenes
    model = hmm.GaussianHMM(3, "full")
    X = np.array(train_seqs)
    X = np.rot90(X, 1)
    X = np.flipud(X)
    X = np.array([X])
    print X[0].shape
    model.fit(X)

    print 'Train finished'


if __name__ == "__main__":
    main()
