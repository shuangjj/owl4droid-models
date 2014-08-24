#!/usr/bin/env python
# File: audio_analyze.py
# Read audio data from db and analyze/classify audio feature

import ast
import datetime as dt
import numpy as np
#import nltk : needs tagged token, which combine a basic token value with a tag.
# refer to [http://docs.huihoo.com/nltk/0.9.5/guides/tag.html] for details
from sklearn import hmm 
from sklearn.naive_bayes import GaussianNB
#: removed since 0.17 release of scikit-learn
#from hmmlearn import hmm
import math
import os
from optparse import OptionParser
import ast

from constants import *
import features

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
    # Classes
    # y = ['office', 'home', 'cafe', 'station', 'gym', 'test']
    scenes = options.scenes.split() #['office', 'cafe', 'home']
    featureEnrolled = options.features.split()

    X = []      # Observations
    for scene in scenes:
        V = features.getFeatureVector('train', scene, featureEnrolled)
        # Add observation for scene
        if len(V) > 0:
            X.append(np.array(V))       

    # Gaussian Naive Bayes
    print INDENT_L2, ">> Training Gaussian Naive Bayes <<"
    model = trainNB(X, scenes)

    ## Test ##
    print INDENT_L2, ">> Testing Gaussian Naive Bayes Model <<"
    T = []
    V = features.getFeatureVector('test', options.testscene,featureEnrolled)
    if len(V) > 0:
        T.append(np.array(V))

    results = testNB(model, T)
    print INDENT_L4, '>> Test %s with model for [%s], probability: [%s]' % (options.testscene, \
            ', '.join(sorted(scenes)), ', '.join(str(x) for x in results))

    score = model.score(np.array(T), [options.testscene])
    print INDENT_L1, "With Gaussian Naive Bayes, the score is %f" % (score)

    ## Gaussian Hidden Markov Model
   # trainHMM(X)
   # testHMM(model, T)
   

#-------------------------------------------------------------------------------
# Train Naive Bayes
# X is python style two dimentional array (n_samples, n_features) of observations 
# y is the target values (n_samples)
#-------------------------------------------------------------------------------
def trainNB(X, y):
    if len(X) == 0:
        print 'No feature selected'
        return
    else:
        print X
    X = np.array(X)

    model = GaussianNB()
    print INDENT_L4, "Shape of data feed to classifier: ", X.shape
    model.fit(X, y)
    return model
#-------------------------------------------------------------------------------
# Test naive bayes model with test vector T (n_samples, n_classes/n_scenes)
# Return the probability of the samples for each class (in sorted order)
#-------------------------------------------------------------------------------
def testNB(model, T):
    if len(T) == 0:
        print 'No test feature selected'
        return
    else:
        print T
        T = np.array(T)
        print INDENT_L4, "Shape of data feed to model: ", T.shape
    return model.predict_proba(T)[0]
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
    return model
#--------------------------------------------------------------------------- 
# Test
#---------------------------------------------------------------------------
def testHMM(T):
    #states = model.predict(X[0])
    #print "Most likely state sequences: \n%s" % states
    score = model.score(T[0])
    print "Observation score for the same data: %f" % score#math.exp(score*math.pow(10, -6))

if __name__ == "__main__":
    main()
