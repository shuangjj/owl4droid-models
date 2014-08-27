#!/usr/bin/env python
# File: models.py
# Models to train and test audio, light, wifi, bluetooth features

import numpy as np
#import nltk : needs tagged token, which combine a basic token value with a tag.
# refer to [http://docs.huihoo.com/nltk/0.9.5/guides/tag.html] for details
from sklearn import hmm 
from sklearn.naive_bayes import GaussianNB
#: removed since 0.17 release of scikit-learn
#from hmmlearn import hmm
import math
from constants import *

class Model:
    def __init__(self, classes):
        self.classes = classes
#-------------------------------------------------------------------------------
# Train Naive Bayes
# X is python style two dimentional array (n_samples, n_features) of observations 
# y is the target values (n_samples)
#-------------------------------------------------------------------------------
    def trainNB(self):
        model = GaussianNB()
        model.fit(self.trainvector, self.labels)
        return model
#-------------------------------------------------------------------------------
# Test naive bayes model with test vector T 
# Return the probability of the samples for each class (in sorted order)
# (n_samples, n_classes/n_scenes)
#-------------------------------------------------------------------------------

    def predictNB(self, model):
        return model.predict(self.testvector)

    def scoreNB(self, model):
       return model.score(self.testvector, self.targets)

    def predict_probaNB(self, model):
        return model.predict_proba(self.testvector)[0]

    def testNB(self, model):
        predicted_targets = self.predictNB(model)
        estimate_probs = self.predict_probaNB(model)
        mean_score = self.scoreNB(model)
        print 'Predicted target: ', predicted_targets
        print ', '.join(str(i) for i in estimate_probs) + ' for ' +  \
                ', '.join(sorted(self.classes))
        print 'Mean score: ', mean_score
 
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
# Test HMM
#---------------------------------------------------------------------------
    def testHMM(T):
        #states = model.predict(X[0])
        #print "Most likely state sequences: \n%s" % states
        score = model.score(T[0])
        print "Observation score for the same data: %f" % score#math.exp(score*math.pow(10, -6))



class AudioModel(Model):
    def __init__(self, classes):
        Model.__init__(self, classes)

    def setTrainVector(self, values):
        X = []
        y = []
        for scene, obs in values: 
            for ob in obs:
                X.append(np.mean(ob, axis=0))   # Append observation
                y.append(scene)         # Append target

        if len(X) == 0:
            print 'No train vector for audio model'
            return 0

        self.trainvector = np.array(X)
        self.labels = y
        print INDENT_L4, "Shape of the train vector", self.trainvector.shape, \
                'for ' + ', '.join(self.labels)
        return len(self.trainvector)

    def setTestVector(self, obs):
        T = []
        targets = []
        for scene, obs in obs:
            T.append(np.mean(obs[0], axis=0))
            targets.append(scene)
        if len(T) == 0:
            print 'No test vector for audio model'
            return 0

        self.testvector = np.array(T)
        self.targets = targets
        print INDENT_L4, "Shape of test vector", self.testvector.shape, 'for ' + ', '.join(targets)
        print self.testvector
        return len(self.testvector)

class LightModel(Model):
    def __init__(self, classes):
        Model.__init__(self, classes)

    def setTrainVector(self, values):
        X = []
        y = []
        for scene, obs in values: 
            for ob in obs:
                X.append([np.mean(ob)])   # Append observation
                y.append(scene)         # Append target
        if len(X) == 0:
            print 'No train vector for light model'
            return 0

        self.trainvector = np.array(X)
        self.labels = y
        print INDENT_L4, "Shape of the train vector", self.trainvector.shape, \
                'for ' + ', '.join(self.labels)
        return len(self.trainvector)

    def setTestVector(self, obs):
        T = []
        targets = []
        for scene, obs in obs:
            T.append([np.mean(obs[0])])
            targets.append(scene)
        if len(T) == 0:
            print 'No test vector for light model'
            return 0

        self.testvector = np.array(T)
        self.targets = targets
        print INDENT_L4, "Shape of test vector", self.testvector.shape, 'for ' + ', '.join(targets)
        print self.testvector
        return len(self.testvector)

    




