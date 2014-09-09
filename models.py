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

import pandas
from collections import defaultdict

class Model:
    def __init__(self, classes, name, weight):
        self.classes = classes
        self.name = name
        self.weight = weight

    def getName(self):
        return self.name
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

    def predict_profile(self, model):

        ##
        scene_false = dict.fromkeys(self.classes, 0)
        scene_true = dict.fromkeys(self.classes, 0)
        

        predicted_targets = model.predict(self.testvector)
        idx = 0
        for target in self.targets:
            if target == predicted_targets[idx]:
                scene_true[target] = scene_true[target] + 1
            else:
                scene_false[target] = scene_false[target] + 1
                #self.weight = self.weight * 0.5
            idx = idx + 1

        #total = len(predicted_targets)
        ## Scene false as punishment
        '''
        for key in scene_false.keys():
            print "%s: %d / %d" % (key, scene_false[key], total)
            scene_false[key] = float(scene_false[key]) / total
        self.scene_false_rate = scene_false
        print self.scene_false_rate
        '''
        ## Scene true as reward
        for key in scene_true.keys():
            total = scene_true[key] + scene_false[key]
            print "%s: %d - %d / %d" % (key, scene_true[key], scene_false[key], total)
            if total == 0:
                scene_true[key]
            else:
                scene_true[key] = float(scene_true[key]) / total
        self.scene_true_rate = scene_true 
        print self.scene_true_rate

        ## Model score
        self.score = model.score(self.testvector, self.targets)

        ## Model weight
        self.weight = self.weight * self.score



    def vote(self, predict):
        return self.scene_true_rate[predict] * self.weight

    def scoreNB(self, model):
       return model.score(self.testvector, self.targets)

    def predict_probaNB(self, model):
        return model.predict_proba(self.testvector)

    def testNB(self, model):
        predicted_targets = self.predictNB(model)
        estimate_probs = self.predict_probaNB(model)
        mean_score = self.scoreNB(model)
        print '=' * 20 + " Test result for naive bayes model " + '=' * 20
        # Prediction result for test vectors
        print 'Predicted target: ', predicted_targets, ' for ', self.targets

        # Confusion matrix
        print INDENT_L4, '-' * 80
        '''
        ob_idx = 0
        print '  '.join(sorted(self.classes))
        for target in self.targets:
            print target + '  ' + '  '.join(str(i) for i in estimate_probs[ob_idx])
            ob_idx = ob_idx + 1
        '''
        print pandas.DataFrame(estimate_probs, self.targets, sorted(self.classes))
        print INDENT_L4, '-' * 80
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
        Model.__init__(self, classes, 'audio', 1)

    def setTrainVector(self, values):
        X = []
        y = []
        for scene, obs in values: 
            X.append(np.mean(obs, axis=0))   # Append observation
            y.append(scene)         # Append target

        if len(X) == 0:
            print 'No train vector for audio model'
            return 0

        self.trainvector = np.array(X)
        self.labels = y
        #print INDENT_L4, "Shape of the train vector", self.trainvector.shape, \
        #        'for ' + ', '.join(self.labels)
        return len(self.trainvector)

    def setTestVector(self, scene_obs):
        T = []
        targets = []
        for scene, obs in scene_obs:
            T.append(np.mean(obs, axis=0))
            targets.append(scene)
        if len(T) == 0:
            print 'No test vector for audio model'
            return 0

        self.testvector = np.array(T)
        self.targets = targets
        #print INDENT_L4, "Shape of test vector", self.testvector.shape, 'for ' + ', '.join(targets)
        #print self.testvector
        return len(self.testvector)


class LightModel(Model):
    def __init__(self, classes):
        Model.__init__(self, classes, 'light', 1)

    def setTrainVector(self, values):
        X = []
        y = []
        for scene, obs in values: 
            X.append(np.mean(obs, axis=0))   # Append observation
            y.append(scene)         # Append target

        if len(X) == 0:
            print 'No train vector for light model'
            return 0

        self.trainvector = np.array(X)
        self.labels = y
        #print INDENT_L4, "Shape of the train vector", self.trainvector.shape, \
        #        'for ' + ', '.join(self.labels)
        return len(self.trainvector)

    def setTestVector(self, scene_obs):
        T = []
        targets = []
        for scene, obs in scene_obs:
            T.append(np.mean(obs, axis=0))
            targets.append(scene)
        if len(T) == 0:
            print 'No test vector for light model'
            return 0

        self.testvector = np.array(T)
        self.targets = targets
        #print INDENT_L4, "Shape of test vector", self.testvector.shape, 'for ' + ', '.join(targets)
        #print self.testvector
        return len(self.testvector)

class WifiModel(Model):
    def __init__(self, classes):
        Model.__init__(self, classes, 'wifi', 1)


    def setTrainVector(self, values):
        X = []
        y = []
        for scene, obs in values: 
            X.append(np.mean(obs, axis=0))   # Append observation
            y.append(scene)         # Append target
        if len(X) == 0:
            print 'No train vector for wifi model'
            return 0

        self.trainvector = np.array(X)
        self.labels = y
        #print INDENT_L4, "Shape of the train vector", self.trainvector.shape, \
        #        'for ' + ', '.join(self.labels)
        return len(self.trainvector)

    def setTestVector(self, scene_obs):
        T = []
        targets = []
        for scene, obs in scene_obs:
            T.append(np.mean(obs, axis=0))
            targets.append(scene)
        if len(T) == 0:
            print 'No test vector for wifi model'
            return 0

        self.testvector = np.array(T)
        self.targets = targets
        #print INDENT_L4, "Shape of test vector", self.testvector.shape, 'for ' + ', '.join(targets)
        #print self.testvector
        return len(self.testvector)

class BluetoothModel(Model):
    def __init__(self, classes):
        Model.__init__(self, classes, 'bluetooth', 1)


    def setTrainVector(self, values):
        X = []
        y = []
        for scene, obs in values: 
            X.append(np.mean(obs, axis=0))   # Append observation
            y.append(scene)         # Append target
        if len(X) == 0:
            print 'No train vector for bluetooth model'
            return 0

        self.trainvector = np.array(X)
        self.labels = y
        #print INDENT_L4, "Shape of the train vector", self.trainvector.shape, \
        #        'for ' + ', '.join(self.labels)
        return len(self.trainvector)

    def setTestVector(self, scene_obs):
        T = []
        targets = []
        for scene, obs in scene_obs:
            T.append(np.mean(obs, axis=0))
            targets.append(scene)
        if len(T) == 0:
            print 'No test vector for wifi model'
            return 0

        self.testvector = np.array(T)
        self.targets = targets
        #print INDENT_L4, "Shape of test vector", self.testvector.shape, 'for ' + ', '.join(targets)
        #print self.testvector
        return len(self.testvector)

