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
from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self, classes, name, weight):
        self.classes = classes
        self.name = name
        self.weight = weight

    def getName(self):
        return self.name

    def setConfusionMatrix(self, cm):
        self.cm = cm

    def setPredictVector(self, pv):
        self.pv = pv

    def setPredictClass(self, predict):
        self.predict = predict
#-------------------------------------------------------------------------------
# Train Naive Bayes
# X is python style two dimentional array (n_samples, n_features) of observations 
# y is the target values (n_samples)
#-------------------------------------------------------------------------------
    def trainNB(self):
        model = GaussianNB()
        model.fit(self.trainvector, self.labels)
        self.model = model
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
        scene_precision = dict.fromkeys(self.classes, 0.0)
        scene_recall = dict.fromkeys(self.classes, 0.0)
        scene_FP = dict.fromkeys(self.classes, 0)
        scene_TP = dict.fromkeys(self.classes, 0)
        scene_FN = dict.fromkeys(self.classes, 0)
        

        predicted_targets = model.predict(self.testvector)
        idx = 0

        for predict in predicted_targets:
            if predict == self.targets[idx]:
                scene_TP[predict] = scene_TP[predict] + 1
            else:
                print predict, self.targets[idx]
                scene_FP[predict] = scene_FP[predict] + 1
                scene_FN[self.targets[idx]] = scene_FN[self.targets[idx]] + 1
                #self.weight = self.weight * 0.5
            idx = idx + 1

        ## Calculate precision and recall
        for key in scene_TP.keys():
            total1 = scene_TP[key] + scene_FP[key]  
            total2 = scene_TP[key] + scene_FN[key]
            if total1 == 0:
                scene_precision[key] = 0.0
            else:
                scene_precision[key] = float(scene_TP[key]) / total1 # TP/(TP+FP)

            if total2 == 0:
                scene_recall[key] = 0.0
            else:
                scene_recall[key] = float(scene_TP[key]) / total2   # TP/(TP+FN)

        self.scene_precision = scene_precision
        self.scene_recall = scene_recall
        print self.getName(), scene_precision, scene_recall

        ## Model score
        self.recognition_rate = model.score(self.testvector, self.targets)

        ## Model weight initialized to model score
        self.weight = self.weight * self.recognition_rate



    def vote(self, predict):
        return self.scene_precision[predict] * self.weight

    def scoreNB(self, model):
        self.score = model.score(self.testvector, self.targets)
        return self.score

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

# 
# Order of learners in the learners list should the same as sensor list
#
class EnsembleModel:
    def __init__(self, classes, learners):
        self.classes = classes
        self.learners = learners
        self.learnerlist = []
        for learner in learners:
            self.learnerlist.append(learner.getName())

    def recognize(self, ensemble_tuples):
        ## 
        total = 0; correct = 0
        target_classes = sorted(self.classes)

        ## Predict and ensemble 
        #sample_tuples = enumerateAllSamples('test', test_scenes, sensors[0], sensors[1:])

        targets = []; predicts = []
        for sample_tuple in ensemble_tuples:
            result_vector = np.zeros(len(target_classes))
            audio_vector = []; light_vector = []; 
            bluetooth_vector = []; wifi_vector = []

            target = sample_tuple[0]
            targets.append(target)
            idx = 1
            # Take opinions from enrolled learners
            for learner in self.learners:
                sample = [(target, sample_tuple[idx])]

                learner.setTestVector(sample)
                predict = learner.predictNB(learner.model)[0]
                learner.setPredictClass(predict)
                # predict vector for current learner
                predict_vector = []
                for t in target_classes:
                    if t == predict:
                        predict_vector.append(learner.vote(predict))
                    else:
                        predict_vector.append(0)
                learner.setPredictVector(predict_vector)
                print learner.getName(), predict_vector
                result_vector = result_vector + np.array(predict_vector)
                idx = idx + 1

            ## Ensemble opinions
            predicted_class = target_classes[result_vector.argmax()]
            predicts.append(predicted_class)
            print 
            print "Predict %s to %s" % (target, predicted_class), result_vector.tolist()
            
            # Correct & Reward learners who are right
            if predicted_class == target:
                correct = correct + 1
                for learner in self.learners:
                    if learner.predict == predicted_class:
                        learner.weight = learner.weight * 1.1
            # Wrong & Punish learners who made the wrong decision
            else:
                for learner in self.learners:
                    if learner.predict == predicted_class:
                        learner.weight = learner.weight * 0.9
                print learner.getName() + " did wrong for predicting target %s to %s" % (target, predicted_class)

            print '-' * 120

            total = total + 1
        print
        self.cm = confusion_matrix(targets, predicts)
        ## Calculate predict score
        self.score = float(correct) / total        
        print "Score of majority voting (%d in %d): %f" % (correct, total, self.score)
        print
        ## 
        print "Recognition rates of individual learners: "
        for learner in self.learners:
            print "<%s, %f>" % (learner.getName(), learner.recognition_rate),
        print; print

        print "Final voting weights for individual learners: "
        for learner in self.learners:
            print "<%s, %f>" % (learner.getName(), learner.weight),
        print


