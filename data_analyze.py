#!/usr/bin/env python
# File: audio_analyze.py
# Read audio data from db and analyze/classify audio feature

import os
from optparse import OptionParser
import numpy as np

# Local definitions
from features_from_db import *
from utils import *
import models

# Drawing packages
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import matplotlib as mpl
import pandas
## Drawing switches
BAR_FOR_PRECISION = True
BAR_FOR_RECALL = True

CM_FOR_SUBMODEL = False
CM_FOR_SUBMODELS = True
CM_FOR_ENSEMBLE = True

SAVE_FIGURE = True
SHOW_FIGURE = False

FIGURE_FORMAT = 'eps'   # Optional 'png'

# Verbose
VERBOSE_PROFILING = False
VERBOSE_ENSEMBLE = False

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

    train_tuples = enumerateAllSamples('train', train_scenes, sensors[0], sensors[1:], 20)
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


    test_tuples = enumerateAllSamples('test', test_scenes, sensors[0], sensors[1:], 10)
    print "# of test tuples: ", len(test_tuples), 'for ', ' , '.join(test_scenes)

    ## Assign test tuples to profile and ensemble tuples
    profile_tuples = []
    ensemble_tuples = []
    profile_scenes = [] # Scenes append to profile tuples
    ensemble_scenes = [] # Scenes append to ensemble tuples
    # Distribute test tuples for profileing and ensembling based on scenes in half
    for sample_tuple in test_tuples:
        if profile_scenes.count(sample_tuple[0]) > ensemble_scenes.count(sample_tuple[0]):
            ensemble_tuples.append(sample_tuple)
            ensemble_scenes.append(sample_tuple[0])
        else:
            profile_tuples.append(sample_tuple)
            profile_scenes.append(sample_tuple[0])
    print "# of profile tuples", len(profile_tuples)
    print "# of ensemble tuples: ", len(ensemble_tuples)

    ## Collect profiling dataset
    test_light = [];    test_audio = []
    test_wifi = [];     test_bluetooth = [] 
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
        # Draw confusion matrix for submodel individually
        if CM_FOR_SUBMODEL:
            draw_confusion_matrix(learner.cm, abbreviate_names(train_scenes, scene_abbr_dict), \
                    abbreviate_names(train_scenes, scene_abbr_dict), "figs/" + \
                    FIGURE_FORMAT + "/cm_"+learner.getName()+"." + FIGURE_FORMAT)


    ## Setup data table for precision and recall
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

    ## Draw bar diagrams
    if BAR_FOR_PRECISION:
        draw_barh(learners, dt_precision, abbreviate_names(train_scenes, scene_abbr_dict), 'Precision')

    if BAR_FOR_RECALL:
        draw_barh(learners, dt_recall, abbreviate_names(train_scenes, scene_abbr_dict), 'Recall')

    xtick_labels = abbreviate_names(train_scenes, scene_abbr_dict)
    ytick_labels = abbreviate_names(train_scenes, scene_abbr_dict)

    ## Draw confusion matrix altogether
    if CM_FOR_SUBMODELS:
        draw_confusion_matrixes(learners, xtick_labels, ytick_labels)
    ##------------------------------------------------------------------------------------
    #                     Mixed model / majority voting
    ##------------------------------------------------------------------------------------
    print 
    #print INDENT_L4, "=" * 40 + " Majority Voting " + "=" * 40
    ensemble = models.EnsembleModel(train_scenes, learners)
    ensemble.recognize(ensemble_tuples, VERBOSE_ENSEMBLE)
    if CM_FOR_ENSEMBLE:
        draw_confusion_matrix(ensemble.cm, xtick_labels, ytick_labels, "figs/" + \
                FIGURE_FORMAT + "/cm_ensemble." + FIGURE_FORMAT)

def model_profiling(learner, train_data, test_data):
    if learner.setTrainVector(train_data) > 0:
        model = learner.trainNB()
        if learner.setTestVector(test_data) > 0:
            learner.scoreNB(model)
            learner.predict_profile(model, VERBOSE_PROFILING)
            ## Confusion matrix
            predicts = learner.predictNB(model)
            cm = confusion_matrix(learner.targets, predicts)
            #print cm
            learner.setConfusionMatrix(cm)

def draw_barh(learners, dt, ytick_label, xlabel):
    ## Draw precision bar chart
    fontsize_labels = 15
    #mpl.rc('figure.subplot', left=0.1, right=0.97, top=0.95)
    #mpl.rc('figure', figsize=(6.12, 5.14))
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
    fig, ax = plt.subplots(figsize=(6.12, 5.14))
    #fig.set_size_inches(6.12, 5.14)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.95)
    cnt = 0
    for learner in learners:
        ax.barh(pos+cnt*width, dt[learner.getName()], width, color=colors[cnt], hatch=patterns[cnt], \
                label=learner.getName())
        cnt = cnt + 1

    ax.set(yticks=pos+(cnt/2)*width, yticklabels=ytick_label, \
            ylim=[-width, len(ytick_label)*step+step])
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.legend(ncol=2, columnspacing=0.1)
    if SAVE_FIGURE:
        plt.savefig("figs/" + FIGURE_FORMAT + "/" + "barh_" + xlabel + "." + FIGURE_FORMAT)
    if SHOW_FIGURE:
        plt.show()



def draw_confusion_matrix(cm, xtick_labels, ytick_labels, savename):
    fontsize_labels = 12
    mpl.rc('xtick', labelsize=fontsize_labels)
    mpl.rc('ytick', labelsize=fontsize_labels)
    # Vertical configs
    #mpl.rc('figure.subplot', wspace=0, top=1, bottom=0)   

    # Horizontal configs
    #mpl.rc('figure.subplot', left=0.16, right=0.97, hspace=0)

    #mpl.rc('figure', figsize=(4.97, 4.04))
    fig, axes = plt.subplots(figsize=(4.97, 4.04))
    #fig.set_size_inches(4.97, 4.04)
    fig.subplots_adjust(wspace=0, top=1, bottom=0, left=0.16, right=0.97, hspace=0)
    ## Show confusion matrix in a separate window

    im = axes.matshow(cm, cmap=plt.cm.GnBu)
    #plt.title('Confusion Matrix', fontsize=12)
    axes.set_xticklabels([''] + xtick_labels, fontsize=fontsize_labels)
    axes.set_yticklabels([''] + ytick_labels, fontsize=fontsize_labels)
    axes.set_ylabel('Actual Scene', fontsize=fontsize_labels)
    axes.set_xlabel('Predicted Scene', fontsize=fontsize_labels)

    fig.colorbar(im, ax=axes, shrink=0.8)
    ## Show figure
    if SAVE_FIGURE:
        plt.savefig(savename)

    if SHOW_FIGURE:
        plt.show()

def draw_confusion_matrixes(learners, xtick_labels, ytick_labels):
    fontsize_labels = 12
    mpl.rc('xtick', labelsize=fontsize_labels)
    mpl.rc('ytick', labelsize=fontsize_labels)
    #mpl.rc('axes', hold=False)
    #mpl.rc('figure.subplot', left=0, right=1, bottom=0.08, \
            #hspace=0.25, wspace=0)
    #mpl.rc('figure', figsize=(3.5, 3.5))
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    #fig.suptitle('Confusion Matrix of Sensing Models', fontsize=12)
    fig.subplots_adjust(left=0, right=1, bottom=0.08, hspace=0.25, wspace=0)
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
    if SAVE_FIGURE:
        plt.savefig("figs/" + FIGURE_FORMAT + "/cm_submodels." + FIGURE_FORMAT)

    if SHOW_FIGURE:
        plt.show()



if __name__ == "__main__":
    main()
