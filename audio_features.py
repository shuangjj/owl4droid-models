#!/usr/bin/env python
# File: audio_features.py
# Class file for audio record archived by AudioFeaturesProbe

import ast
import datetime as dt

class AudioFeatures:
    
    def __init__(self, rec):
    # Columns: data (id text, device text, probe text, timestamp long, value text
        self.data_id = rec[0]
        self.dev_id = rec[1]
        self.probe = rec[2]
        self.timestamp = rec[3]
        self.values = rec[4]
        # Convert string values to python map style feature set
        self.features = ast.literal_eval(self.values)
#-------------------------------------------------------------------------------
# Audio features getters
#-------------------------------------------------------------------------------
    def getTime(self):
        return dt.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')   

    def getFeatures(self):
        return self.features

    def getMfccs(self):
        return self.features["mfccs"]

    def getDiffSecs():
        return self.features["diffSecs"]

    def getL1Norm(self):
        return self.features["l1Norm"]

    def getL2Norm(self):
        return self.features["l2Norm"]

    def getLinfNorm(self):
        return self.features["linfNorm"]

    def getPsdAcrossFrequencyBands(self):
        return self.featurns["psdAcrossFrequencyBands"]


