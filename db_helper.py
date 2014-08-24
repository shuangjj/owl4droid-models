#!/usr/bin/env python
# File: db_healper.py
# Healper functions for accessing sensor databases

from constants import *
import os
import sqlite3
## Funf framework path config
FUNFSENS_ROOT = "/home/shuang/workspace/funfsens/"
SCRIPT_PATH = FUNFSENS_ROOT + "scripts-0.2.3/"
DATA_PATH = SCRIPT_PATH + "myserver/uploads"

import sys
sys.path.insert(0, SCRIPT_PATH + "/data_processing")
import decrypt, dbdecrypt, dbmerge
# funf database configs
DES_key = 'changeme'
TABLE_NAME = 'data'

class DBHelper:
    def __init__(self, path):
        self.dbDir = path


    def __init__(self, usage='train', feature='audio', scene='office', data_root=DATA_PATH):
        self.data_root = data_root
        self.usage = usage
        self.feature = feature
        self.scene=scene
    #---------------------------------------------------------------------------
    # Decrypt and merge database files
    # Return name of the merged database
    #---------------------------------------------------------------------------
    def fetchDB(self):
        print INDENT_L4, ">> Fetching DB for %s/%s/%s" % (self.usage, self.feature, self.scene)
        DB_PATH = self.data_root + '/' +  self.usage + '/' + self.feature + '/' + self.scene + '/'
        DB_NAME = "merged_" + self.usage + "_" + self.feature + '_' + self.scene + ".db"
        if not os.path.exists(DB_PATH + DB_NAME):
            ## Decrypt database segments 
            for db_seg in os.listdir(DB_PATH):
                if db_seg.endswith('.db') and not db_seg.startswith('merged'):
                    dbdecrypt.decrypt_if_not_db_file(DB_PATH+db_seg, DES_key)
            ## Merge db segments
            db_files = [DB_PATH+file for file in os.listdir(DB_PATH) if file.endswith('.db') and not file.startswith('merged')]
            dbmerge.merge(db_files, DB_PATH+DB_NAME)
            #print db_files
        #else:
            #print "%s already exists!" % (TRAIN_DB_OFFICE)

        self.dbDir = DB_PATH + DB_NAME
        return self.dbDir
    #---------------------------------------------------------------------------
    # Fetch all data from data table
    #---------------------------------------------------------------------------
    def fetchData(self):
        ## Connect to Sqlite3 DB
        self.conn = sqlite3.connect(self.dbDir,detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES )
        cur = self.conn.cursor()
        ## Extract Bluetoth feature
        cur.execute("SELECT * FROM " + TABLE_NAME)
        return cur

    def closeDB(self):
        self.conn.close()


