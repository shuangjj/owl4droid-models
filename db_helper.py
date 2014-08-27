#!/usr/bin/env python
# File: db_healper.py
# Healper functions for accessing sensor databases

import constants
import os
import sqlite3
## Funf framework path config
SCRIPT_PATH = constants.SCRIPT_PATH
DATA_PATH = SCRIPT_PATH + "myserver/uploads"

import sys
sys.path.insert(0, SCRIPT_PATH + "/data_processing")
import decrypt, dbdecrypt, dbmerge


class DBHelper:
    def __init__(self, dbname, path):
        self.dbname = dbname
        self.dbpath = path
        self.dbdir =  os.path.join(self.dbpath, self.dbname)

    def execute_db(self, execute_sql, args):
        conn = sqlite3.connect(self.dbdir)
        cur = conn.cursor()
        cur.execute(execute_sql, args)
        conn.commit()
        conn.close()

    def query_db(self, sql_query, args):
        conn = sqlite3.connect(self.dbdir)
        cur = conn.cursor()
        cur.execute(sql_query, args)
        rows = cur.fetchall()
        #affected = len(cur.fetchall())
        conn.commit()
        conn.close()
        return rows

    def closeDB(self):
        self.conn.close()

