#!/usr/bin/env python
# File: audio_analyze.py
# Read audio data from db and analyze/classify audio feature

import sqlite3
import ast
import datetime as dt
import audio_features as af

# funf database table 
TABLE_NAME = 'data'
 

def main():
    ## SQLite3 database
    db_path = '/home/shuang/workspace/funfsens/simple_server/uploads/merged_1405366637.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    ## Fetch and extract data
    cur.execute("SELECT * FROM " + TABLE_NAME)
    rec = cur.fetchone()

    audio = af.AudioFeatures(rec) 
    print audio.getTime(), audio.getMfccs()

if __name__ == "__main__":
    main()
