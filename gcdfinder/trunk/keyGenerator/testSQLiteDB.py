#!/usr/bin/python

'''
   just prints out all the PEM blobs from the db...
'''

import sqlite3

dbConn = sqlite3.connect('keys.db')
c = dbConn.cursor()
c.execute("""SELECT * FROM keys""")

for l in c:
   print(l)
   print('\n')
