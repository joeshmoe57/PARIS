#!/usr/bin/python

'''
   Fills an sqlite DB with keys... 

From Wiki: The public key consists of the modulus n and the public (or encryption) exponent e. The private key consists of the modulus n and the private (or decryption) exponent d which must be kept secret. (p, q, and mod phi(n) must also be kept secret because they can be used to calculate d.)
'''

import random
import Crypto
from Crypto.PublicKey import RSA
import ctypes
import sys
import sqlite3


'''
   Generates random numbers in some known way
'''
def randFunc(n):
   o = str(random.getrandbits(n))
   print('Randomly got: ' + o)
   return o

numKeys = 10
if len(sys.argv) > 1:
   numKeys = int(sys.argv[1])


print('About to generate a db of ' + str(numKeys) + ' 1024 bit keys')

dbConn = sqlite3.connect('keys.db')
c = dbConn.cursor()
c.execute('''DROP TABLE IF EXISTS keys''')
c.execute('''CREATE TABLE keys (blob pem)''')
"""
# Create table
c.execute('''CREATE TABLE stocks (date text, trans text, symbol text, qty real, price real)''')
# Insert a row of data
c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")
# Save (commit) the changes
conn.commit()
# We can also close the cursor if we are done with it
c.close()
"""

#setup our random number generator(this is clearly a bad one to use)
random.seed(0)

for i in range(numKeys):
   if numKeys <= 10 or i % 100 == 1:
      print('Making key ' + str(i) + ' - ' + str(100 * (float(i) / float(numKeys))) + '%')
      #this is here because i think python stores db changes until you commit, which builds up memory
      dbConn.commit()

   k = RSA.generate(1024, e=65537)
   
   pem = k.exportKey('PEM')
   c.execute("""INSERT INTO keys VALUES (?)""", [pem])

print('Finished making ' + str(i + 1) + ' keys')
dbConn.commit()
#close the db handle
c.close()
