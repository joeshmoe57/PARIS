#!/usr/bin/python

'''
   Fills an sqlite DB with keys... 
   And its NOT THREADED. Due to an interpreter thing, threading is a farce, and
      all threads actually run inside one OS thread (See GIL). So it uses
      multiple processes.

From Wiki: The public key consists of the modulus n and the public (or encryption) exponent e. The private key consists of the modulus n and the private (or decryption) exponent d which must be kept secret. (p, q, and mod phi(n) must also be kept secret because they can be used to calculate d.)
'''

import time
import random
import Crypto
from Crypto.PublicKey import RSA
import ctypes
import sys
import sqlite3

from multiprocessing import Process, Queue

'''
   This is a thread, of which there are many, which generates RSA keys and
   send them to the writing thread for output.
'''
def GenerateProcess(q):
   #generate keys until the program ends
   while True:
      #make the key
      k = RSA.generate(1024, e=65537)

      #convert the key into a string
      pem = k.exportKey('PEM')

      #put the key in the q to get written to the DB elsewhere
      q.put(pem, block=True)



'''
   Generates random numbers in some known way
'''
def randFunc(n):
   o = str(random.getrandbits(n))
   print 'Randomly got: ' + o
   return o

numKeys = 10
numProcesses = 1
if len(sys.argv) > 2:
   numKeys = long(sys.argv[2])
   numProcesses = int(sys.argv[1])
else:
   print 'Wrong num of params. #1 is number of processes, #2 is number of keys'
   exit(1)


print 'About to generate a db of ' + str(numKeys) + ' 1024 bit keys with ' + str(numProcesses) + ' processes'

dbConn = sqlite3.connect('keys.db')
c = dbConn.cursor()
c.execute('''DROP TABLE IF EXISTS keys''')
c.execute('''CREATE TABLE keys (blob pem)''')

#setup our random number generator(this is clearly a bad one to use)
random.seed(0)

#build out inter-thread q
#the number is the Q length. I randomly chose this...
q = Queue(150)

#spin out as many processes as we need
for t in range(numProcesses):
   #t = threading.Thread(target=GeneratorThread, args=(q))
   p = Process(target=GenerateProcess, args=(q,))
   p.daemon = True;
   p.start()

'''
become the consuming thread. Consume until we have the correct total, then exit.
Because all other threads are daemons, they will automatically die (and we dont
care about their data.
'''

created = 0
while created < numKeys:
   if numKeys <= 10 or created % 100 == 1:
      print 'Making key ' + str(created) + ' - ' + str(100 * (float(created) / float(numKeys))) + '%'
      #this is here because i think python stores db changes until you commit, which builds up memory
      dbConn.commit()

   #get the next key from the Q
   pem = q.get(block=True)


   #insert the new key into the DB
   c.execute("""INSERT INTO keys VALUES (?)""", [pem])

   #record keeping
   #print pem
   created = created + 1


dbConn.commit()

#close the db handle
c.close()
