#!/usr/bin/python

'''
   Fills an sqlite DB with keys... 
   And its threaded!

From Wiki: The public key consists of the modulus n and the public (or encryption) exponent e. The private key consists of the modulus n and the private (or decryption) exponent d which must be kept secret. (p, q, and mod phi(n) must also be kept secret because they can be used to calculate d.)
'''

import time
import random
import Crypto
from Crypto.PublicKey import RSA
import ctypes
import sys
import sqlite3

from Queue import Queue
import threading

'''
   This is a thread, of which there are many, which generates RSA keys and
   send them to the writing thread for output.
'''
class GeneratorThread(threading.Thread):
   outQueue = None

   #takes the q to write to, and the number of keys to generate
   def __init__(self, q):
      threading.Thread.__init__(self)
      self.outQueue = q

   def run(self):
      time.sleep(0.001)
      #generate keys until the program ends
      while True:
         #make the key
         k = RSA.generate(1024, e=65537)
   
         #convert the key into a string
         pem = k.exportKey('PEM')

         #put the key in the q to get written to the DB elsewhere
         self.outQueue.put(pem, block=True)







'''
   Generates random numbers in some known way
'''
def randFunc(n):
   o = str(random.getrandbits(n))
   print('Randomly got: ' + o)
   return o

numKeys = 10
numThreads = 1
if len(sys.argv) > 2:
   numKeys = int(sys.argv[2])
   numThreads = int(sys.argv[1])
else:
   print('Wrong num of params. #1 is number of threads, #2 is number of keys')
   exit(1)


print('About to generate a db of ' + str(numKeys) + ' 1024 bit keys with ' + str(numThreads) + ' threads')

dbConn = sqlite3.connect('keys.db')
c = dbConn.cursor()
c.execute('''DROP TABLE IF EXISTS keys''')
c.execute('''CREATE TABLE keys (blob pem)''')

#setup our random number generator(this is clearly a bad one to use)
random.seed(0)

#build out inter-thread q
#the number is the Q length. I randomly chose this...
q = Queue(150)

#spin out as many threads as we need
for t in range(numThreads):
   print('making thread ' + str(t))
   #t = threading.Thread(target=GeneratorThread, args=(q))
   t = GeneratorThread(q)
   t.daemon = True;
   t.start()

'''
become the consuming thread. Consume until we have the correct total, then exit.
Because all other threads are daemons, they will automatically die (and we dont
care about their data.
'''

print('about to start eating from the q')
created = 0
while created < numKeys:
   if numKeys <= 10 or created % 100 == 1:
      print('Making key ' + str(created) + ' - ' + str(100 * (float(created) / float(numKeys))) + '%')
      #this is here because i think python stores db changes until you commit, which builds up memory
      dbConn.commit()
      print(str(q.qsize()))

   #get the next key from the Q
   pem = q.get(block=True)


   #insert the new key into the DB
   c.execute("""INSERT INTO keys VALUES (?)""", [pem])

   #record keeping
   #print pem
   created = created + 1


dbConn.commit()

print('Finished making  keys')

#close the db handle
c.close()
