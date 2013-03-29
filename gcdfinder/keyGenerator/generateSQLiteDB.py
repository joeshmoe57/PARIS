#!/usr/bin/python

'''
   Fills an sqlite DB with keys... 
   And its NOT THREADED. Due to an interpreter thing, threading is a farce, and
      all threads actually run inside one OS thread (See GIL). So it uses
      multiple processes.

From Wiki: The public key consists of the modulus n and the public (or encryption) exponent e. The private key consists of the modulus n and the private (or decryption) exponent d which must be kept secret. (p, q, and mod phi(n) must also be kept secret because they can be used to calculate d.)
'''

import time
import Crypto
from Crypto.PublicKey import RSA
#from Crypto import Random
import ctypes
import sys
import sqlite3
import os
import re

from multiprocessing import Process, Queue

'''
new strategy: generate RSA objects, the pull out their primes and intentionally reuse them!
RSA.generate().n/.p/.q/etc

the process which generates bad keys and incrementally used by the consumer process
to add them to the DB
'''
def GenerateBadProcess(q):
   used = {}
   r = Crypto.Random.new()
   #generate keys until the program ends
   while True:
      #make the key
      k = RSA.generate(1024, e=65537)

      #save the components we want to dupe
      if 0 not in used:
         used[0] = k.q

      #inject already used values
      k.q = used[0]

      #recalc N
      k.n = k.p * k.q

      #convert the key into a string
      pem = k.exportKey('PEM')

      #put the key in the q to get written to the DB elsewhere
      q.put(pem, block=True)


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
   print( 'Randomly got: ' + o)
   return o

numKeys = 10
numProcesses = 1
numBad = 0
badStep = 1
if len(sys.argv) == 4:
   numKeys = long(sys.argv[2])
   numProcesses = int(sys.argv[1])
   numBad = int(sys.argv[3])
   if numBad != 0:
      badStep = numKeys / numBad
else:
   print ('''Wrong num of params. #1 is number of processes, #2 is number of 
         keys, #3 is number out of the total to make vulnerable''')
   exit(1)


print ('About to generate a db of ' + str(numKeys) + ' 1024 bit keys with ' + str(numProcesses) + ' processes.\nThere will be ' + str(numBad) + ' bad keys\n')

#nuke the file if it exists
try:
   os.remove('keys.db')
except OSError:
   print ('')

#setup the file to write annotations in
annotations = open('keys-annotations-' + str(numKeys) + '-' + str(numBad) + '.txt', 'w')
annotations.write(str(numKeys) + ' total keys\n' + str(numKeys - numBad) + 
                  ' good keys\n' + str(numBad) + ' bad keys\n')
annotations.write('Indicies of bad keys follow:\n')

#build the DB connection and go
dbConn = sqlite3.connect('keys-' + str(numKeys) + '-' + str(numBad) + '.db')
c = dbConn.cursor()
c.execute('''DROP TABLE IF EXISTS keys''')
c.execute('''CREATE TABLE keys (blob pem)''')

#build out inter-thread q
#the number is the Q length. I randomly chose this...
q = Queue(150)
badQ = Queue(150)

#spin out as many processes as we need
for t in range(numProcesses):
   #t = threading.Thread(target=GeneratorThread, args=(q))
   p = Process(target=GenerateProcess, args=(q,))
   p.daemon = True;
   p.start()

#and one process to make bad keys
if numBad > 0:
   print ('making bad process')
   badKeyProc = Process(target=GenerateBadProcess, args=(badQ,))
   badKeyProc.daemon = True
   badKeyProc.start()

'''
become the consuming thread. Consume until we have the correct total, then exit.
Because all other threads are daemons, they will automatically die (and we dont
care about their data.
'''

created = 0
while created < numKeys:
   if numKeys <= 10 or created % 1000 == 1:
      print( 'Making key ' + str(created) + ' - ' + str(100 * (float(created) / float(numKeys))) + '%')
      #this is here because i think python stores db changes until you commit, which builds up memory
      dbConn.commit()

   #inject a bad key every badStep keys generated
   if numBad > 0 and created % badStep == 0:
      print( 'bad one injected at = ' + str(created))
      annotations.write(str(created) + '\n')

      pem = badQ.get(block=True)

   else:
      #get the next good key from the Q
      pem = q.get(block=True)

   #insert the new key into the DB
   c.execute("""INSERT INTO keys VALUES (?)""", [pem])

   #record keeping
   created = created + 1

dbConn.commit()

#close the db handle
c.close()

annotations.close()
