#!/usr/bin/python

'''
   Generates a file of the specified number of 1024 bit RSA keys, then pickles it.

From Wiki: The public key consists of the modulus n and the public (or encryption) exponent e. The private key consists of the modulus n and the private (or decryption) exponent d which must be kept secret. (p, q, and mod phi(n) must also be kept secret because they can be used to calculate d.)
'''

import pickle
import random
import Crypto
from Crypto.PublicKey import RSA
import ctypes
import sys


'''
   Generates random numbers in some known way
'''
def randFunc(n):
   o = str(random.getrandbits(n))
   print 'Randomly got: ' + o
   return o

numKeys = 10
print sys.argv
if len(sys.argv) > 1:
   numKeys = long(sys.argv[1])

print numKeys
dir = 'keys/'

print 'About to generate a file of ' + str(numKeys) + ' 1024 bit keys'

#setup our random number generator(this is clearly a bad one to use)
random.seed(0)

for i in range(numKeys):
   if i % 1000 == 1:
      print 'Making key ' + str(i)

   k = RSA.generate(1024, e=65537)
   #keys.append(RSA.generate(1024, randfunc=randFunc, e=65537))
   
#   pem = k.exportKey('PEM')
   pem = k.publickey().exportKey('PEM')

   f = open(dir + str(i) + '.pem', 'w')
   f.write(pem)
   f.close()

