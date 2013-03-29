#!/usr/bin/python

'''
   Generates a file of the specified number of 1024 bit RSA keys.

From Wiki: The public key consists of the modulus n and the public (or encryption) exponent e. The private key consists of the modulus n and the private (or decryption) exponent d which must be kept secret. (p, q, and mod phi(n) must also be kept secret because they can be used to calculate d.)
'''
import sys
import pickle
import random
import Crypto
from Crypto.PublicKey import RSA
import ctypes





PyLong_AsByteArray = ctypes.pythonapi._PyLong_AsByteArray
PyLong_AsByteArray.argtypes = [ctypes.py_object,
                               ctypes.c_char_p,
                               ctypes.c_size_t,
                               ctypes.c_int,
                               ctypes.c_int]

def packl_ctypes(lnum):
   a = ctypes.create_string_buffer(lnum.bit_length()//8 + 1)
   PyLong_AsByteArray(lnum, a, len(a), 0, 1)
   return a.raw


if(len(sys.argv) != 2):
   print('pass the file to consume as the only arg')
   exit()


inKeys = open(sys.argv[1], 'r')
keys = pickle.load(inKeys)


outBin = open(str(len(keys)) + 'keysBin.bin', 'w')

print(('Got ' + str(len(keys)) + ' keys to process'))

#outBin.write(str(len(packl_ctypes(keys[0].n))) + '\n')

for k in keys:
   print(k)
   outBin.write(packl_ctypes(k.n))
   #print packl_ctypes(k.n)
   print((k.n))
   print((len(packl_ctypes(keys[0].n))))

