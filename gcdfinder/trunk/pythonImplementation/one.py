#!/usr/bin/python



def invmodp(a, p):
   '''
   The multiplicitive inverse of a in the integers modulo p.
   Return b s.t.
   a * b == 1 mod p
   '''
   r = a
   d = 1
   for count in xrange(p):
      #d = ((p // r + 1) * d) % p
      #make d equal to count 
      d = count
      r = (d * a) % p
      if r == 1:
         break
   else:
      raise ValueError('%d has no inverse mod %d' % (a, p))
   return d

def __invmodp__test__():
   p = 101
   for i in range(1, p):
      iinv = invmodp(i, p)
      assert (iinv * i) % p == 1




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




d = 1
e = 53

phi = 43200

while 1 != (e * d) % phi:
   print d
   d = d + 1

print '=== Got D = ' + str(d) + ' ==='

#===================================
#another way, from the C file I found: d = e^-1 mod phi 
d2 = invmodp(e, phi)
print '=== Got another D as ' + str(d2) + ' ==='

res = (e * d2) % phi
print 'did it meet 1 != (e * d2) % phi? 1 == ' + str(res)

#===================================
d3 = modInvEuclid(e, phi)
print '=== Got another D as ' + str(d3) + ' ==='  
