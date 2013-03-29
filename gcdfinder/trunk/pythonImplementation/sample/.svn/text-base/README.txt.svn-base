This directory contains a sample public key, the corresponding private
key, and a message that has been encrypted with the private key.

You should be able to decrypt the message using the command-line openssl
utility like this:

  openssl rsautl -inkey private.pem -decrypt < message.bin


If you want to see the details of the keys' contents, try:

  openssl rsa -in public.pem -pubin -text -noout
  openssl rsa -in private.pem -text -noout

Note that the output from OpenSSL will be in hexadecimal!

The most fundamental difference between the public and private keys is
that the public key includes the modulus n ("modulus"), while the
private key also includes the two primes p ("prime1") and q ("prime2")
such that p×q = n.


You can also decrypt the sample message in Python using the M2Crypto
module, like this:

import M2Crypto
key = M2Crypto.RSA.load_key("private.pem")
message = open("message.bin").read()
print key.private_decrypt(message, M2Crypto.RSA.pkcs1_padding)

(Note that the Python "Crypto" module isn't ideal for this because
it doesn't properly understand the format of the message.bin file.)


Here is a Ruby example:

require 'openssl'
key = OpenSSL::PKey::RSA.new File.read 'private.pem'
print key.private_decrypt File.read 'message.bin'

