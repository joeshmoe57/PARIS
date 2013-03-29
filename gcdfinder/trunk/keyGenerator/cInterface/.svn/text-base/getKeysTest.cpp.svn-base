/*
 * Reads in 1024 bit RSA keys from the keys sub directory, and sticks them
 * into our internal 1024 bit representation.
 */

#include <stdio.h>
#include <string.h>
#include <vector>

//provides generic fmemopen for non linux systems
#include "cfmemopen.h"

#include "sqlite3.h"

//OpenSSL stuff
#include "lcrypto.h"
#include "lrsa.h"
#include "lpem.h"
#include "lbn.h"

//they key function
#include "getKeys.h"


int main(int argc, char* argv[]){
   int res;
   int toGet;

   std::vector<RSA*> allPrivateKeys;
   uint32_t allPublicKeys[100][32];

   if(argc != 3){
      puts("wrong number of args. takes db to open, num to get");
      return 0;
   }

   puts("Starting key printer...");

   toGet = atoi(argv[2]);

   //run through the DB passed in and get all they keys from it
   res = getAllKeys(argv[1], toGet, &allPrivateKeys, allPublicKeys);

   printf("Asked for %d keys, got %d\n", toGet, res);
   printf("after, vector of private keys was %d\n", (int)allPrivateKeys.size());
   printf("after, vector of public keys was %d\n", res);

   //RSA_print_fp(stdout, allPrivateKeys.at(0), 0);

   //0 is MSB for our 1024
   BIGNUM *n = allPrivateKeys.at(0)->n;
   BN_print_fp(stdout, n);
   printf("\ntop = %d and 0th is\t%02x\n", n->dmax, (uint32_t)n->d[0]);
   printf("bottom = \t\t%02x\n", allPublicKeys[0][31]);

   return 0;
}
