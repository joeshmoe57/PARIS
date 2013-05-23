#include <utility>
#include <vector>
#include <set>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <sys/resource.h>
#include <stdlib.h>
#include "hrt.h"

#include "lcrypto.h"
#include "lrsa.h"
#include "lpem.h"
#include "lbn.h"

//#define KEYS_DB "keys-16-2.db"
//#define KEYS_DB "../../../keys-2000000-4000.db"
#define KEYS_DB "keys-2000000-4000.db"

#define PRIVATE_KEY_DIR "./private_keys/"

#define NUM_INTS 32

#define DEBUG 0

typedef std::pair<unsigned long long, unsigned long long> keyPair;
typedef std::vector<keyPair> keyPairList;
typedef std::set<unsigned long long> keySet;
typedef std::set<unsigned long long>::iterator keySetIter;

void writeKeyToPEM(uint32_t * modulus);

void processBadKeys(keyPairList & badKeyPairList, uint32_t * moduli,
      int parallel, unsigned long long numKeys) {

   char out[80];
   memset(out, 0, 80);
   sprintf(out, "bad-key-list_%s_%llu_", (parallel == 1 ? "CUDA" : "seq"), numKeys);

   strcat(out, KEYS_DB);

   FILE * outfp;
   if ((outfp = fopen(out, "w")) == NULL) {
         fprintf(stderr, "Error with file: %s\n", out);
         perror("Error creating output file.");
         exit(-1);
   }

   if (badKeyPairList.size() > 0) {
      fprintf(outfp,
         "PARIS found %ld vulnerable pairs of keys in the list your provided in %s.\nTheir paired indices follow, with one pair per line, separated by a \'|\'\n",
         badKeyPairList.size(), KEYS_DB);
   }

   for (unsigned long long i = 0; i < badKeyPairList.size(); ++i) {
      writePEMFileForKeyPair(moduli + badKeyPairList[i].first * NUM_INTS,
            moduli + badKeyPairList[i].second * NUM_INTS)
      fprintf(outfp, "%llu | %llu\n", badKeyPairList[i].first,
            badKeyPairList[i].second);
   }
}

void writeKeyToPEM(uint32_t * modulusA, uint32_t * modulusB) {
   RSA * RSAkey = RSA_new();
   // use modulus for "n"


   // e is known

}
