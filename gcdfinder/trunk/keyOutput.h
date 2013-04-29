#include <utility>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <sys/resource.h>
#include <stdlib.h>

#define KEYS_DB "keys-16-2.db"
//#define KEYS_DB "../../../keys-2000000-4000.db"
//#define KEYS_DB "keys-2000000-4000.db"

#define NUM_INTS 32

#define DEBUG 0

void writeKeyPairToPEMs(std::pair<int, int>, uint32_t * moduli, uint32_t * Es);
void processBadKeys(std::vector<std::pair<int, int> > badKeyPairList,
      uint32_t * moduli, uint32_t * Es, int parallel);

void processBadKeys(std::vector<std::pair<int, int> > badKeyPairList,
      uint32_t * moduli, uint32_t * Es, int parallel) {

   char out[80];
   memset(out, 0, 80);
   strcat(out, "bad-key-list_");
   if (parallel == 1) {
      strcat(out, "CUDA_");
   } else {
      strcat(out, "seq_");
   }

   strcat(out, KEYS_DB);

   FILE * outfp;
   if ((outfp = fopen(out, "w")) == NULL) {
         perror("Error creating output file.");
         exit(-1);
   }

   if (badKeyPairList.size() > 0) {
      fprintf(outfp,
         "PARIS found %ld vulnerable pairs of keys in the list your provided in %s.\nTheir paired indices follow, with one pair per line, separated by a \'|\'\n",
         badKeyPairList.size(), KEYS_DB);
   }

   for (unsigned int i = 0; i < badKeyPairList.size(); ++i) {
      //writeKeyPairToPEMs(badKeyPairList[i], moduli, Es)
      fprintf(outfp, "%d | %d\n", badKeyPairList[i].first / NUM_INTS,
            badKeyPairList[i].second);
   }

}
