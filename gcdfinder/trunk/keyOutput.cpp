#include "keyOutput.h"

/*
void writeKeyPairToPEMs(std::pair<int, int>, uint32_t * moduli, uint32_t * Es) {

}

void processBadKeys(std::vector<std::pair<int, int> > badKeyPairList,
      uint32_t * moduli, uint32_t * Es) {

   char out[80];
   strcat(out, "bad-key-list_");
   strcat(out, KEYS_DB);

   FILE * outfp;
   if ((outfp = fopen(out, "w")) == NULL) {
         perror("Error creating output file.");
         exit(-1);
   }

   if (badKeyPairList.size() > 0) {
      fprintf(outfp,
         "PARIS found %ld vulnerable pairs of keys in the list your provided in %s.\n Their paired indices follow, with one pair per line, separated by a \'|\'",
         badKeyPairList.size(), KEYS_DB);
   }

   for (unsigned int i = 0; i < badKeyPairList.size(); ++i) {
      //writeKeyPairToPEMs(badKeyPairList[i], moduli, Es)
      fprintf(outfp, "%d | %d\n", badKeyPairList[i].first, badKeyPairList[i].second);
   }

}
*/
