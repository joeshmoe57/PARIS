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

#include "getKeys.h"


/*
 * Gets the specified number of keys out of the specified database and puts them in two places:
 * a vector of RSA* private keys, and a vector of uint32_t[1024] n's, set to
 * out endian-ness.
 * Returns the number of keys retreived on success, negative on failure.
 */
int getAllKeys(char* fileName, int num, std::vector<RSA*> *privKeys, uint32_t pubKeys[][32]) {
   //according to dox, DO NOT malloc this. Open creates it
   sqlite3 *db = NULL; 
   sqlite3_stmt *statement;
   char *zErrMsg = 0;
   int res = 0;
   int keyIndex;

   //connect to the db
   if(!fileName || !privKeys || !pubKeys){
      puts("bad file name!");
      return -1;
   }

   res = sqlite3_open(fileName, &db);
   if(res){
      printf("Error opening DB: %s\n", sqlite3_errmsg(db));
      sqlite3_close(db);
      return -1;
   }

   //run a query (get all keys in this case)
   char qGetAll[] = "SELECT * FROM keys;";
   res = sqlite3_prepare_v2(db, qGetAll, strlen(qGetAll), &statement, (const char**)&zErrMsg);
   printf("res was %d\n", res);
   if(res != SQLITE_OK){
      fprintf(stderr, "SQL error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
      sqlite3_close(db);
      return -1;
   }

   //now loop through the entire database, adding things to the vector
   int bufSize;
   RSA* tkey;
   uint8_t* buf;
   uint32_t* ourform;
   FILE* pem;
   keyIndex = 0;
   while(keyIndex < num && (res = sqlite3_step(statement)) == SQLITE_ROW){
      //get the actual blob out
      bufSize = sqlite3_column_bytes(statement, 0);
      buf = (uint8_t*)sqlite3_column_text(statement, 0);

      //open the PEM byffer as a FILE* so we can convert to RSA
      pem = SCFmemopen(buf, bufSize, "r");

      //convert it to an RSA*
      tkey = RSA_new();
      PEM_read_RSAPrivateKey(pem, &tkey, NULL, 0);
      if(!tkey){
         puts("error decoding PEM string into RSA key");
         return -1;
      }

      privKeys->push_back(tkey);
      //RSA_print_fp(stdout, privKeys->at(0), 0);
      
      //now convert N into our format, and store that as well
      ourform = (uint32_t*)calloc(32, sizeof(uint32_t));
      res = bnToUs(tkey->n, ourform);
      if(res != 0){
         puts("Error convering big num types");
         return -1;
      }

      //pubKeys->push_back(ourform);
      memcpy(pubKeys[keyIndex], ourform, 32);
      keyIndex++;
   }

   sqlite3_finalize(statement);
   sqlite3_close(db);

   return keyIndex;
}

/*
 * Converts a BIGNUM into one of our 1024 bit ints (with opposing endianness).
 * BIGNUMs have slot [max] = MSB. We have slot [0] as MSB. It expects everything
 * incoming to be pre-allocated
 * Returns -1 on failure (null buffer), 1 on success.
 */
int bnToUs(BIGNUM *bn, uint32_t *us){
   if(!bn || !us){
      puts("null to reverse");
      return -1;
   }

   //go through the bn forward, and the us in reverse to store each byte
   int i;
   for(i = 0; i < bn->top; i++){
      us[31-i] = bn->d[i];
   }
   return 0;
}
