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

int getAllKeys(char* fileName, std::vector<uint8_t*> *keys);

int main(int argc, char* argv[]){
   FILE* pemBuf;
   int res;

   std::vector<uint8_t*> allKeys;

   if(argc != 2){
      puts("wrong number of args. takes db to open");
      return 0;
   }

   puts("Starting key printer...");

   //run through the DB passed in and get all they keys from it
   res = getAllKeys(argv[1], &allKeys);

   printf("res of call was %d\n", res);
   printf("after, vector of keys was %d\n", (int)allKeys.size());

   //now try using OpenSSL's crypto to decode those PEM strings into RSA keys
   RSA* rkey = RSA_new();

   printf("pem = [%s]\nPEM strlen = %d\n", (allKeys.at(0)), (int)strlen((char*)allKeys.at(0)));

   //open the pem string as a FILE* so we can use OpenSSL
   //pemBuf = fmemopen(((char*)allKeys[0]), strlen((char*)allKeys[0]), "r");
   pemBuf = SCFmemopen(((char*)allKeys.at(0)), strlen((char*)allKeys.at(0)), "r");
   if(!pemBuf){
      puts("error opening key as buffer");
      return -1;
   }

   //get the key
   //rkey = PEM_read_RSAPrivateKey(pemBuf, NULL, NULL, 0);
   PEM_read_RSAPrivateKey(pemBuf, &rkey, NULL, 0);
   if(!rkey){
      puts("error decoding PEM string into RSA key");
      return -1;
   }

   //now print the BN's
   //BN_print_fp(stdout, rkey->n);
   //printf("was the bn %d big?\n", BN_num_bytes(rkey->n));
   //printf("top of bn was %d\n", rkey->e->dmax);
   RSA_print_fp(stdout, rkey, 0);

   //and free the key
   RSA_free(rkey);

   return 0;
}


/*
 * Gets all the keys out of the specified database and puts them in the vector.
 * Returns 0 on success, negative on failure.
 */

int getAllKeys(char* fileName, std::vector<uint8_t*> *keys){
   //according to dox, DO NOT malloc this. Open creates it
   sqlite3 *db = NULL; 
   sqlite3_stmt *statement;
   char *zErrMsg = 0;
   int res = 0;

   //connect to the db
   if(!fileName){
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
   res = sqlite3_prepare_v2(db, qGetAll, strlen(qGetAll), &statement, NULL);
   if(res != SQLITE_OK){
      fprintf(stderr, "SQL error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
      sqlite3_close(db);
      return -1;
   }

   //now loop through the entire database, adding things to the vector
   int bufSize;
   uint8_t *buf;
   while((res = sqlite3_step(statement)) == SQLITE_ROW){
      //get the actual blob out
      bufSize = sqlite3_column_bytes(statement, 0);
      buf = (uint8_t*)malloc(bufSize);
      buf = (uint8_t*)sqlite3_column_text(statement, 0);

      //printf("%s\n", buf);

      keys->push_back(buf);
      
      printf("%s\n", keys->at(0));
   }

   sqlite3_finalize(statement);
   sqlite3_close(db);

   return 0;
}

/*
   static int callback(void *NotUsed, int argc, char **argv, char **azColName){
   int i;
   for(i=0; i<argc; i++){
   printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
   }
   printf("\n");
   return 0;
   }
   */
