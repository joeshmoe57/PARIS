/*
 * Reads in 1024 bit RSA keys from the keys sub directory, and sticks them
 * into our internal 1024 bit representation.
 */

#include <stdio.h>
#include <vector>
#include "sqlite3.h"

int getAllKeys(char* fileName, std::vector<uint8_t*> *keys);

int main(void){
   puts("Starting key printer...");

   std::vector<uint8_t*> allKeys;

   int res = getAllKeys("keys-100k.db", &allKeys);

   printf("res of call was %d\n", res);
   printf("after, vector of keys was %d\n", (int)allKeys.size());

   return 0;
}


/*
 * Gets all the keys out of the specified database and puts them in the vector.
 * Returns 0 on success, negative on failure.
 */

int getAllKeys(char* fileName, std::vector<uint8_t*> *keys){
   sqlite3 *db;
   sqlite3_stmt *statement;
   char *zErrMsg = 0;
   int res;

   //connect to the db
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
