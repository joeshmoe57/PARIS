#ifndef GLOBALS_H
#define GLOBALS_H

#include "keyOutput.h"

//#define KEYS_DB "keys-20-2.db"
//#define KEYS_DB "../../../keys-2000000-4000.db"
//#define KEYS_DB "keys-2000000-4000.db"
#define BYTES_PER_KEY 128
#define BLKDIM 4 

#define dprint(...) do { if(DEBUG) fprintf(stderr, __VA_ARGS__); } while(0);

#define P(varname) fprintf(stderr, "%s = %d\n", #varname, varname);
#define DP(varname) do { if(DEBUG) fprintf(stderr, "%s = %d\n", #varname, varname); } while(0);

#endif
