#include <stdint.h>
#include <vector>
#include "lcrypto.h"
#include "lbn.h"
#include "lrsa.h"

#define RSA_384_BITS  384 
#define RSA_512_BITS  512 
#define RSA_768_BITS  768 
#define RSA_1024_BITS 1024
#define RSA_2048_BITS 2048
#define RSA_2432_BITS 2432
#define RSA_3072_BITS 3072 
#define RSA_4096_BITS 4096
#define RSA_8192_BITS 8192 
#define ALLOWED_RSA_BITS RSA_1024_BITS
#define BITS_PER_BYTE 8

//gets all the keys in the database fileName, and puts them in the 2 vectors.
int getAllKeys(const char* fileName, int num, std::vector<RSA*> *privKeys, uint32_t* pubKeys);

//converts OpenSSL BigNum buffers to our format
int bnToUs(BIGNUM *bn, uint32_t *us);
