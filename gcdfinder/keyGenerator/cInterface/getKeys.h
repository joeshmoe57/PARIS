#include <vector>
#include "lcrypto.h"
#include "lbn.h"
#include "lrsa.h"

//gets all the keys in the database fileName, and puts them in the 2 vectors.
int getAllKeys(char* fileName, int num, std::vector<RSA*> *privKeys, uint32_t pubKeys[][32]);

//converts OpenSSL BigNum buffers to our format
int bnToUs(BIGNUM *bn, uint32_t *us);
