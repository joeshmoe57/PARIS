#include "seqGCD.h"
#include <vector>
#include "getKeys.h"

extern "C"{
#include "lpem.h"
#include "keyMath.h"
}
int main(int argc, char ** argv) {

    uint32_t *u;
    uint32_t w[NUM_INTS];
    long total_keys;
    std::vector<RSA*> privKeys;

    //get number of keys to process
    if(argc != 2)
    {
        printf("Wrong number of args");
        exit(1);
    }

    sscanf(argv[1], "%ld", &total_keys);

    //get keys
    if((u = (uint32_t *) malloc(total_keys * NUM_INTS * sizeof(uint32_t))) == 0)
    {
        perror("Cannot Malloc Key Vector");
        exit(1);
    }

    total_keys = getAllKeys(KEYS_DB, total_keys, &privKeys, u);

    int sq = BLKDIM * BLKDIM * 2;
    int shift = 0;
    while (!(sq & 1)) {
        sq >>= 1;
        ++shift;
    }

    long numBlocks = (total_keys * total_keys + BLKDIM * total_keys) >> (shift);
    printf("numBlocks = %ld\n", numBlocks);

    //Put in a bad pair
    /*
       memset(u[1479], 0, sizeof(int) * 32);
       memset(u[1478], 0, sizeof(int) * 32);
       u[1479][31] = 1989;
       u[1478][31] = 867;
     */

    //compare
    uint32_t one[NUM_INTS] = {0};
    one[31] = 1;

    uint32_t blkX = 0, blkY = 0;
    for (int k = 0; k < numBlocks; ++k)
    {
        uint16_t block_res = 0;
        if (blkX == total_keys / BLKDIM)
            blkX = ++blkY;

        for (int i = 0; i < BLKDIM; ++i)
        {
            for (int j = 0; j < BLKDIM; ++j)
            {
                uint32_t x, y;
                uint32_t tmpX[NUM_INTS];
                uint32_t tmpY[NUM_INTS];

                //calc index
                x = i + BLKDIM * blkX;
                y = j + BLKDIM * blkY;

                memcpy(tmpX, u + x * NUM_INTS, sizeof(uint32_t) * NUM_INTS);
                memcpy(tmpY, u + y * NUM_INTS, sizeof(uint32_t) * NUM_INTS);

                if(x > y)
                {

                   uint32_t tempx[NUM_INTS];
                   memcpy(tempx, tmpX, NUM_INTS * sizeof(uint32_t));
                   uint32_t tempy[NUM_INTS];
                   memcpy(tempy, tmpY, NUM_INTS * sizeof(uint32_t));

                    gcd1024(tmpX, tmpY, w);
                    //printf("k = %d i = %d j = %d\n", k, x, y);
                    if(equalTo(w, one) == 0)
                    {
                    printf("x\n");
                    printNumHex(tempx);
                    printf("y\n");
                    printNumHex(tempy);
                        //printf("GOT GCD\n");
                        printNumHex(w);
                        block_res |= (1 << (i + j * BLKDIM));
                    }
                }
            }
        }

        printf("k = %d %x\n", k, block_res);
        blkX++;
    }

    return 0;
}
