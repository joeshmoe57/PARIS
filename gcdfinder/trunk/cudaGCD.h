#include "globals.h"
#include <math.h>

#include "getKeys.h"

#define THREADS_PER_BLOCK  512 
#define MAX_BLOCK_DIM      65535
#define NUM_GPUS           2

static int XMASKS[BLKDIM] = { 0x1111,
                              0x2222,
                              0x4444,
                              0x8888 };

static int YMASKS[BLKDIM] = { 0x000F,
                              0x00F0,
                              0x0F00,
                              0xF000 };

// determined experimentally
static int GPUS_TO_USE[NUM_GPUS] = { 0,
                                     3 };

typedef struct xy {
   uint16_t x;
   uint16_t y;
} xyCoord;

typedef struct {
   unsigned long long yNumKeys;
   unsigned long long xNumKeys;

   unsigned long long numBlocks;

   uint32_t * h_yKeys;

   uint32_t * h_keysOrCoords;

   uint16_t ** h_gcd;

   uint32_t * d_yKeys;

   uint32_t * d_keysOrCoords;

   uint16_t * d_gcd;

   cudaStream_t stream;
} GPUplan;

__global__ void GCD_Compare_Diagonal(unsigned *x_dev, xyCoord * dev_coord,
      uint16_t *gcd_dev, unsigned long long numBlocks,
      unsigned long long keysInSet);
__global__ void GCD_Compare_Upper(unsigned *x_dev, unsigned *y_dev,
      uint16_t *gcd_dev, unsigned long long numBlocks,
      unsigned long long keysInXSet, unsigned long long keysInYSet);
__device__ void dev_printNumHex(uint32_t buf[NUM_INTS]);
__device__ void gcd(volatile unsigned *x, volatile unsigned *y);
__device__ void shiftR1(volatile unsigned *x);
__device__ void shiftL1(volatile unsigned *x);
__device__ void cusubtract(volatile unsigned *x, volatile unsigned *y,
      volatile unsigned *z);
__device__ int geq(volatile unsigned *x, volatile unsigned *y);

void dimConversion(unsigned long long numBlocks, int width, xyCoord * coords);
unsigned long long calculateNumberOfBlocks(unsigned long long numKeys);
unsigned long long calculateNumberOfBlocks(unsigned long long xNumKeys,
      unsigned long long yNumKeys);
unsigned long long calculateMaxBlocks(unsigned long long numKeys);
long maximizeKeys(int deviceNumber, bool diagonal);

unsigned long long * calculateKeyListSegments(unsigned long long numKeys,
      int segments);

void printCoords(xyCoord * coords, unsigned long long numBlocks);

void doDiagonalKernel(uint32_t * dev_keys, xyCoord * dev_coords,
      uint16_t * dev_gcd, unsigned long long numBlocks,
      unsigned long long numKeys);
void doUpperKernel(uint32_t * dev_xKeys, uint32_t * dev_yKeys,
      uint16_t * dev_gcd, unsigned long long numBlocks,
      unsigned long long xNumKeys, unsigned long long yNumKeys);

void checkBlockForGCD(uint16_t gcd_res, int blockX, int blockY,
      unsigned long long prevKeysX, unsigned long long prevKeysY,
      keyPairList & badKeyPairList);
void parseGCDResults(unsigned long long numBlocks,
      keyPairList & badKeyPairList, xyCoord * coords, uint16_t * gcd_res,
      unsigned long long prevKeysX, unsigned long long prevKeysY);
void parseGCDResults(unsigned long long numBlocks,
      keyPairList & badKeyPairList, unsigned long long xNumKeys,
      unsigned long long yNumKeys, uint16_t * gcd_res, unsigned long long xIdx,
      unsigned long long yIdx);
