#include <stdio.h>
#include <stdint.h>
#include <sys/resource.h>
#include <errno.h>
#include "hrt.h"
#include <math.h>

#include "getKeys.h"
#include <vector>

#include "globals.h"


static int XMASKS[BLKDIM] = { 0x1111,
                              0x2222,
                              0x4444,
                              0x8888 };

static int YMASKS[BLKDIM] = { 0x000F,
                              0x00F0,
                              0x0F00,
                              0xF000 };

#define THREADS_PER_BLOCK  512 
#define MAX_BLOCK_DIM      65535

#define dprint(...) do { if(DEBUG) fprintf(stderr, __VA_ARGS__); } while(0);
//#define d2print(...) do { if(DEBUG_2) fprintf(stderr, __VA_ARGS__); } while(0);

typedef struct xy {
   uint16_t x;
   uint16_t y;
} xyCoord;

//__global__ void HighThroughputGCD(unsigned *x_dev, unsigned *gcd_dev);
__global__ void GCD_Compare_All(unsigned *x_dev, uint16_t *gcd_dev, xyCoord * dev_coord);
__device__ void gcd(volatile unsigned *x, volatile unsigned *y);
__device__ void shiftR1(volatile unsigned *x);
__device__ void shiftL1(volatile unsigned *x);
__device__ void cusubtract(volatile unsigned *x, volatile unsigned *y, volatile unsigned *z);
__device__ int geq(volatile unsigned *x, volatile unsigned *y);
void dimConversion(int numBlocks, int width, xyCoord * coords);
long calculateNumberOfBlocks(long keys);
long calculateMaxKeysForDevice(int deviceNumber);
