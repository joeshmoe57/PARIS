#include "cudaGCD.h"

#define DEBUG 1

extern "C" {
#include "lpem.h"
#include "keyMath.h"
}

//Kernel Function
/* TODO Figure out if de_coords could be moved into shared memory */
__global__ void GCD_Compare_All(uint32_t * x_dev, uint16_t * gcd_dev, xyCoord * dev_coord, int numBlocks) {
   //Load up shared memory
   __shared__ volatile unsigned int x[BLKDIM][BLKDIM][NUM_INTS];
   __shared__ volatile unsigned int y[BLKDIM][BLKDIM][NUM_INTS];
   int tidx = threadIdx.x;
   int tidy = threadIdx.y;
   // remove and see if reg count decreases
   int tidz = threadIdx.z;
   int blkIdx = blockIdx.x + blockIdx.y * gridDim.x;

   if (blkIdx < numBlocks) {
      xyCoord coord = dev_coord[blkIdx];
      int keyX = tidx + NUM_INTS * (tidy + coord.x * BLKDIM);
      int keyY = tidx + NUM_INTS * (tidz + coord.y * BLKDIM);

      // if (coord.x == coord.y)
      //    we're on a diagonal, and should only use tidy > tidz
      // else // coord.x > coord.y)
      //    we must do an entire 4x4 block
      if (coord.x != coord.y || tidy > tidz) {
         //Check this!! - FIX THIS
         //if(/*tidy == 0 && tidz == 0 && */tidx < NUM_INTS && tidy < BLKDIM && tidz < BLKDIM) {
         //    printf("tidx = %d; tidy = %d; idx = %d\n", tidx, tidy, idx);
         x[tidy][tidz][tidx] = x_dev[keyX];
         y[tidy][tidz][tidx] = x_dev[keyY];

         gcd(x[tidy][tidz], y[tidy][tidz]);
         //gcd_dev[keyX * BLKDIM + keyY * BLKDIM * NUM_INTS] = y[tidy][tidz][tidx];

         // TODO figure out why this is here.
         if (tidx == 31)
            y[tidy][tidz][tidx] -= 1;
//         __syncthreads();
         if (__any(y[tidy][tidz][tidx])) {
            gcd_dev[blkIdx] |= 1<<(tidy + BLKDIM * tidz);
         }

         //gcd_dev[tidx + NUM_INTS * (tidy + tidz * BLKDIM) + THREADS_PER_BLOCK * blkIdx] 
         //   = y[tidy][tidz][tidx];
         
/*      } else {
        gcd_dev[tidx + NUM_INTS * (tidy + tidz * BLKDIM) + THREADS_PER_BLOCK * blkIdx] = 0;
         //y[tidy][tidz][tidx] = 0;
 */     }
   }
}

__device__ void gcd(volatile unsigned *x, volatile unsigned *y) {
   int c;
   c = 0;
   int tid = threadIdx.x;

   //printf("start -> %d: x = %u; y = %u\n", threadIdx.x, x[tid], y[tid]);
   while(((x[NUM_INTS - 1] | y[NUM_INTS - 1]) & 1) == 0) {
      shiftR1(x);
      shiftR1(y);
      c++;
   }

   //parallelNonzero(x)
   while(__any(x[tid])) {
      /* x[0] ? */
      while((x[NUM_INTS - 1] & 1) == 0)
         shiftR1(x);

      /* y[0] ? */
      while((y[NUM_INTS - 1] & 1) == 0)
         shiftR1(y);

      if(geq(x, y)) {
         cusubtract(x, y, x);
         shiftR1(x);
      } else {
         cusubtract(y, x, y);
         shiftR1(y);
      }
   }

   //printf("c = %d\n", c);
   for(int i = 0; i < c; i++)
      shiftL1(y);
}

__device__ void shiftR1(volatile unsigned *x) {
   unsigned x1 = 0;
   //printf("right shifting %d\n", threadIdx.z);
   int tid = threadIdx.x;

   if(tid)
      x1 = x[tid - 1];

   x[tid] = (x[tid] >> 1) | (x1 << 31);
}

__device__ void shiftL1(volatile unsigned *x) {
   //printf("left shifting %d\n", threadIdx.z);
   unsigned x1 = 0;
   int tid = threadIdx.x;

   if(tid != NUM_INTS - 1)
      x1 = x[tid + 1];

   /* unsigned x1 = threadIdx.x ? 0 : x[threadIdx.x - 1];*/
   x[tid] = (x[tid] << 1) | (x1 >> 31);
}

__device__ int geq(volatile unsigned *x, volatile unsigned *y) {
   //printf("greater than %d\n", threadIdx.z);
   // pos is the maximum index (which int in the key) where the values are not the same
   __shared__ unsigned int pos[BLKDIM][BLKDIM];
   int tid = threadIdx.x;

   if(tid== 0)
      pos[threadIdx.y][threadIdx.z] = NUM_INTS - 1;

   if(x[tid] != y[tid])
      atomicMin(&pos[threadIdx.y][threadIdx.z], tid);

   //printf("pos = %d, x = %u; y = %u\n", pos[threadIdx.y], x[0], y[0]);
   return x[pos[threadIdx.y][threadIdx.z]] >= y[pos[threadIdx.y][threadIdx.z]];
}

__device__ void cusubtract(volatile unsigned *x, volatile unsigned *y, volatile unsigned *z) {
   //printf("subtracting %d\n", threadIdx.z);
   __shared__ unsigned char s_borrow[BLKDIM][BLKDIM][NUM_INTS];
   // borrow points to j
   unsigned char *borrow = s_borrow[threadIdx.y][threadIdx.z];
   int tid = threadIdx.x;
   //printf("%u subtracting %u from %u\n", threadIdx.x, y[threadIdx.x], x[threadIdx.x]);

   if(tid == 0)
      borrow[NUM_INTS - 1] = 0;

   unsigned int t;
   t = x[tid] - y[tid];

   if(tid)
      borrow[tid - 1] = (t > x[tid]);

   while(__any(borrow[tid]))
   {
      //printf("t = %u; borrow[%u] = %u\n", t, threadIdx.x, borrow[threadIdx.x]);
      if(borrow[tid]) {
         t--;
      }

      if(tid)
         borrow[tid - 1] = (t == 0xffffffffU && borrow[tid]);
   }

   //printf("t = %d\n" , t);
   z[tid] = t;
}

// TODO document
void dimConversion(int numBlocks, int width, xyCoord * coords) {
   int x = 0, y = 0;
   for (int i = 0; i < numBlocks; ++i) {
      if (x == width) {
         ++y;
         x = y;
      }
      coords[i].x = x++;
      coords[i].y = y;
   }
}

/* This is a modified version of the geometic expansion of:
 *    sum from 1 to keys/BLKDIM 
 * That is,
 * sum(x, 1, n) = 1/2 * n * (n + 1)
 * So,
 * sum(x, 1, keys / BLKDIM)
 *    === 1/2 * (keys / BLKDIM) * (keys / BLKDIM + 1)
 *    === 1/2 * ((keys / BLKDIM) ^ 2 + keys / BLKDIM)
 *    === 1/2 * (1 / BLKDIM ^ 2) * (keys ^ 2 + BLKDIM * keys)
 *
 * This is also a generic solution that allows for multiple versions of BLKDIM.
 * Since for our implementation, this value is fixed, one could also just use:
 * numBlocks = (keys * keys + 4 * keys) >> 5;
 */
long calculateNumberOfBlocks(long keys) {
   // TODO this algorithm assumes BLKDIM ^ 2 is a power of 2
   int sq = BLKDIM * BLKDIM * 2;
   int shift = 0;
   while (!(sq & 1)) {
      sq >>= 1;
      ++shift;
   }

   dprint("keys = %d; shift = %d\n", keys, shift);
   int rem = keys % BLKDIM > 0 ? 1 : 0;

   //return (keys * keys + BLKDIM * keys) >> (shift);
   return 1/2.0 * (( keys / BLKDIM + rem) * (keys / BLKDIM + rem) +  keys / BLKDIM + rem); 
}

/* In order to maximize the number of keys processed per card, the amount of
 * free memory is queried, and used to obtain a maximal number of keys based on
 * the known memory contraints and formulas.
 *
 * The value returned corresponds to the maximal number of keys that can be
 * completely processed on the provided device.
 */
long calculateMaxKeysForDevice(int deviceNumber) {
   size_t f = 0, t = 0;
   cudaSetDevice(deviceNumber);
   cudaMemGetInfo(&f, &t);
   dprint("Free: %ld Total: %ld\n", (long) f, (long) t);

   // For the current implementation, limited to 4x4 blocks, and 1024-bit keys,
   // The following is a manipulation of the quadratic formula
   long A = 3;
   long B;
   //if (diagonal) {
      // B will always be 2060
      //B = BLKDIM * (BLKDIM * BYTES_PER_KEY + A);
   //} else {
      // B will always be 3072 
      B = BLKDIM * BLKDIM * (BYTES_PER_KEY + BYTES_PER_KEY / 2);
   //}
   long C = BLKDIM * BLKDIM * t;

   return (long) ((1.0/(2 * A)) * (-B + sqrt(B * B + 4 * A * C)));
}

/* Calculates the number of total kernel launches necessary for a set of divisions.
 * Based on the way we are dividing the matrix (into 4 parts each time), this
 * value will just be increasing powers of 4.
 */
int calculateNumberOfKernelLaunches(int numDivisions) {
   int kernels = 1;
   int i;
   for (i = 1; i < numDivisions; ++i) {
      kernels *= 4;
   }
   return kernels;
}

/* Calculates the indices in the key list where segmentations will occur. 
 * This function returns a pointer to the list of indices; the caller is
 * responsible for freeing this memory.
 */
int * calculateKeyListSegments(unsigned long numKeys, int segments) {
   int leftOverKeys = numKeys % segments;
   dprint("%d\n", leftOverKeys);
   int stepBase = numKeys / segments;
   dprint("%d\n", stepBase);

   int * indexList = (int *) malloc(segments * sizeof(int));

   int i = 0;
   indexList[i] = stepBase;
   do {
      // This is the remainder of keys, and gets distributed to the segments
      // as soon as possible. 
      // Notice the final index of the list is not checked for this:
      // if it was the case that leftOvers > 0 on the last index, it wouldn't
      // be a remainder since this implies leftOverKeys == segments
      if (leftOverKeys > 0) {
         indexList[i] = indexList[i] + 1;
         --leftOverKeys;
      }
      ++i;
      indexList[i] = stepBase + indexList[i - 1];
   } while (i < segments - 1);

   if (leftOverKeys > 0)
      fprintf(stderr, "ERROR: Left over keys remain after assigning indices.\n");

   return indexList;
}

int main(int argc, char**argv) {
   unsigned long totalNumKeys;
   std::vector<RSA*> privKeys;
   uint32_t *keys;
   // This holds the the number of times the key set will be divided so that 
   // results will fit onto the GPU. 
   // Note: this is not the same as the number of kernels that will be launched.
   int segmentDivisionFactor = 1;

   if(argc != 2) {
       printf("Wrong number of args (Only number of keys)");
       exit(1);
   }
   
   // TODO use args not scanf
   //get number of keys to process
   sscanf(argv[1], "%lu", &totalNumKeys);

   // TODO assumption that totalNumKeys is a multiple of BLKDIM is being made
   if((keys = (uint32_t *) malloc(totalNumKeys * NUM_INTS * sizeof(uint32_t)))
       == NULL) {
      perror("Cannot malloc Key Vector");
      exit(-1);
   }

   totalNumKeys = getAllKeys(KEYS_DB, totalNumKeys, &privKeys, keys);
   dprint("getKeys returns %d\n", totalNumKeys);

   int device = 0;
   int maxKeys = calculateMaxKeysForDevice(device);
   dprint("maxKeys = %ld\n", maxKeys);

   unsigned long keysSegmentSize = totalNumKeys;
   while (keysSegmentSize > maxKeys) {
      segmentDivisionFactor <<= 1;
      keysSegmentSize >>= 1;
   }
   dprint("segment factor: %ld\n", segmentDivisionFactor);
   dprint("total / segment factor: %ld\n", totalNumKeys / segmentDivisionFactor);
   dprint("keysSegmentSize: %ld\n", keysSegmentSize);
   int numberOfKernelLaunches =
      calculateNumberOfKernelLaunches(segmentDivisionFactor);
   dprint("number of kernel launches: %d\n", numberOfKernelLaunches);

   int * segmentIndices = calculateKeyListSegments(totalNumKeys, segmentDivisionFactor);
   for (int i = 0; i < segmentDivisionFactor; ++i)
      dprint("i = %d | %d\n", i, segmentIndices[i]);

   for (int i = 0; i < numberOfKernelLaunches; ++i) {
      int currentNumKeys = totalNumKeys / segmentDivisionFactor;
   }

   /* TODO Assumes totalNumberOfKeys is less than sqrt(LONG_MAX) */ 
   long numBlocks = calculateNumberOfBlocks(totalNumKeys);
   dprint("numBlocks = %ld\n", numBlocks);

   xyCoord * coords;
   if ((coords = (xyCoord * ) malloc(numBlocks * sizeof(xyCoord))) == NULL) {
      perror("Cannot malloc space for coordinates");
      exit(-1);
   }
   dprint("calling dimConversion with %d, %d\n", numBlocks, totalNumKeys / BLKDIM);
   dimConversion(numBlocks, totalNumKeys / BLKDIM, coords);

   /*
#if DEBUG
   // Print the coordinates of the blocks if they were in a square
   for (int i = 0; i < numBlocks; ++i) 
      printf("(%d, %d)\n", coords[i].x, coords[i].y);
   fflush(stdout);
#endif
*/

   uint16_t * gcd_res;
   if ((gcd_res = (uint16_t *) malloc(numBlocks * sizeof(uint16_t))) == NULL) {
      perror("Cannot malloc space for gcd results");
      exit(-1);
   }

   uint32_t * dev_keys;
   uint16_t * dev_gcd;
   xyCoord * dev_coords;

   /* Sizes added since they are used many times. Trying to keep consistency 
    * so changes don't break things.*/
   unsigned int dev_keysSize = totalNumKeys * NUM_INTS * sizeof(int);
   unsigned int dev_gcdSize = numBlocks * sizeof(uint16_t);
   unsigned int dev_coordSize = numBlocks * sizeof(xyCoord);
   dprint("%u bytes for dev_keys\n", dev_keysSize);
   dprint("%u bytes for dev_gcd\n", dev_gcdSize);
   dprint("%u bytes for dev_coords\n", dev_coordSize);
   dprint("%lu bytes for all\n", dev_keysSize + dev_gcdSize + dev_coordSize);

   memset(gcd_res, 0, dev_gcdSize);

   cudaMalloc((void **)&dev_keys, dev_keysSize);
   dprint("cudaMalloc:%s\n", cudaGetErrorString(cudaGetLastError()));
   cudaMemcpy(dev_keys, keys, dev_keysSize, cudaMemcpyHostToDevice);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   cudaMalloc((void **)&dev_gcd, dev_gcdSize);
   dprint("cudaMalloc:%s\n", cudaGetErrorString(cudaGetLastError()));
   cudaMemcpy(dev_gcd, gcd_res, dev_gcdSize, cudaMemcpyHostToDevice);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   cudaMalloc((void **)&dev_coords, dev_coordSize);
   dprint("cudaMalloc:%s\n", cudaGetErrorString(cudaGetLastError()));
   cudaMemcpy(dev_coords, coords, dev_coordSize, cudaMemcpyHostToDevice);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   int dimGridx = numBlocks > MAX_BLOCK_DIM ? MAX_BLOCK_DIM : numBlocks;
   int dimy = 1 + numBlocks / MAX_BLOCK_DIM;
   int dimGridy = 1 < dimy ? dimy : 1;
   dim3 dimGrid(dimGridx, dimGridy); 
   dim3 dimBlock(NUM_INTS, BLKDIM, BLKDIM);

   dprint("dimGrid = %d %d %d; dimBlock = %d %d %d\n", dimGrid.x, dimGrid.y,
         dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

   //hrt_start();
   GCD_Compare_All<<<dimGrid, dimBlock>>>(dev_keys, dev_gcd, dev_coords, numBlocks);
   dprint("kernel:%s\n", cudaGetErrorString(cudaGetLastError()));
   cudaDeviceSynchronize();
   dprint("cudaDeviceSynchronize:%s\n", cudaGetErrorString(cudaGetLastError()));

   //hrt_stop();
   //fprintf(stderr, "Kernel took %s.\n", hrt_string());

   // Copy the results from the card to the CPU
   cudaMemcpy(gcd_res, dev_gcd, dev_gcdSize, cudaMemcpyDeviceToHost);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));
   
   // END GPU WORK

   // open a file and write results into it

   for (int k = 0; k < numBlocks; ++k) {
      /* check this block for bad keys */
      if (gcd_res[k] > 0) {
         printf("bad key found in block: (%d, %d)\n", coords[k].x, coords[k].y);
         printf("k = %d %x\n", k, gcd_res[k]);

         // Visit each block
         for (int j = 0; j < BLKDIM; ++j) {
            // check if any bits in a column are found
            if (gcd_res[k] & YMASKS[j]) {
               // find which row 
               for (int i = 0; i < BLKDIM; ++i) {
                  if (gcd_res[k] & YMASKS[j] & XMASKS[i]) {
                     int one = NUM_INTS * (BLKDIM * coords[k].x + i);
                     int two = NUM_INTS * (BLKDIM * coords[k].y + j);
                     printf("key %d, %d in block (%d, %d)\n", i, j, coords[k].x, coords[k].y);
                     printf("VULNERABLE KEY PAIR FOUND:\n");
                     printf("one: %d\n", one);
                     printf("two: %d\n", two);
                     printNumHex(keys + one);
                     printNumHex(keys + two);
                     uint32_t gcd[NUM_INTS];
                     memset(gcd, 0, NUM_INTS * sizeof(uint32_t));

                     gcd1024(&keys[one],
                             &keys[two],
                             gcd);
                     printf("Done calculating gcd.\n");

                     printf("PRIVATE KEY:\n");
                     printNumHex(gcd);
                  }
               }
            }
         }
      }
         /*
      for (int i = 0; i < BLKDIM; ++i)
         for (int j = 0; j < BLKDIM; ++j) {
               int x = i + BLKDIM * coords[k].x;
               int y = j + BLKDIM * coords[k].y;

               if(x > y )
               {
                  printf("x\n");
                  printNumHex(keys[x]);
                  printf("y\n");
                  printNumHex(keys[y]);

                  printf("k = %d i = %d j = %d\n", k, x, y);
                  printNumHex(gcd_res + NUM_INTS * (BLKDIM * (k * BLKDIM + j)+ i));
               }
         }
         */
   }
   free(keys);
   free(gcd_res);
   free(coords);
   return 0;
}

