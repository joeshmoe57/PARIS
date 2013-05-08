#include "cudaGCD.h"

extern "C" {
#include "lpem.h"
#include "keyMath.h"
}

__global__ void GCD_Compare_Upper(unsigned *x_dev, unsigned *y_dev,
      uint16_t *gcd_dev, unsigned long long keysInXSet,
      unsigned long long keysInYSet) {
   __shared__ volatile uint32_t x[BLKDIM][BLKDIM][NUM_INTS];
   __shared__ volatile uint32_t y[BLKDIM][BLKDIM][NUM_INTS];
   uint32_t tidx = threadIdx.x;
   uint32_t tidy = threadIdx.y;
   uint32_t tidz = threadIdx.z;
   uint32_t blkx = blockIdx.x;
   uint32_t blky = blockIdx.y;
   uint32_t blkIdx = blkx + blky * gridDim.x;

   uint32_t whichKeyX = tidy + blkx * BLKDIM;
   uint32_t whichKeyY = tidz + blky * BLKDIM;

   if (whichKeyX < keysInXSet && whichKeyY < keysInYSet) {
      uint32_t whichIntX = tidx + NUM_INTS * whichKeyX;
      uint32_t whichIntY = tidx + NUM_INTS * whichKeyY;

      x[tidy][tidz][tidx] = x_dev[whichIntX];
      y[tidy][tidz][tidx] = y_dev[whichIntY];

      gcd(x[tidy][tidz], y[tidy][tidz]);

      /* If we're on the least significant thread, subtract 1 so that
       * results that equal 1, now equal zero, and will fail the following
       * "any".
       * This is okay to do since we're just storing bit-vectors, and don't
       * need to preserve the actual values, but instead, just whether the
       * answer was greater than 1
       */
      if (tidx == 31)
         y[tidy][tidz][tidx] -= 1;
      if (__any(y[tidy][tidz][tidx])) {
         gcd_dev[blkIdx] |= 1<<(tidy + BLKDIM * tidz);
      }
   }
}

__global__ void GCD_Compare_Diagonal(uint32_t * x_dev, xyCoord * dev_coord,
      uint16_t * gcd_dev, unsigned long long numBlocks,
      unsigned long long keysInSet) {
   __shared__ volatile uint32_t x[BLKDIM][BLKDIM][NUM_INTS];
   __shared__ volatile uint32_t y[BLKDIM][BLKDIM][NUM_INTS];
   uint32_t tidx = threadIdx.x;
   uint32_t tidy = threadIdx.y;
   uint32_t tidz = threadIdx.z;
   uint32_t blkIdx = blockIdx.x + blockIdx.y * gridDim.x;

   if (blkIdx < numBlocks) {
      xyCoord coord = dev_coord[blkIdx];
      uint32_t whichKeyX = tidy + coord.x * BLKDIM;
      uint32_t whichKeyY = tidz + coord.y * BLKDIM;

      if (whichKeyX < keysInSet && whichKeyY < keysInSet) {
         uint32_t whichIntX = tidx + NUM_INTS * whichKeyX;
         uint32_t whichIntY = tidx + NUM_INTS * whichKeyY;

         if (coord.x != coord.y || tidy > tidz) {
            x[tidy][tidz][tidx] = x_dev[whichIntX];
            y[tidy][tidz][tidx] = x_dev[whichIntY];

            gcd(x[tidy][tidz], y[tidy][tidz]);

            /* If we're on the least significant thread, subtract 1 so that
             * results that equal 1, now equal zero, and will fail the following
             * "any".
             * This is okay to do since we're just storing bit-vectors, and don't
             * need to preserve the actual values, but instead, just whether the
             * answer was greater than 1
             */
            if (tidx == 31)
               y[tidy][tidz][tidx] -= 1;

            if (__any(y[tidy][tidz][tidx])) {
               gcd_dev[blkIdx] |= 1<<(tidy + BLKDIM * tidz);
            }

         }
      }
   }
}

__device__ void dev_printNumHex(uint32_t buf[NUM_INTS]){
   int i;
   for (i = 0; i < NUM_INTS; ++i) {
      printf("%08x ", buf[i]);
   }
   printf("\n");
}

__device__ void gcd(volatile unsigned *x, volatile unsigned *y) {
   int c;
   c = 0;
   int tid = threadIdx.x;

   while(((x[NUM_INTS - 1] | y[NUM_INTS - 1]) & 1) == 0) {
      shiftR1(x);
      shiftR1(y);
      c++;
   }

   while(__any(x[tid])) {
      while((x[NUM_INTS - 1] & 1) == 0)
         shiftR1(x);

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

   for(int i = 0; i < c; i++)
      shiftL1(y);
}

__device__ void shiftR1(volatile unsigned *x) {
   unsigned x1 = 0;
   int tid = threadIdx.x;

   if(tid)
      x1 = x[tid - 1];

   x[tid] = (x[tid] >> 1) | (x1 << 31);
}

__device__ void shiftL1(volatile unsigned *x) {
   unsigned x1 = 0;
   int tid = threadIdx.x;

   if(tid != NUM_INTS - 1)
      x1 = x[tid + 1];

   x[tid] = (x[tid] << 1) | (x1 >> 31);
}

__device__ int geq(volatile unsigned *x, volatile unsigned *y) {
   // pos is the maximum index (which int in the key) where the values are not the same
   __shared__ unsigned int pos[BLKDIM][BLKDIM];
   int tid = threadIdx.x;

   if(tid== 0)
      pos[threadIdx.y][threadIdx.z] = NUM_INTS - 1;

   if(x[tid] != y[tid])
      atomicMin(&pos[threadIdx.y][threadIdx.z], tid);

   return x[pos[threadIdx.y][threadIdx.z]] >= y[pos[threadIdx.y][threadIdx.z]];
}

__device__ void cusubtract(volatile unsigned *x, volatile unsigned *y, volatile unsigned *z) {
   __shared__ unsigned char s_borrow[BLKDIM][BLKDIM][NUM_INTS];
   // borrow points to j
   unsigned char *borrow = s_borrow[threadIdx.y][threadIdx.z];
   int tid = threadIdx.x;

   if(tid == 0)
      borrow[NUM_INTS - 1] = 0;

   unsigned int t;
   t = x[tid] - y[tid];

   if(tid)
      borrow[tid - 1] = (t > x[tid]);

   while(__any(borrow[tid]))
   {
      if(borrow[tid]) {
         t--;
      }

      if(tid)
         borrow[tid - 1] = (t == 0xffffffffU && borrow[tid]);
   }

   z[tid] = t;
}

/* This function fills the coords array with a lookup table. 
 * The table is a mapping of sequentially numbered blocks in the upper triangle
 * of a matrix, to the actual row, column values.
 * The table is used to translate back numbered blocks to their "original"
 * positions for purposes of indexing into the correct location in a key array.
 * 
 * numBlocks is the total number of blocks in the matrix
 * width is the number of blocks across the top of the matrix
 * coords is the array of xyCoord pairs to be filled.
 */
void dimConversion(unsigned long long numBlocks, int width, xyCoord * coords) {
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
 */
unsigned long long calculateNumberOfBlocks(unsigned long long numKeys) {
   int rem = numKeys % BLKDIM > 0 ? 1 : 0;

   // the /2 is fine here, since the dividend is guarenteed to be even
   return ((numKeys / BLKDIM + rem) * (numKeys / BLKDIM + rem) +
         numKeys / BLKDIM + rem) / 2; 
}

unsigned long long calculateNumberOfBlocks(unsigned long long xKeys,
      unsigned long long yKeys) {
   int xRem = xKeys % BLKDIM > 0 ? 1 : 0;
   int yRem = yKeys % BLKDIM > 0 ? 1 : 0;

   return (xKeys / BLKDIM + xRem) * (yKeys / BLKDIM + yRem);
}

/* In order to maximize the number of keys processed per card, the amount of
 * free memory is queried, and used to obtain a maximal number of keys based on
 * the known memory contraints and formulas.
 *
 * The value returned corresponds to the maximal number of keys that can be
 * completely processed on the provided device.
 */
unsigned long long calculateMaxKeysForDevice(int deviceNumber) {
   size_t f = 0, t = 0;
   cudaSetDevice(deviceNumber);
   cudaMemGetInfo(&f, &t);
   dprint("Free: %ld Total: %ld\n", (long) f, (long) t);

   // For the current implementation, limited to 4x4 blocks, and 1024-bit keys,
   // The following is a manipulation of the quadratic formula
   long A = 3;
   long B;
   // B will always be 2060
   B = BLKDIM * (BLKDIM * BYTES_PER_KEY + A);
   long C = BLKDIM * BLKDIM * t;

   return (unsigned long long) ((1.0/(2 * A)) * (-B + sqrt(B * B + 4 * A * C)));
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
unsigned long long * calculateKeyListSegments(unsigned long long numKeys,
      int segments) {
   int leftOverKeys = numKeys % segments;
   DP(leftOverKeys);
   int stepBase = numKeys / segments;
   DP(stepBase);
   DP(segments);

   unsigned long long * indexList;
   if ((indexList = (unsigned long long *) malloc(segments * sizeof(unsigned long long))) == NULL) {
      perror("Cannot malloc space for segment index list");
      exit(-1);
   }

   int i = 0;
   DP(i);
   indexList[i] = stepBase;
   if (leftOverKeys > 0) {
      indexList[i] += 1;
      --leftOverKeys;
   }

   while (++i < segments) {
      indexList[i] = stepBase + indexList[i - 1];
      // This is the remainder of keys, and gets distributed to the segments
      // as soon as possible. 
      // Notice the final index of the list is not checked for this:
      // if it was the case that leftOvers > 0 on the last index, it wouldn't
      // be a remainder since this implies leftOverKeys == segments
      if (leftOverKeys > 0) {
         indexList[i] += 1;
         --leftOverKeys;
      }
   }

   return indexList;
}

void doDiagonalKernel(uint32_t * dev_keys, xyCoord * dev_coords, 
      uint16_t * dev_gcd, unsigned long long numBlocks, unsigned long long numKeys) { 
   int dimGridx = numBlocks > MAX_BLOCK_DIM ? MAX_BLOCK_DIM : numBlocks;
   int dimy = 1 + numBlocks / MAX_BLOCK_DIM;
   int dimGridy = 1 < dimy ? dimy : 1;
   dim3 dimGrid(dimGridx, dimGridy); 
   dim3 dimBlock(NUM_INTS, BLKDIM, BLKDIM);

   dprint("dimGrid = %d %d %d; dimBlock = %d %d %d\n", dimGrid.x, dimGrid.y,
         dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
   DP(numBlocks);

   GCD_Compare_Diagonal<<<dimGrid, dimBlock>>>(dev_keys, dev_coords, dev_gcd, numBlocks, numKeys);
}

void doUpperKernel(uint32_t * dev_yKeys, uint32_t * dev_xKeys, uint16_t * dev_gcd,
      unsigned long long numBlocks, unsigned long long yNumKeys,
      unsigned long long xNumKeys) { 

   int dimGridx = xNumKeys / BLKDIM + (xNumKeys % BLKDIM ? 1 : 0);
   int dimGridy = yNumKeys / BLKDIM + (yNumKeys % BLKDIM ? 1 : 0);
   dim3 dimGrid(dimGridx, dimGridy); 
   dim3 dimBlock(NUM_INTS, BLKDIM, BLKDIM);

   dprint("dimGrid = %d %d %d; dimBlock = %d %d %d\n", dimGrid.x, dimGrid.y,
         dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
   DP(numBlocks);

   GCD_Compare_Upper<<<dimGrid, dimBlock>>>(dev_xKeys, dev_yKeys, dev_gcd,
         xNumKeys, yNumKeys);
}

/* Returns the maximum number of blocks in either the diagonal segments or the
 * upper rectangular segments, given a number of keys to be used.
 */
unsigned long long calculateMaxBlocks(unsigned long long numKeys) {
   unsigned long long maxDiagBlocks = calculateNumberOfBlocks(numKeys);
   int r = numKeys % BLKDIM > 0 ? 1 : 0;
   unsigned long long maxUpperBlocks = calculateNumberOfBlocks(numKeys, numKeys / 2 + r);
   return maxDiagBlocks > maxUpperBlocks ? maxDiagBlocks : maxUpperBlocks;
}

/* The 2nd CUDA parameters with either be the coordinate lookup table, or the
 * set of keys along the x-axis. We want to minimize allocations, so this
 * function finds the size of the larger of those 2 things.
 * Most often, it will corespond to the coordinate table; however, for small
 * numbers of keys, the key set can be larger.
 */
unsigned long long calculateMax2ndParamSize(unsigned long long numKeys,
      unsigned long long maxBlocks) {
   // calculate max xyCoords size
   unsigned long long maxXYCoordsSize = sizeof(xyCoord) * maxBlocks;
   DP(maxXYCoordsSize);

   // calculate the max size of keys in the X direction
   unsigned long long maxXKeys = numKeys / 2 + (numKeys % 2 ? 1 : 0);
   unsigned long long maxXKeySize = maxXKeys * NUM_INTS * sizeof(uint32_t);
   return maxXYCoordsSize > maxXKeySize ? maxXYCoordsSize : maxXKeySize;
}

void checkBlockForGCD(uint16_t gcd_res, int blockX, int blockY,
      unsigned long long prevKeysX, unsigned long long prevKeysY,
      keyPairList & badKeyPairList) {
   /* check this block for bad keys */
   if (gcd_res > 0) {
      dprint("bad key found in block: (%d, %d)\n", blockX, blockY);
      // traverse columns
      for (int j = 0; j < BLKDIM; ++j) {
         // check if any bits in a column are found
         if (gcd_res & YMASKS[j]) {
            // traverse rows 
            for (int i = 0; i < BLKDIM; ++i) {
               if (gcd_res & YMASKS[j] & XMASKS[i]) {
                  unsigned long long one = NUM_INTS * (BLKDIM * blockX + i + prevKeysX);
                  unsigned long long two = NUM_INTS * (BLKDIM * blockY + j + prevKeysY);
                  badKeyPairList.push_back(std::make_pair(one, two));
                  dprint("key %d, %d in block (%d, %d)\n", i, j, blockX, blockY);
                  dprint("VULNERABLE KEY PAIR FOUND:\n");
                  dprint("one: %llu\n", one);
                  dprint("two: %llu\n", two);
                  dprint("x\n");
                  dprint("y\n");
               }
            }
         }
      }
   }
}

void parseGCDResults(unsigned long long numBlocks,
      keyPairList & badKeyPairList,
      xyCoord * coords, uint16_t * gcd_res, unsigned long long prevKeysX,
      unsigned long long prevKeysY) {
   for (unsigned long long blockId = 0; blockId < numBlocks; ++blockId) {
      int blockX = coords[blockId].x;
      int blockY = coords[blockId].y;

      checkBlockForGCD(gcd_res[blockId], blockX, blockY, prevKeysX, prevKeysY,
            badKeyPairList);
   }
}

void parseGCDResults(unsigned long long numBlocks,
      keyPairList & badKeyPairList,
      unsigned long long xNumKeys, unsigned long long yNumKeys,
      uint16_t * gcd_res, unsigned long long prevKeysX,
      unsigned long long prevKeysY) {
   int xBlocks = xNumKeys / BLKDIM + (xNumKeys % BLKDIM ? 1 : 0);
   int yBlocks = yNumKeys / BLKDIM + (yNumKeys % BLKDIM ? 1 : 0);

   for (int blockX = 0; blockX < xBlocks; ++blockX) {
      for (int blockY = 0; blockY < yBlocks; ++blockY) {
         int blockId = blockX + blockY * xBlocks;

         checkBlockForGCD(gcd_res[blockId], blockX, blockY, prevKeysX,
               prevKeysY, badKeyPairList);
      }
   }
}

void printCoords(xyCoord * coords, unsigned long long numBlocks) {
   // Print the coordinates of the blocks if they were in a square
   for (unsigned long long i = 0; i < numBlocks; ++i) 
      printf("(%d, %d)\n", coords[i].x, coords[i].y);
   fflush(stdout);
}

int main(int argc, char**argv) {
   unsigned long long totalNumKeys;
   std::vector<RSA*> privKeys;
   keyPairList badKeyPairList;
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
   sscanf(argv[1], "%llu", &totalNumKeys);

   /* TODO keys must fit into RAM as well as the other allocated variables like
    * gcd_res and xyCoord
    */
   if((keys = (uint32_t *) malloc(totalNumKeys * NUM_INTS * sizeof(uint32_t)))
         == NULL) {
      perror("Cannot malloc Key Vector");
      exit(-1);
   }

   if ((totalNumKeys = getAllKeys(KEYS_DB, totalNumKeys, &privKeys, keys)) <= 0) {
      fprintf(stderr, "Error getting keys from database.\n");
      exit(-1);
   }
   DP(totalNumKeys);

   hrt_start();

   int device = 0;
   int deviceCount = 0;
   checkCudaErrors(cudaGetDeviceCount(&deviceCount));
   for (int i = 0; i < deviceCount; ++i) {
      dprint("keys for device %d = %d\n", i, calculateMaxKeysForDevice(i));
   }

   unsigned long long maxKeys = calculateMaxKeysForDevice(device);

   unsigned long keysSegmentSize = totalNumKeys;
   int divs = 1;
   while (keysSegmentSize > maxKeys) {
      segmentDivisionFactor <<= 1;
      keysSegmentSize >>= 1;
      ++divs;
   }

   if (NUM_GPUS == 2) {
      if (segmentDivisionFactor == 1)
         segmentDivisionFactor <<= 1;
      if (keysSegmentSize == totalNumKeys)
         keysSegmentSize >>= 1;
      if (divs == 1)
         ++divs;
   }

   unsigned long long * segmentIndices =
      calculateKeyListSegments(totalNumKeys, segmentDivisionFactor);

   int numKernels = calculateNumberOfKernelLaunches(divs);
   DP(numKernels);
   int runs = numKernels / 2;
   if (runs == 0)
      runs = 1;

   // We look at the first segment index since it will also be a count, and if
   // there were remainders, they will have been allocated to this index.
   unsigned long long firstNumKeys = segmentIndices[0];
   unsigned long long maxYKeySize = firstNumKeys * NUM_INTS * sizeof(uint32_t);
   unsigned long long maxBlocks = calculateMaxBlocks(firstNumKeys);
   unsigned long long max2ndParamSize = calculateMax2ndParamSize(firstNumKeys, maxBlocks);
   unsigned long long maxGCDSize = sizeof(uint16_t) * maxBlocks;

   GPUplan plan[NUM_GPUS];

   // malloc and create streams on GPU
   for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
      checkCudaErrors(cudaSetDevice(GPUS_TO_USE[gpu]));
      checkCudaErrors(cudaStreamCreate(&plan[gpu].stream));
      checkCudaErrors(cudaMalloc((void **) &plan[gpu].d_yKeys, maxYKeySize));
      checkCudaErrors(cudaMalloc((void **) &plan[gpu].d_keysOrCoords, max2ndParamSize));
      checkCudaErrors(cudaMalloc((void **) &plan[gpu].d_gcd, maxGCDSize));
      /*
      if ((plan[gpu].h_keysOrCoords = (uint32_t *) malloc(max2ndParamSize)) == NULL) {
         perror("Cannot malloc space for xy coords");
         exit(-1);
      }
      */
      plan[gpu].h_gcd = (uint16_t **) malloc(runs * sizeof(uint16_t *));
      for (int i = 0; i < runs; ++i)
         checkCudaErrors(cudaMallocHost((void **) &plan[gpu].h_gcd[i], maxGCDSize * sizeof(uint16_t)));
         //plan[gpu].h_gcd[i] = (uint16_t *) malloc(maxGCDSize * sizeof(uint16_t));
   }

   // Perform work on GPU
   xyCoord * coords = (xyCoord * ) malloc(maxBlocks * sizeof(xyCoord));
   unsigned long long xIdx = 0, yIdx = 0;
   unsigned long long xPrevIdx = 0, yPrevIdx = 0;
   int workNum = 0, index, gpu;
   int x = 0, y = 0;
   for (y = 0; y < segmentDivisionFactor; ++y, ++workNum) {
      for (x = y; x < segmentDivisionFactor; ++x, ++workNum) {
         if (x == y) {
            index = workNum / NUM_GPUS;
            gpu = workNum % NUM_GPUS;

            checkCudaErrors(cudaSetDevice(GPUS_TO_USE[gpu]));

            yIdx = segmentIndices[y];
            plan[gpu].h_yKeys = keys + yPrevIdx * NUM_INTS;
            plan[gpu].yNumKeys = yIdx - yPrevIdx;

            checkCudaErrors(cudaMemcpyAsync(plan[gpu].d_yKeys, plan[gpu].h_yKeys,
                  plan[gpu].yNumKeys * BYTES_PER_KEY, cudaMemcpyHostToDevice,
                  plan[gpu].stream));

            plan[gpu].numBlocks = calculateNumberOfBlocks(plan[gpu].yNumKeys);

            memset(plan[gpu].h_gcd[index], 0, maxGCDSize);
            checkCudaErrors(cudaMemcpyAsync(plan[gpu].d_gcd, plan[gpu].h_gcd[index],
                  plan[gpu].numBlocks * sizeof(uint16_t), cudaMemcpyHostToDevice,
                  plan[gpu].stream));

            int rem = plan[gpu].yNumKeys % BLKDIM > 0 ? 1 : 0;
            dimConversion(plan[gpu].numBlocks, plan[gpu].yNumKeys / BLKDIM + rem,
                  coords);
            plan[gpu].h_keysOrCoords = (uint32_t *) coords;

            checkCudaErrors(cudaMemcpyAsync(plan[gpu].d_keysOrCoords, plan[gpu].h_keysOrCoords,
                  plan[gpu].numBlocks * sizeof(xyCoord), cudaMemcpyHostToDevice,
                  plan[gpu].stream));

            doDiagonalKernel(plan[gpu].d_yKeys,
                  (xyCoord * ) plan[gpu].d_keysOrCoords, plan[gpu].d_gcd,
                  plan[gpu].numBlocks, plan[gpu].yNumKeys);

            checkCudaErrors(cudaMemcpyAsync(plan[gpu].h_gcd[index], plan[gpu].d_gcd,
                  plan[gpu].numBlocks * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                  plan[gpu].stream));

            xPrevIdx = yIdx;
         } else {
            unsigned long long totalXKeys = segmentIndices[x] - xPrevIdx;
            unsigned long long halfStep = totalXKeys / 2 +
               (totalXKeys % 2 ? 1 : 0);
            unsigned long long halfPoint = xPrevIdx + halfStep;

            for (int i = 0; i < 2; ++i, ++workNum) {
               index = workNum / NUM_GPUS;
               gpu = workNum % NUM_GPUS;

               checkCudaErrors(cudaSetDevice(GPUS_TO_USE[gpu]));

               yIdx = segmentIndices[y];
               plan[gpu].h_yKeys = keys + yPrevIdx * NUM_INTS;
               plan[gpu].yNumKeys = yIdx - yPrevIdx;

               checkCudaErrors(cudaMemcpyAsync(plan[gpu].d_yKeys, plan[gpu].h_yKeys,
                     plan[gpu].yNumKeys * BYTES_PER_KEY, cudaMemcpyHostToDevice,
                     plan[gpu].stream));

               xIdx = i == 0 ? halfPoint : segmentIndices[x];

               plan[gpu].h_keysOrCoords = keys + xPrevIdx * NUM_INTS;

               plan[gpu].xNumKeys = xIdx - xPrevIdx;

               size_t xKeysSize = plan[gpu].xNumKeys * BYTES_PER_KEY;
               cudaMemcpyAsync(plan[gpu].d_keysOrCoords, plan[gpu].h_keysOrCoords,
                     xKeysSize, cudaMemcpyHostToDevice, plan[gpu].stream);

               plan[gpu].numBlocks =
                  calculateNumberOfBlocks(plan[gpu].xNumKeys, plan[gpu].yNumKeys);

               memset(plan[gpu].h_gcd[index], 0, maxGCDSize);
               cudaMemcpyAsync(plan[gpu].d_gcd, plan[gpu].h_gcd[index],
                     plan[gpu].numBlocks * sizeof(uint16_t), cudaMemcpyHostToDevice,
                     plan[gpu].stream);

               doUpperKernel(plan[gpu].d_yKeys,
                     plan[gpu].d_keysOrCoords, plan[gpu].d_gcd,
                     plan[gpu].numBlocks, plan[gpu].yNumKeys,
                     plan[gpu].xNumKeys);

               checkCudaErrors(cudaMemcpyAsync(plan[gpu].h_gcd[index], plan[gpu].d_gcd,
                     plan[gpu].numBlocks * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                     plan[gpu].stream));

               xPrevIdx = xIdx;
            }
            --workNum;
         } // a square section in the upper area is finished
      } // a row is finished
      --workNum;
      xPrevIdx = yPrevIdx = yIdx;
   }

   // Wait for all GPU work to finish, and clean up each GPU
   for (int gpu = 0; gpu < NUM_GPUS; ++gpu) {
      checkCudaErrors(cudaSetDevice(GPUS_TO_USE[gpu]));
      checkCudaErrors(cudaStreamSynchronize(plan[gpu].stream));
      checkCudaErrors(cudaFree(plan[gpu].d_yKeys));
      checkCudaErrors(cudaFree(plan[gpu].d_keysOrCoords));
      checkCudaErrors(cudaFree(plan[gpu].d_gcd));
      checkCudaErrors(cudaStreamDestroy(plan[gpu].stream));
   }

   // Process GPU results
   xIdx = 0, yIdx = 0;
   xPrevIdx = 0, yPrevIdx = 0;
   int numBlocks;
   workNum = 0, index, gpu;
   for (y = 0; y < segmentDivisionFactor; ++y, ++workNum) {
      yIdx = segmentIndices[y];
      unsigned long long yNumKeys = yIdx - yPrevIdx;
      for (x = y; x < segmentDivisionFactor; ++x, ++workNum) {
         if (x == y) {
            index = workNum / NUM_GPUS;
            gpu = workNum % NUM_GPUS;

            numBlocks = calculateNumberOfBlocks(plan[gpu].yNumKeys);

            int rem = yNumKeys % BLKDIM > 0 ? 1 : 0;
            dimConversion(numBlocks, yNumKeys / BLKDIM + rem, coords);

            parseGCDResults(numBlocks, badKeyPairList, coords, plan[gpu].h_gcd[index],
                  xPrevIdx, yPrevIdx);
            checkCudaErrors(cudaFreeHost(plan[gpu].h_gcd[index]));

            xPrevIdx = yIdx;
         } else {
            unsigned long long totalXKeys = segmentIndices[x] - xPrevIdx;
            unsigned long long halfStep = totalXKeys / 2 +
               (totalXKeys % 2 ? 1 : 0);
            unsigned long long halfPoint = xPrevIdx + halfStep;
            for (int i = 0; i < 2; ++i, ++workNum) {
               index = workNum / NUM_GPUS;
               gpu = workNum % NUM_GPUS;

               yIdx = segmentIndices[y];
               xIdx = i == 0 ? halfPoint : segmentIndices[x];
               unsigned long long xCurrNumKeys = xIdx - xPrevIdx;
               numBlocks = calculateNumberOfBlocks(xCurrNumKeys, yNumKeys);

               parseGCDResults(numBlocks, badKeyPairList, xCurrNumKeys,
                     yNumKeys, plan[gpu].h_gcd[index], xPrevIdx, yPrevIdx);
               checkCudaErrors(cudaFreeHost(plan[gpu].h_gcd[index]));

               xPrevIdx = xIdx;
            }
            --workNum;
         }
      }
      --workNum;
      xPrevIdx = yPrevIdx = yIdx;
   }
   free(plan[gpu].h_gcd);
   free(coords);
   

   free(segmentIndices);
   hrt_stop();
   printf("CUDA run lasted %s\n", hrt_string());

   processBadKeys(badKeyPairList, keys, NULL, 1, totalNumKeys);
   free(keys);

   return 0;
}

