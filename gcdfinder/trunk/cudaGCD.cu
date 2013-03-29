#include "cudaGCD.h"

#define DEBUG 1

extern "C" {
#include "lpem.h"
#include "keyMath.h"
}

//Kernel Function
/* TODO Figure out if de_coords could be moved into shared memory */
__global__ void GCD_Compare_All(unsigned *x_dev, uint16_t *gcd_dev, xyCoord * dev_coord, int numBlocks) {
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

int main(int argc, char**argv) {
   long totalNumKeys;
   std::vector<RSA*> privKeys;
   uint32_t *keys;

   if(argc != 2) {
       printf("Wrong number of args (Only number of keys)");
       exit(1);
   }

   //get number of keys to process
   sscanf(argv[1], "%d", &totalNumKeys);
   // TODO assumption that totalNumKeys is a multiple of BLKDIM is being made

   if((keys = (uint32_t *) malloc(totalNumKeys * NUM_INTS * sizeof(uint32_t)))
       == 0) {
      perror("Cannot Malloc Key Vector");
      exit(1);
   }

   totalNumKeys = getAllKeys(KEYS_DB, totalNumKeys, &privKeys, keys);

   dprint("getKeys returns %d\n", totalNumKeys);

   // Print what is at memory where the keys will be going
   /*
#if DEBUG
   for (int i = 0; i < totalNumKeys; ++i) 
      printNumHex(keys + i * NUM_INTS);
   fflush(stdout);
#endif
*/

   /* This is a modified version of the geometic expansion of the sum from 1 to totalNumKeys/BLKDIM*/
   /* This is also a generic solution that allows for multiple versions of BLKDIM. Since for our implementation, this value is fixed, one could also just use:
    * int numBlocks = (totalNumKeys * totalNumKeys + 4 * totalNumKeys) >> 5;
    */
   int sq = BLKDIM * BLKDIM * 2;
   int shift = 0;
   while (!(sq & 1)) {
      sq >>= 1;
      ++shift;
   }

   dprint("totalNumKeys = %d; shift = %d\n", totalNumKeys, shift);

   /* TODO Assumes totalNumberOfKeys is less than sqrt(INT_MAX) */ 
   long numBlocks = (totalNumKeys * totalNumKeys + BLKDIM * totalNumKeys) >> (shift);
   printf("totalNumKeys = %d; BLKDIM = %d; shift = %d\n product = %d product = %d\n", 
         totalNumKeys, BLKDIM, shift, totalNumKeys * totalNumKeys, BLKDIM * totalNumKeys);

   dprint("numBlocks = %ld\n", numBlocks);

   //unsigned int *gcd_res = (unsigned int * ) malloc(numBlocks * BLKDIM * BLKDIM * NUM_INTS * sizeof(int));
   uint16_t * gcd_res = (uint16_t * ) malloc(numBlocks * sizeof(uint16_t));
   if (gcd_res == NULL) {
      fprintf(stderr, "Error with malloc\n");
      exit(-1);
   }

   xyCoord * coords = (xyCoord * ) malloc(numBlocks * sizeof(xyCoord));

   dprint("calling dimConversion with %d, %d\n", numBlocks, totalNumKeys / BLKDIM);

   dimConversion(numBlocks, totalNumKeys / BLKDIM, coords);

   // Print the coordinates of the blocks if they were in a square

   /*
#if DEBUG
   for (int i = 0; i < numBlocks; ++i) 
      printf("(%d, %d)\n", coords[i].x, coords[i].y);
   fflush(stdout);
#endif
*/

   unsigned int * dev_keys;
   uint16_t * dev_gcd;
   xyCoord * dev_coords;

   int ret = 0;
   /* Sizes added since they are used many times. Trying to keep consistency 
    * so changes don't break things.*/
   int dev_keysSize = totalNumKeys * NUM_INTS * sizeof(int);
   //int dev_gcdSize = numBlocks * BLKDIM * BLKDIM * NUM_INTS * sizeof(int);
   int dev_gcdSize = numBlocks * sizeof(uint16_t);
   memset(gcd_res, 0, dev_gcdSize);

   ret = cudaMalloc((void **)&dev_keys, dev_keysSize);
   dprint("cudaMalloc:%s\n", cudaGetErrorString(cudaGetLastError()));

   dprint("%d bytes for dev_keys\n", dev_keysSize);
   dprint("malloc: %d\n", ret);
   dprint("%d bytes for dev_gcd\n", dev_gcdSize);

   ret = cudaMalloc((void **)&dev_gcd, dev_gcdSize);
   dprint("cudaMalloc:%s\n", cudaGetErrorString(cudaGetLastError()));
   cudaMemcpy(dev_gcd, gcd_res, dev_gcdSize, cudaMemcpyHostToDevice);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   dprint("malloc: %d\n", ret);

   ret = cudaMalloc((void **)&dev_coords, numBlocks * sizeof(xyCoord));
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   dprint("malloc: %d\n", ret);
   
   ret = cudaMemcpy(dev_coords, coords, numBlocks * sizeof(xyCoord), cudaMemcpyHostToDevice);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   dprint("memcopy: %d\n", ret);

   ret = cudaMemcpy(dev_keys, keys, dev_keysSize, cudaMemcpyHostToDevice);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   dprint("memcopy: %d\n", ret);

   int dimGridx = numBlocks > MAX_BLOCK_DIM ? MAX_BLOCK_DIM : numBlocks;
   int dimy = 1 + numBlocks / MAX_BLOCK_DIM;
   int dimGridy = 1 < dimy ? dimy : 1;
   dim3 dimGrid(dimGridx, dimGridy); 
   dim3 dimBlock(NUM_INTS, BLKDIM, BLKDIM);


   dprint("dimGrid = %d %d %d; dimBlock = %d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

   //hrt_start();
   GCD_Compare_All<<<dimGrid, dimBlock>>>(dev_keys, dev_gcd, dev_coords, numBlocks);
   dprint("kernel:%s\n", cudaGetErrorString(cudaGetLastError()));
   cudaThreadSynchronize();
   dprint("cudaThreadSynchronize:%s\n", cudaGetErrorString(cudaGetLastError()));

   //hrt_stop();
   //fprintf(stderr, "Kernel took %s.\n", hrt_string());

   ret = cudaMemcpy(gcd_res, dev_gcd, dev_gcdSize, cudaMemcpyDeviceToHost);
   dprint("cudaMemcpy:%s\n", cudaGetErrorString(cudaGetLastError()));

   dprint("memcopy: %d\n", ret);

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

   free(gcd_res);
   return 0;
}

