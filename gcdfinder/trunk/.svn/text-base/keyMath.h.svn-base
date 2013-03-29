/*
 * These are some functions for working with integers represented in buffers
 * of arbitrary size. They each take a buffer (or two) to work on and the size
 * of that buffer. They expect buffers to be stored in little endian! (with
 * byte 0 at address 0, not byte 4 at address 0).
 */
#include <stdint.h>
#include <limits.h>

#define BIT_LENGTH 1024
#define NUM_INTS   32
#define NUM_BYTES   (NUM_INTS * 4)

/*
 * This struct holds onto a single number in the form of a buffer and length
 */
typedef struct bigNum{
    uint8_t*    b;
    //length of the buffer B in terms of what? Bytes?
    int         len;
}bigNum_t;

void printNumHex(uint32_t buf[NUM_INTS]);
void printNumBin(uint32_t buf[NUM_INTS]);
void printByteBin(uint32_t b);

void rightShift(uint32_t value[NUM_INTS]);
void leftShift(uint32_t value[NUM_INTS]);
void AND(uint32_t u[NUM_INTS], uint32_t v[NUM_INTS], uint32_t w[NUM_INTS]);
void OR(uint32_t u[NUM_INTS], uint32_t v[NUM_INTS], uint32_t w[NUM_INTS]);

int equalsZero(uint32_t v[NUM_INTS]);
int equalTo(uint32_t u[NUM_INTS], uint32_t v[NUM_INTS]);
int LSB(uint32_t v[NUM_INTS]);

void assign(uint32_t v[NUM_INTS], uint32_t u[NUM_INTS]);
int greaterThan(uint32_t u[NUM_INTS], uint32_t v[NUM_INTS]);

void subtract(uint32_t u[NUM_INTS], uint32_t v[NUM_INTS], uint32_t w[NUM_INTS]);
void gcd1024(uint32_t u[NUM_INTS], uint32_t v[NUM_INTS], uint32_t w[NUM_INTS]);

int bufAdd(bigNum_t* out, bigNum_t* u, bigNum_t* v);
int bufSub(uint8_t* u, int uBits, uint8_t* v, int vBits);
