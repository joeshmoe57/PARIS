/*
 * Contains one function which validates the math library. It will print.
 */

#include <stdio.h>
#include <string.h>

#include "keyMath.h"
#include "validate.h"

#define PRINT_PASS(v)   (v == 1 ? "passed" : "failed")

int checkRightShift();
int checkLeftShift();
int checkSub();
int checkEquals();

/*
 * runs the validation and returns.
 */
int main(void){
   validateMath();

   return 0;
}

/*
 * Returns 0 if everything passes, < 0 if things fail.
 */
int validateMath(void){
   int res = -1, totalRes = -5;

   puts("About to validate each function of our 1024 bit math functions...");

   res = checkEquals();
   totalRes += res;
   printf("equals %s\n", PRINT_PASS(res));
   if(res != 1){
      printf("Checking equals returned %d! Bailing\n", res);
      return -1;
   }

   res = checkRightShift();
   totalRes += res;
   printf("right shift %s\n", PRINT_PASS(res));

   res = checkLeftShift();
   totalRes += res;
   printf("left shift %s\n", PRINT_PASS(res));

   res = checkSub();
   totalRes += res;
   printf("subtract %s\n", PRINT_PASS(res));

   return totalRes;
}


/*
 * These are the functions that actually check functionality. They return 1 on
 * a pass, and !1 on fail.
 *  buffer[0] is the LSI
 */
//use it like an equal sign, returns 1 if true

int checkEquals(){
   int res, ret = 1;
   uint32_t buf[NUM_INTS], checkBuf[NUM_INTS];

   memset(buf, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);


   //check 0 int equal
   ret += (res = equalTo(buf, checkBuf));
   ret--;
   if(res != 1)
      puts("\tfailed equal 1");

   buf[0] = 0xFF;
   checkBuf[0] = 0xFF;

   //check 0 int equal
   ret += (res = equalTo(buf, checkBuf));
   ret--;
   if(res != 1)
      puts("\tfailed equal 2");


   //check max int equal
   buf[NUM_INTS -1] = 0x4F;
   checkBuf[NUM_INTS- 1] = 0x4F;

   ret += (res = equalTo(buf, checkBuf));
   ret--;
   if(res != 1)
      puts("\tfailed equal 3");


   //check multiple bytes equal
   buf[NUM_INTS - 2] = 0x0F0F;
   checkBuf[NUM_INTS - 2] = 0x0F0F;

   ret += (res = equalTo(buf, checkBuf));
   ret--;
   if(res != 1)
      puts("\tfailed equal 4");



   //check int 0 not equal
   buf[0] = 0x40;
   checkBuf[0] = 0x4F;

   ret += (res = equalTo(buf, checkBuf));
   if(res != 0)
      puts("\tfailed equal 5");

   //check multiple bytes unequal
   buf[0] = 0x40;
   checkBuf[0] = 0x4F;
   buf[3] = 0x1111;
   checkBuf[4] = 0xCCAA;

   ret += (res = equalTo(buf, checkBuf));
   if(res != 0)
      puts("\tfailed equal 6");

   return ret;
}

int checkRightShift(){
   int ret = 1;;
   uint32_t buf[NUM_INTS], checkBuf[NUM_INTS];

   memset(buf, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[NUM_INTS - 1] = 0x80;
   checkBuf[NUM_INTS - 1] = 0x40;
   rightShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed right shift of 10...0");

   buf[NUM_INTS - 1] = 0x01;
   checkBuf[NUM_INTS - 1] = 0x0;
   checkBuf[NUM_INTS - 2] = 0x80; //0b10000000;
   rightShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed right shift of 000000010...0");

   memset(buf, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[0] = 1;
   checkBuf[0] = 0;
   rightShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed right shift of 0...01");

   buf[0] = 0x80;
   checkBuf[0] = 0x40;
   rightShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed right shift of 0...10000000");

   buf[0] = 0xc0;
   checkBuf[0] = 3;
   rightShift(buf);
   rightShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed double right shift of 0...11000000");

   buf[0] = 0;
   buf[NUM_INTS - 1] = 0x80000000;
   int i;
   for (i = 0; i < BIT_LENGTH; ++i) {
      printNumHex(buf);
      rightShift(buf);
   }
   printNumHex(buf);

   return ret > 0;
}

int checkLeftShift(){
   int ret = 1;
   uint32_t buf[NUM_INTS], checkBuf[NUM_INTS];

   memset(buf, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[NUM_INTS - 1] = 0x40;
   checkBuf[NUM_INTS - 1] = 0x80;
   leftShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed left of 010...0");

   buf[NUM_INTS - 1] = 0x00;
   buf[NUM_INTS - 2] = 0x80;
   checkBuf[NUM_INTS - 1] = 0x01;
   checkBuf[NUM_INTS - 2] = 0x00; 
   leftShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed left shift of 0000000010...0");

   memset(buf, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[NUM_INTS - 1] = 0x80;
   checkBuf[NUM_INTS - 1] = 0;
   leftShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed left shift of 10...0");

   buf[0] = 0x40;
   checkBuf[0] = 0x80;
   leftShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed left shift of 0...01000000");

   buf[0] = 0x03;
   checkBuf[0] = 0xc0;
   leftShift(buf);
   leftShift(buf);
   if (!(ret += equalTo(buf, checkBuf)))
      puts("\tfailed double left shift of 0...00110000");

   buf[0] = 1;
   int i;
   for (i = 0; i < BIT_LENGTH; ++i) {
      printNumHex(buf);
      leftShift(buf);
   }
   printNumHex(buf);

   return ret > 0;
}

int checkSub(){
   int ret = 1, rer;
   uint32_t buf[NUM_INTS], sub[NUM_INTS], res[NUM_INTS], checkBuf[NUM_INTS];

   memset(buf, 0, NUM_BYTES);
   memset(sub, 0, NUM_BYTES);
   memset(res, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[0] = 45;
   sub[0] = 9;
   checkBuf[0] = 36;
   subtract(buf, sub, res);

   if (!(rer = equalTo(res, checkBuf)))
      printf("\tfailed subtract 45 - 9: %u|%u != %u|%u\n", res[1], res[0], checkBuf[1], checkBuf[0]);
   ret += rer;
   
   memset(buf, 0, NUM_BYTES);
   memset(sub, 0, NUM_BYTES);
   memset(res, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[1] = 1;
   sub[0] = 9;
   checkBuf[0] = UINT_MAX - 9 + 1;
   subtract(buf, sub, res);

   if (!(rer = equalTo(res, checkBuf)))
      puts("\tfailed subtract INT_MAX - 9 ");
   ret += rer;

   memset(buf, 0, NUM_BYTES);
   memset(sub, 0, NUM_BYTES);
   memset(res, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[1] = 3;
   sub[1] = 1;
   checkBuf[1] = 2;
   subtract(buf, sub, res);

   if (!(rer = equalTo(res, checkBuf)))
      puts("\tfailed subtract INT_MAX + 2");
   ret += rer;

   memset(buf, 0, NUM_BYTES);
   memset(sub, 0, NUM_BYTES);
   memset(res, 0, NUM_BYTES);
   memset(checkBuf, 0, NUM_BYTES);

   buf[NUM_INTS - 1] = 1<<31;
   sub[0] = 1;
   int i;
   for (i = 0; i < NUM_INTS - 1; ++i)
      checkBuf[i] = 0 - 1;//xffffffff;
   checkBuf[NUM_INTS - 1] = 0x7fffffff;
   subtract(buf, sub, res);

   if (!(rer = equalTo(res, checkBuf))) {
      puts("\tfailed subtract most significant int - 1");
      printNumHex(res);
      printNumHex(checkBuf);
   }
   ret += rer;

   return ret > 0;
}
