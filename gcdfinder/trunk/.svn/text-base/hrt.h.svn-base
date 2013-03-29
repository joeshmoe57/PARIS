#ifndef HRT_H_
#define HRT_H_

/**
 * HRT
 * Bob Somers 2012
 *
 * Just a simple wrapper for the system's high-resolution timer, specifically
 * the one that is not affected by time jumps (settimeofday()), clock skew
 * (NTP, etc.), or processor affinity on SMP systems.
 *
 * Requires Linux. Not affected by time skew with kernel 2.6.28+.
 *
 * Sorry Windows guys. Look into QueryPerformanceCounter instead. Or perhaps
 * get a better operating system. (I kid... I kid...)
 *
 * You must link with the real-time library. Adding -lrt to your linker line
 * should do it.
 */

#include <stdint.h>

#define HRT_MAX_STR_LEN 24

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Call this right before you the thing you want to time.
 */
void hrt_start();

/**
 * Call this right after the thing you want to time.
 */
void hrt_stop();

/**
 * Returns a 64-bit unsigned integer representing the number of nanoseconds
 * between calls to hrt_start() and hrt_stop().
 *
 * This will work as long as you're timing things that take less than
 * 584.5 years. If it takes longer than that, the results are undefined, and
 * frankly, your code is probably wrong.
 *
 * If you're going to print this, make sure you're using the proper format
 * code for a 64-bit unsigned int. DO NOT cast it to make it print with %d.
 * You may potentially lose precision. If in doubt, see hrt_string(), below.
 */
uint64_t hrt_result();

/**
 * Returns a statically allocated string that is the number of nanoseconds
 * between the last hrt_start() and hrt_stop() with no precision loss. In
 * other words, hrt_result() converted to a string.
 *
 * Statically allocated means that I have no responsibility to keep the
 * contents of the returned char* from changing after I hand it to you. In
 * other words, if you want to save this, you should strcpy() out of this
 * string for safe keeping.
 *
 * The value of the string will not change between uses of the timer. For
 * example, this will work fine:
 *
 *     hrt_start();
 *     do_work();
 *     hrt_stop();
 *     printf("First result: %s\n", hrt_string());
 *
 *     hrt_start();
 *     do_more_work();
 *     hrt_stop();
 *     printf("Second result: %s\n, hrt_string());
 *
 *     > First result: 12345ns
 *     > Second result: 67890ns
 *
 * This however, will not:
 *
 *     hrt_start();
 *     do_work();
 *     hrt_stop();
 *     const char* first_result = hrt_string();
 *
 *     hrt_start();
 *     do_more_work();
 *     hrt_stop();
 *     const char* second_result = hrt_string();
 *
 *     printf("First result: %s\n", first_result);
 *     printf("Second result: %s\n", second_result);
 *
 *     > First result: 67890ns
 *     > Second result: 67890ns
 *
 * If you really don't want to print the string out in between timing things,
 * you can just strcpy it. A buffer of at least HRT_MAX_STR_LEN will work
 * fine:
 *
 *     hrt_start();
 *     do_work();
 *     hrt_stop();
 *     char first_result[HRT_MAX_STR_LEN];
 *     strncpy(first_result, hrt_string(), HRT_MAX_STR_LEN);
 *
 *     hrt_start();
 *     do_more_work();
 *     hrt_stop();
 *     char* second_result[HRT_MAX_STR_LEN];
 *     strncpy(second_result, hrt_string(), HRT_MAX_STR_LEN);
 *
 *     printf("First result: %s\n", first_result);
 *     printf("Second result: %s\n", second_result);
 *
 *     > First result: 12345ns
 *     > Second result: 67890ns
 *
 */
const char* hrt_string();

#ifdef __cplusplus
}
#endif

#endif /* HRT_H_ */
