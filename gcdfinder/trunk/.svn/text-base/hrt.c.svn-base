#include "hrt.h"

#include <time.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdio.h>

#ifdef CLOCK_MONOTONIC_RAW
    /* Prefer the raw monotonic clock to avoid time skew. */
#   define HRT_CLOCK CLOCK_MONOTONIC_RAW
#else
    /* Fall back on the regular monotonic clock. */
#   define HRT_CLOCK CLOCK_MONOTONIC
#endif

static char hrt_str[HRT_MAX_STR_LEN];
static struct timespec hrt_begin;
static struct timespec hrt_end;

void hrt_start() {
    clock_gettime(HRT_CLOCK, &hrt_begin);
}

void hrt_stop() {
    clock_gettime(HRT_CLOCK, &hrt_end);
}

uint64_t hrt_result() {
    uint64_t duration = 0;

    if (hrt_end.tv_nsec < hrt_begin.tv_nsec) {
        duration += (hrt_end.tv_sec - hrt_begin.tv_sec - 1) * 1000000000;
        duration += (hrt_end.tv_nsec + 1000000000) - hrt_begin.tv_nsec;
    } else {
        duration += (hrt_end.tv_sec - hrt_begin.tv_sec) * 1000000000;
        duration += hrt_end.tv_nsec - hrt_begin.tv_nsec;
    }

    return duration;
}

const char* hrt_string() {
    uint64_t duration = hrt_result();
    snprintf(hrt_str, HRT_MAX_STR_LEN, "%" PRIu64 "ns", duration);
    return hrt_str;
}

#undef HRT_CLOCK
