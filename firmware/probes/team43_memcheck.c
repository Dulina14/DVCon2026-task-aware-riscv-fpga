/* team43_memcheck.c -- is DRAM healthy on this FPGA programming?
 *
 * team43_kdepth exercises the accelerator and the syscall path, and both can
 * look perfect while the pipeline runs four times too slow. The reason is that
 * kdepth's whole working set is about 8 KB, so it never leaves cache. The
 * pipeline streams multi-megabyte tensors, and on 27 August a power cycle plus
 * reprogram left the board with DRAM at roughly a quarter of its usual
 * bandwidth: memcpy fell from 7.9 to 1.9 MB/s, per-pixel colour conversion went
 * from 2.5 to 51.2 us, and a 256 px query took 1108 s instead of 275 s. Every
 * accelerator number was normal throughout.
 *
 * This probe needs no device, no model files and no network, so it can run the
 * moment Linux boots -- before spending eight minutes transferring 46 MB to a
 * board that is going to be too slow to demonstrate.
 *
 * HEALTHY (measured 26 Aug):    copy >= 6 MB/s, scan >= 6 M elem/s
 * DEGRADED (measured 27 Aug):   copy ~1.9 MB/s, scan ~0.9 M elem/s
 *
 * Build:  riscv64-...-gcc -std=gnu99 -O2 -static -Wall -Wextra \
 *             -o team43_memcheck team43_memcheck.c -lrt
 */
#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MB (1024L*1024L)
#define N  (4L*MB)          /* bigger than any plausible cache */

static double now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(void)
{
    char  *a = malloc(N), *b = malloc(N);
    float *f = malloc(N);
    if (!a || !b || !f) { fprintf(stderr, "FAIL: out of memory\n"); return 1; }
    for (long i = 0; i < N; i++) a[i] = (char)(i * 7);
    for (long i = 0; i < (long)(N/sizeof(float)); i++) f[i] = (float)(i % 1000) - 500.0f;

    printf("=== DRAM health check ===\n");
    printf("working set: %ld MB per buffer, well beyond cache\n\n", N/MB);

    double t = now_s();
    for (int r = 0; r < 3; r++) memcpy(b, a, N);
    double copy_mb = (3.0 * N / MB) / (now_s() - t);

    t = now_s();
    float amax = 0.0f;
    long nf = N / sizeof(float);
    for (int r = 0; r < 3; r++)
        for (long i = 0; i < nf; i++) { float v = f[i] < 0 ? -f[i] : f[i];
                                        if (v > amax) amax = v; }
    double scan_me = (3.0 * nf / 1e6) / (now_s() - t);

    t = now_s();
    long acc = 0;
    for (long i = 0; i < N; i += 64) acc += a[i];        /* one byte per line */
    double stride_us = (now_s() - t) * 1e6 / (N / 64);

    printf("  memcpy            : %8.2f MB/s      (healthy >= 6, degraded ~1.9)\n", copy_mb);
    printf("  sequential scan   : %8.2f M elem/s  (healthy >= 6, degraded ~0.9)\n", scan_me);
    printf("  strided read      : %8.2f us/line   (one byte every 64)\n", stride_us);
    printf("  (checksum %ld, amax %.1f -- ignore, keeps the loops honest)\n\n", acc, (double)amax);

    if (copy_mb >= 6.0 && scan_me >= 6.0)
        printf("  VERDICT: DRAM healthy. A 256 px query should take about 275 s.\n");
    else if (copy_mb < 3.0 || scan_me < 3.0)
        printf("  VERDICT: DRAM DEGRADED. Do not deploy -- power cycle and\n"
               "           reprogram the FPGA, then run this again. A query in\n"
               "           this state takes ~1100 s instead of ~275 s.\n");
    else
        printf("  VERDICT: marginal. Reprogram and re-check before demonstrating.\n");

    free(a); free(b); free(f);
    return 0;
}
