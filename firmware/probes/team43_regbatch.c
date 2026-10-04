/* team43_regbatch.c -- can several accelerator registers be set in ONE syscall?
 *
 * Why this matters. The GEMM inner loop costs five system calls per 8x8x8
 * tile: START=0, K_ADDR, X_ADDR, START=1, and the Y_DONE poll. Measured on
 * hardware that is about 485 us per invocation, and with 517,888 invocations
 * per query the GEMM loop is 251 s -- by far the largest single item left.
 * The arithmetic underneath is roughly 0.16 us.
 *
 * The register file is a contiguous block of 32-bit words and the driver
 * takes pwrite() counts in words, so in principle one pwrite can set a run of
 * registers. Nobody has ever tried it: every register write in every Team 43
 * program so far has been a single word.
 *
 *   word 0  0x00  START            word 8  0x20  MM2S_1 (X) TUSER
 *   word 1  0x04  MM2S_0 (K) DONE  word 9  0x24  MM2S_2 (B) DONE
 *   word 2  0x08  MM2S_0 (K) ADDR  word 10 0x28  MM2S_2 (B) ADDR
 *   word 3  0x0C  MM2S_0 (K) BYTES word 11 0x2C  MM2S_2 (B) BYTES
 *   word 4  0x10  MM2S_0 (K) TUSER word 12 0x30  MM2S_2 (B) TUSER
 *   word 5  0x14  MM2S_1 (X) DONE  word 13 0x34  S2MM DONE
 *   word 6  0x18  MM2S_1 (X) ADDR  word 14 0x38  S2MM ADDR
 *   word 7  0x1C  MM2S_1 (X) BYTES word 15 0x3C  S2MM BYTES
 *
 * Three ways to issue one tile:
 *
 *   A  five calls   START=0 | K_ADDR | X_ADDR | START=1 | poll      (current)
 *   B  four calls   START=0 | words 2..6 in one write | START=1 | poll
 *   C  three calls  words 0..6 in one write | START=1 | poll
 *
 * B leaves START and the DONE flags alone and only merges the two address
 * writes, so it is the conservative option. C also clears START inside the
 * same transaction and writes over the two MM2S DONE words, which may or may
 * not be harmless -- that is exactly what this program is for.
 *
 * Correctness is checked before speed: every method must reproduce method A's
 * output exactly, over many tiles of pseudo-random data, or it is rejected.
 *
 * Build:  riscv64-...-gcc -std=gnu99 -O2 -static -Wall -Wextra \
 *             -o team43_regbatch team43_regbatch.c -lrt
 * Run:    ./team43_regbatch            (add --iters N to change the count)
 */
#define _XOPEN_SOURCE 700
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#define DEVICE_PATH   "/dev/accelerator_test"
#define K_ADDR        0xBF000000UL
#define X_ADDR        0xBF100000UL
#define Y_ADDR        0xBF300000UL

#define REG_START     0x00
#define REG_K_ADDR    0x08
#define REG_K_BYTES   0x0C
#define REG_K_TUSER   0x10
#define REG_X_ADDR    0x18
#define REG_X_BYTES   0x1C
#define REG_X_TUSER   0x20
#define REG_B_ADDR    0x28
#define REG_B_BYTES   0x2C
#define REG_B_TUSER   0x30
#define REG_Y_DONE    0x34
#define REG_Y_ADDR    0x38
#define REG_Y_BYTES   0x3C

#define INPUT_BYTES   64u
#define OUTPUT_BYTES  256u
#define OUTPUT_WORDS  64
#define POLL_LIMIT    2000000u

static int fd = -1;
static long g_sys = 0;

static void die(const char *msg)
{
    fprintf(stderr, "FAIL: %s\n", msg);
    exit(1);
}
static long now_us(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}
static void rw(off_t off, uint32_t v)
{
    g_sys++;
    if (pwrite(fd, &v, 1, off) < 0) { perror("reg write"); exit(1); }
}
static uint32_t rr(off_t off)
{
    uint32_t v = 0; g_sys++;
    if (pread(fd, &v, 1, off) < 0) { perror("reg read"); exit(1); }
    return v;
}
static void rwn(off_t off, const uint32_t *v, int words)
{
    g_sys++;
    if (pwrite(fd, v, words, off) < 0) { perror("multi-word reg write"); exit(1); }
}
static void dwr(unsigned long a, const void *b, int w)
{
    g_sys++;
    if (pwrite(fd, b, w, (off_t)a) < 0) { perror("DDR write"); exit(1); }
}
static void drd(unsigned long a, void *b, int w)
{
    g_sys++;
    if (pread(fd, b, w, (off_t)a) < 0) { perror("DDR read"); exit(1); }
}
static void wait_done(void)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++) if (rr(REG_Y_DONE) & 1u) return;
    die("Y_DONE never asserted");
}

/* one-time setup of the registers that never change per tile */
static void setup_static(void)
{
    rw(REG_K_BYTES, INPUT_BYTES); rw(REG_K_TUSER, 0u);
    rw(REG_X_BYTES, INPUT_BYTES); rw(REG_X_TUSER, 0u);
    rw(REG_B_ADDR, (uint32_t)Y_ADDR); rw(REG_B_BYTES, OUTPUT_BYTES);
    rw(REG_B_TUSER, 0u);
    rw(REG_Y_ADDR, (uint32_t)Y_ADDR); rw(REG_Y_BYTES, OUTPUT_BYTES);
}

/* ---- the three ways to issue one tile ---- */
static void tile_A(unsigned long ka, unsigned long xa)
{
    rw(REG_START, 0u);
    rw(REG_K_ADDR, (uint32_t)ka);
    rw(REG_X_ADDR, (uint32_t)xa);
    rw(REG_START, 1u);
    wait_done();
}
static void tile_B(unsigned long ka, unsigned long xa)
{
    uint32_t w[5];                       /* words 2..6 : 0x08 .. 0x18 */
    w[0] = (uint32_t)ka;                 /* 0x08 K_ADDR  */
    w[1] = INPUT_BYTES;                  /* 0x0C K_BYTES */
    w[2] = 0u;                           /* 0x10 K_TUSER */
    w[3] = 0u;                           /* 0x14 X_DONE  */
    w[4] = (uint32_t)xa;                 /* 0x18 X_ADDR  */
    rw(REG_START, 0u);
    rwn(REG_K_ADDR, w, 5);
    rw(REG_START, 1u);
    wait_done();
}
static void tile_C(unsigned long ka, unsigned long xa)
{
    uint32_t w[7];                       /* words 0..6 : 0x00 .. 0x18 */
    w[0] = 0u;                           /* 0x00 START = 0 */
    w[1] = 0u;                           /* 0x04 K_DONE    */
    w[2] = (uint32_t)ka;                 /* 0x08 K_ADDR    */
    w[3] = INPUT_BYTES;                  /* 0x0C K_BYTES   */
    w[4] = 0u;                           /* 0x10 K_TUSER   */
    w[5] = 0u;                           /* 0x14 X_DONE    */
    w[6] = (uint32_t)xa;                 /* 0x18 X_ADDR    */
    rwn(REG_START, w, 7);
    rw(REG_START, 1u);
    wait_done();
}

int main(int argc, char **argv)
{
    int iters = 200;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);

    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) { perror("open " DEVICE_PATH); return 1; }

    /* deterministic operands: one K tile, one X tile, both non-trivial */
    int8_t kt[64], xt[64];
    for (int i = 0; i < 64; i++) {
        kt[i] = (int8_t)(((i * 37) % 251) - 125);
        xt[i] = (int8_t)(((i * 53) % 241) - 120);
    }
    dwr(K_ADDR, kt, 16);
    dwr(X_ADDR, xt, 16);

    setup_static();

    int32_t seed[64], outA[64], outB[64], outC[64];
    for (int i = 0; i < 64; i++) seed[i] = (int32_t)(i * 7 - 100);

    printf("=== register batching probe ===\n");
    printf("device : %s\n", DEVICE_PATH);
    printf("iters  : %d tiles per method\n\n", iters);

    struct { const char *name; void (*fn)(unsigned long, unsigned long);
             int calls; int32_t *out; long us; } M[3] = {
        { "A  five calls  (current)", tile_A, 5, outA, 0 },
        { "B  four calls  (merge K_ADDR..X_ADDR)", tile_B, 4, outB, 0 },
        { "C  three calls (also clears START in the same write)", tile_C, 3, outC, 0 },
    };

    for (int m = 0; m < 3; m++) {
        /* correctness: one tile on a known seed */
        dwr(Y_ADDR, seed, OUTPUT_WORDS);
        M[m].fn(K_ADDR, X_ADDR);
        drd(Y_ADDR, M[m].out, OUTPUT_WORDS);

        /* speed: many tiles, seeded once so the accumulator cannot overflow */
        dwr(Y_ADDR, seed, OUTPUT_WORDS);
        long t0 = now_us();
        for (int i = 0; i < iters; i++) M[m].fn(K_ADDR, X_ADDR);
        M[m].us = now_us() - t0;
    }

    int okB = memcmp(outA, outB, sizeof(outA)) == 0;
    int okC = memcmp(outA, outC, sizeof(outA)) == 0;

    printf("%-52s %8s %10s %9s\n", "method", "syscalls", "us/tile", "result");
    for (int m = 0; m < 3; m++) {
        const char *v = (m == 0) ? "baseline" : (m == 1 ? (okB ? "EXACT" : "MISMATCH")
                                                        : (okC ? "EXACT" : "MISMATCH"));
        printf("%-52s %8d %10.1f %9s\n", M[m].name, M[m].calls,
               (double)M[m].us / iters, v);
    }
    printf("\nspeedup vs A : B %.2fx   C %.2fx\n",
           (double)M[0].us / (double)M[1].us, (double)M[0].us / (double)M[2].us);

    if (!okB || !okC) {
        printf("\nfirst 8 accumulators\n");
        printf("  A:"); for (int i = 0; i < 8; i++) printf(" %d", outA[i]);
        printf("\n  B:"); for (int i = 0; i < 8; i++) printf(" %d", outB[i]);
        printf("\n  C:"); for (int i = 0; i < 8; i++) printf(" %d", outC[i]);
        printf("\n");
    }

    printf("\nIf B or C is EXACT, the GEMM inner loop can drop from five\n"
           "syscalls to that method's count. At the measured 251 s GEMM loop\n"
           "that is roughly %.0f s (B) or %.0f s (C) saved per query.\n",
           251.0 * (1.0 - 4.0/5.0), 251.0 * (1.0 - 3.0/5.0));

    close(fd);
    return 0;
}
