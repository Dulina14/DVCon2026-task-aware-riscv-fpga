/* team43_regstress.c -- does register batching survive a REAL GEMM pattern?
 *
 * The first probe (team43_regbatch) ran one tile, 200 times, with the same
 * K_ADDR and X_ADDR every time, and checked a single result. It reported both
 * batched variants EXACT. In the full pipeline the three-call variant then
 * produced complete garbage: 53 detections instead of 4, every confidence
 * pinned at 1.000. The probe was wrong because it exercised none of what the
 * real loop does.
 *
 * This one mirrors gemm_run():
 *
 *   for each (m0, n0) tile-pair
 *       seed Y with the bias vector
 *       for k0 in 0..KT-1
 *           issue one tile with ADDRESSES THAT CHANGE EVERY TIME
 *           (the accelerator accumulates into Y in place)
 *       read Y back
 *
 * so it covers changing addresses, the accumulation chain across k0, and the
 * seed/harvest traffic interleaved between tile groups. Every accumulator of
 * every tile-pair is compared, not one value of one tile.
 *
 * Four ways to issue a tile. The registers sit in a contiguous word array:
 *
 *   word 0 START | word 1 K DONE | word 2 K ADDR  | word 3 K BYTES
 *   word 4 K TUSER | word 5 X DONE | word 6 X ADDR | word 7 X BYTES
 *
 * K_ADDR and X_ADDR are words 2 and 6, so ANY single write that covers both
 * must also pass over word 5, the X DONE flag. That is unavoidable, and it is
 * the prime suspect: Probe E3 already showed that a stale DONE lets the poll
 * return before the operands have actually landed.
 *
 *   A  5 calls  START=0 | K_ADDR | X_ADDR | START=1 | poll      (known good)
 *   B  4 calls  START=0 | words 2..6 | START=1 | poll           (writes X DONE)
 *   C  3 calls  words 0..6 | START=1 | poll                     (both DONEs + START)
 *   E  4 calls  words 0..2 | X_ADDR | START=1 | poll            (writes K DONE only)
 *
 * B, C and E each disturb a different subset, so whichever survives tells us
 * which register is actually fatal rather than leaving it a guess.
 *
 * Method A is itself checked against a software reference first. If A is
 * wrong the harness is wrong, and nothing else it says can be believed.
 *
 * Build:  riscv64-...-gcc -std=gnu99 -O2 -static -Wall -Wextra \
 *             -o team43_regstress team43_regstress.c -lrt
 * Run:    ./team43_regstress              [--rounds N] [--mt N] [--nt N] [--kt N]
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
#define MAXT          16           /* max tiles per dimension */
#define CHUNKW        256

static int fd = -1;

static void die(const char *m) { fprintf(stderr, "FAIL: %s\n", m); exit(1); }
static long now_us(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}
static void rw(off_t o, uint32_t v)
{ if (pwrite(fd, &v, 1, o) < 0) { perror("reg write"); exit(1); } }
static uint32_t rr(off_t o)
{ uint32_t v = 0; if (pread(fd, &v, 1, o) < 0) { perror("reg read"); exit(1); } return v; }
static void rwn(off_t o, const uint32_t *v, int n)
{ if (pwrite(fd, v, n, o) < 0) { perror("multi reg write"); exit(1); } }
static void dwr(unsigned long a, const void *b, int w)
{ if (pwrite(fd, b, w, (off_t)a) < 0) { perror("DDR write"); exit(1); } }
static void drd(unsigned long a, void *b, int w)
{ if (pread(fd, b, w, (off_t)a) < 0) { perror("DDR read"); exit(1); } }
static void dw_bulk(unsigned long a, const void *b, long bytes)
{
    long words = bytes / 4, done = 0; const uint8_t *p = b;
    while (done < words) {
        int n = (int)((words - done) > CHUNKW ? CHUNKW : (words - done));
        dwr(a + (unsigned long)done * 4, p + done * 4, n);
        done += n;
    }
}
static void wait_done(void)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++) if (rr(REG_Y_DONE) & 1u) return;
    die("Y_DONE never asserted");
}

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
    uint32_t w[5];
    w[0] = (uint32_t)ka; w[1] = INPUT_BYTES; w[2] = 0u; w[3] = 0u; w[4] = (uint32_t)xa;
    rw(REG_START, 0u);
    rwn(REG_K_ADDR, w, 5);
    rw(REG_START, 1u);
    wait_done();
}
static void tile_C(unsigned long ka, unsigned long xa)
{
    uint32_t w[7];
    w[0] = 0u; w[1] = 0u; w[2] = (uint32_t)ka; w[3] = INPUT_BYTES;
    w[4] = 0u; w[5] = 0u; w[6] = (uint32_t)xa;
    rwn(REG_START, w, 7);
    rw(REG_START, 1u);
    wait_done();
}
static void tile_E(unsigned long ka, unsigned long xa)
{
    uint32_t w[3];
    w[0] = 0u; w[1] = 0u; w[2] = (uint32_t)ka;   /* START=0, K DONE, K ADDR */
    rwn(REG_START, w, 3);
    rw(REG_X_ADDR, (uint32_t)xa);
    rw(REG_START, 1u);
    wait_done();
}

static int8_t  Ktile[MAXT * MAXT][64];   /* [n0*KT + k0] */
static int8_t  Xtile[MAXT * MAXT][64];   /* [m0*KT + k0] */
static int32_t bias[64];
static int32_t ref[MAXT * MAXT][64];     /* software reference per (m0,n0)  */
static int32_t got[MAXT * MAXT][64];

int main(int argc, char **argv)
{
    int MT = 6, NT = 6, KT = 6, rounds = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mt") && i + 1 < argc) MT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nt") && i + 1 < argc) NT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kt") && i + 1 < argc) KT = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
    }
    if (MT > MAXT || NT > MAXT || KT > MAXT) die("dimension above MAXT");

    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) { perror("open " DEVICE_PATH); return 1; }

    /* operands: every tile distinct, so a stale address shows up immediately */
    uint32_t s = 12345u;
    for (int i = 0; i < NT * KT; i++)
        for (int j = 0; j < 64; j++)
        { s = s * 1103515245u + 12345u; Ktile[i][j] = (int8_t)((s >> 16) % 127 - 63); }
    for (int i = 0; i < MT * KT; i++)
        for (int j = 0; j < 64; j++)
        { s = s * 1103515245u + 12345u; Xtile[i][j] = (int8_t)((s >> 16) % 127 - 63); }
    for (int j = 0; j < 64; j++) bias[j] = (int32_t)(j * 11 - 300);

    /* software reference: partial[cc*8+rr] = bias + sum_k sum_kk X[kk*8+rr]*W[kk*8+cc] */
    for (int m0 = 0; m0 < MT; m0++)
        for (int n0 = 0; n0 < NT; n0++) {
            int32_t *o = ref[m0 * NT + n0];
            memcpy(o, bias, sizeof(bias));
            for (int k0 = 0; k0 < KT; k0++) {
                const int8_t *X = Xtile[m0 * KT + k0], *W = Ktile[n0 * KT + k0];
                for (int cc = 0; cc < 8; cc++)
                    for (int rr2 = 0; rr2 < 8; rr2++) {
                        int32_t a = 0;
                        for (int kk = 0; kk < 8; kk++)
                            a += (int32_t)X[kk * 8 + rr2] * (int32_t)W[kk * 8 + cc];
                        o[cc * 8 + rr2] += a;
                    }
            }
        }

    dw_bulk(K_ADDR, Ktile, (long)NT * KT * 64);
    dw_bulk(X_ADDR, Xtile, (long)MT * KT * 64);

    rw(REG_K_BYTES, INPUT_BYTES); rw(REG_K_TUSER, 0u);
    rw(REG_X_BYTES, INPUT_BYTES); rw(REG_X_TUSER, 0u);
    rw(REG_B_ADDR, (uint32_t)Y_ADDR); rw(REG_B_BYTES, OUTPUT_BYTES);
    rw(REG_B_TUSER, 0u);
    rw(REG_Y_ADDR, (uint32_t)Y_ADDR); rw(REG_Y_BYTES, OUTPUT_BYTES);

    struct { const char *name; void (*fn)(unsigned long, unsigned long); int calls; }
    M[4] = {
        { "A  5 calls  nothing merged        ", tile_A, 5 },
        { "B  4 calls  merges over X DONE    ", tile_B, 4 },
        { "C  3 calls  merges over both DONEs", tile_C, 3 },
        { "E  4 calls  merges over K DONE    ", tile_E, 4 },
    };

    printf("=== register batching, real GEMM pattern ===\n");
    printf("tiles  : MT=%d NT=%d KT=%d  ->  %d tile-pairs, %d tiles per round\n",
           MT, NT, KT, MT * NT, MT * NT * KT);
    printf("rounds : %d\n\n", rounds);
    printf("%-36s %6s %10s %12s %10s\n",
           "method", "calls", "us/tile", "bad accums", "max error");

    int failed_any = 0;
    for (int m = 0; m < 4; m++) {
        long bad_total = 0, maxerr = 0, us_total = 0;
        for (int rd = 0; rd < rounds; rd++) {
            long t0 = now_us();
            for (int m0 = 0; m0 < MT; m0++)
                for (int n0 = 0; n0 < NT; n0++) {
                    dwr(Y_ADDR, bias, OUTPUT_WORDS);
                    for (int k0 = 0; k0 < KT; k0++)
                        M[m].fn(K_ADDR + ((unsigned long)n0 * KT + k0) * 64,
                                X_ADDR + ((unsigned long)m0 * KT + k0) * 64);
                    drd(Y_ADDR, got[m0 * NT + n0], OUTPUT_WORDS);
                }
            us_total += now_us() - t0;

            for (int p = 0; p < MT * NT; p++)
                for (int j = 0; j < 64; j++) {
                    long d = (long)got[p][j] - (long)ref[p][j];
                    if (d) { bad_total++; if (d < 0) d = -d; if (d > maxerr) maxerr = d; }
                }
        }
        long tiles = (long)MT * NT * KT * rounds;
        printf("%-36s %6d %10.1f %12ld %10ld  %s\n",
               M[m].name, M[m].calls, (double)us_total / tiles, bad_total, maxerr,
               bad_total ? "CORRUPT" : "exact");
        if (m == 0 && bad_total) {
            printf("\nMethod A disagrees with the software reference. The harness or the\n"
                   "packing convention is wrong, so nothing below can be trusted.\n");
            close(fd); return 2;
        }
        if (bad_total) failed_any = 1;
    }

    printf("\n%d of %d accumulators checked per method.\n",
           MT * NT * 64 * rounds, MT * NT * 64 * rounds);
    if (failed_any)
        printf("At least one batched variant corrupts results under a real GEMM\n"
               "pattern. Whichever rows read 'exact' name the registers that are\n"
               "safe to merge; the CORRUPT rows name the ones that are not.\n");
    else
        printf("All variants exact. Re-run with --rounds 5 before believing it.\n");

    close(fd);
    return 0;
}
