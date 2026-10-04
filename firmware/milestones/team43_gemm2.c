/*
 * team43_gemm2.c  --  DVCon India 2026, Stage 3B, Team 43
 *
 * Prototype of the Milestone B GEMM kernel, measured A/B against the
 * Milestone A kernel in the same run, on the same operands.
 *
 * WHAT THE PROBE TAUGHT US (measured on the board, 16 August):
 *   syscall floor (lseek)      25.3 us
 *   register write / read      65.3 / 62.1 us
 *   64 B DDR write            188.8 us     <- ~155 us of that is fixed overhead
 *   256 B DDR write / read    267.5 / 274.7 us
 *   marginal cost above 16 words: ~1.6 us/word
 *
 * The transfer cost is almost entirely PER CALL, not per byte. A 64-byte write
 * costs 188 us; a 1024-byte write costs 562 us. So the winning move is not to
 * shrink the transfers but to STOP MAKING THEM: the address registers are full
 * 32-bit pointers, so every tile of every operand can be uploaded to DDR once,
 * in a handful of big transfers, and the inner loop reduced to pointing the
 * accelerator at the right offsets.
 *
 *   Milestone A inner loop : 8 syscalls, 4 of them data transfers
 *   this inner loop        : 5 syscalls, ALL of them 4-byte register pokes
 *
 * Operand upload becomes 147 KB of W + 3 KB of X in ~150 chunked writes, once,
 * ~85 ms total, instead of 2,304 x 2 small writes.
 *
 * THIS PROGRAM PROVES OR DISPROVES THAT. It runs the real Milestone A shape --
 * X[8,384] * W[384,384] + bias, 2,304 invocations -- both ways, and checks both
 * against an exact software INT32 GEMM.
 *
 * Build (dev PC):
 *   cd /home/Team43/Desktop/DVCon_2026/Application
 *   docker run --rm -t -v "$PWD":/workspace -w /workspace --user $(id -u):$(id -g) \
 *     buildroot-2004_new \
 *     bash -c 'buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc \
 *         -std=gnu99 -O2 -static -Wall -Wextra -o team43_gemm2 team43_gemm2.c -lrt'
 *
 * Run (board):
 *   /tmp/team43/team43_gemm2
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define DEV_DEFAULT "/dev/accelerator_test"

#define K_ADDR    0xBF000000UL      /* 1 MB window: holds ALL 2,304 W tiles */
#define X_ADDR    0xBF100000UL      /* holds all 48 X tiles                 */
#define BIAS_ADDR 0xBF200000UL      /* Milestone A path only                */
#define Y_ADDR    0xBF300000UL      /* the accumulator                      */

#define REG_START   0x00
#define REG_K_ADDR  0x08
#define REG_K_BYTES 0x0C
#define REG_K_TUSER 0x10
#define REG_X_ADDR  0x18
#define REG_X_BYTES 0x1C
#define REG_X_TUSER 0x20
#define REG_B_ADDR  0x28
#define REG_B_BYTES 0x2C
#define REG_B_TUSER 0x30
#define REG_Y_DONE  0x34
#define REG_Y_ADDR  0x38
#define REG_Y_BYTES 0x3C

#define TILE          8
#define M             8
#define KDIM          384
#define NDIM          384
#define KT            (KDIM / TILE)      /* 48 */
#define NT            (NDIM / TILE)      /* 48 */
#define INPUT_WORDS   16
#define OUTPUT_WORDS  64
#define INPUT_BYTES   64u
#define OUTPUT_BYTES  256u
#define POLL_LIMIT    20000u
#define CHUNK_WORDS   256                /* largest size the probe confirmed  */

static int fd = -1;

static long now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(2);
}

static long g_syscalls = 0;

static void rw(off_t off, uint32_t v)
{
    g_syscalls++;
    if (pwrite(fd, &v, 1, off) < 0)
        die("register write 0x%02lx: %s", (unsigned long)off, strerror(errno));
}
static uint32_t rr(off_t off)
{
    uint32_t v = 0;
    g_syscalls++;
    if (pread(fd, &v, 1, off) < 0)
        die("register read 0x%02lx: %s", (unsigned long)off, strerror(errno));
    return v;
}
static void dw(unsigned long addr, const void *buf, int words)
{
    g_syscalls++;
    if (pwrite(fd, buf, words, (off_t)addr) < 0)
        die("DDR write 0x%lx (%d words): %s", addr, words, strerror(errno));
}
static void dr(unsigned long addr, void *buf, int words)
{
    g_syscalls++;
    if (pread(fd, buf, words, (off_t)addr) < 0)
        die("DDR read 0x%lx (%d words): %s", addr, words, strerror(errno));
}

/* upload an arbitrary byte range in CHUNK_WORDS-sized pieces */
static void dw_bulk(unsigned long addr, const void *buf, long bytes)
{
    const uint8_t *p = (const uint8_t *)buf;
    long words = bytes / 4;
    long done = 0;
    while (done < words) {
        int n = (int)((words - done) > CHUNK_WORDS ? CHUNK_WORDS : (words - done));
        dw(addr + (unsigned long)done * 4, p + done * 4, n);
        done += n;
    }
}

static int wait_done(void)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++)
        if (rr(REG_Y_DONE) & 1u) return 0;
    return -1;
}

/* ======================================================= operands ========= */

static int8_t  X[M][KDIM];
static int8_t  W[KDIM][NDIM];     /* [in][out], already in accelerator order  */
static int32_t bias[NDIM];
static int32_t y_sw[M][NDIM], y_a[M][NDIM], y_b[M][NDIM];

/* packed tile stores, uploaded to DDR once */
static int8_t packW[NT * KT * 64];   /* 147,456 B, indexed (n0*KT + k0)      */
static int8_t packX[KT * 64];        /*   3,072 B, indexed k0                */

static void build_operands(void)
{
    for (int r = 0; r < M; r++)
        for (int k = 0; k < KDIM; k++)
            X[r][k] = (int8_t)(((r * 37 + k * 11) % 61) - 30);
    for (int k = 0; k < KDIM; k++)
        for (int c = 0; c < NDIM; c++)
            W[k][c] = (int8_t)(((k * 17 + c * 5) % 71) - 35);
    for (int c = 0; c < NDIM; c++)
        bias[c] = (c % 13) - 6;
}

static void pack_all(void)
{
    for (int k0 = 0; k0 < KT; k0++) {
        int8_t *t = packX + k0 * 64;
        for (int kk = 0; kk < TILE; kk++)
            for (int r = 0; r < TILE; r++)
                t[kk * TILE + r] = (r < M) ? X[r][k0 * TILE + kk] : 0;
    }
    for (int n0 = 0; n0 < NT; n0++)
        for (int k0 = 0; k0 < KT; k0++) {
            int8_t *t = packW + ((long)n0 * KT + k0) * 64;
            for (int kk = 0; kk < TILE; kk++)
                for (int cc = 0; cc < TILE; cc++)
                    t[kk * TILE + cc] = W[k0 * TILE + kk][n0 * TILE + cc];
        }
}

static void gemm_sw(void)
{
    for (int r = 0; r < M; r++)
        for (int c = 0; c < NDIM; c++) {
            int32_t a = bias[c];
            for (int k = 0; k < KDIM; k++)
                a += (int32_t)X[r][k] * (int32_t)W[k][c];
            y_sw[r][c] = a;
        }
}

static int check(const int32_t got[M][NDIM], const char *label)
{
    long bad = 0;
    int fr = -1, fc = -1;
    for (int r = 0; r < M; r++)
        for (int c = 0; c < NDIM; c++)
            if (got[r][c] != y_sw[r][c]) {
                if (bad == 0) { fr = r; fc = c; }
                bad++;
            }
    printf("  %-28s %s  %ld/%d match\n", label,
           bad ? "*** MISMATCH ***" : "EXACT", (long)(M * NDIM) - bad, M * NDIM);
    if (bad) printf("      first bad [%d][%d]: got %d expected %d\n",
                    fr, fc, got[fr][fc], y_sw[fr][fc]);
    return bad == 0;
}

/* ============================================ KERNEL A (Milestone A) ====== */

static void kernel_a(void)
{
    int8_t xt[64], wt[64];
    int32_t par[64];

    rw(REG_K_ADDR,  (uint32_t)K_ADDR);
    rw(REG_K_BYTES, INPUT_BYTES);
    rw(REG_K_TUSER, 0u);
    rw(REG_X_ADDR,  (uint32_t)X_ADDR);
    rw(REG_X_BYTES, INPUT_BYTES);
    rw(REG_X_TUSER, 0u);
    rw(REG_B_ADDR,  (uint32_t)BIAS_ADDR);
    rw(REG_B_BYTES, OUTPUT_BYTES);
    rw(REG_B_TUSER, 0u);
    rw(REG_Y_ADDR,  (uint32_t)Y_ADDR);
    rw(REG_Y_BYTES, OUTPUT_BYTES);

    for (int n0 = 0; n0 < NT; n0++) {
        for (int cc = 0; cc < TILE; cc++)
            for (int r = 0; r < TILE; r++)
                par[cc * TILE + r] = bias[n0 * TILE + cc];

        for (int k0 = 0; k0 < KT; k0++) {
            for (int kk = 0; kk < TILE; kk++) {
                for (int r = 0; r < TILE; r++)
                    xt[kk * TILE + r] = (r < M) ? X[r][k0 * TILE + kk] : 0;
                for (int cc = 0; cc < TILE; cc++)
                    wt[kk * TILE + cc] = W[k0 * TILE + kk][n0 * TILE + cc];
            }
            rw(REG_START, 0u);
            dw(K_ADDR, wt, INPUT_WORDS);
            dw(X_ADDR, xt, INPUT_WORDS);
            dw(BIAS_ADDR, par, OUTPUT_WORDS);
            rw(REG_START, 1u);
            if (wait_done() < 0) die("kernel A timeout at n0=%d k0=%d", n0, k0);
            dr(Y_ADDR, par, OUTPUT_WORDS);
            rw(REG_START, 0u);
        }
        for (int cc = 0; cc < TILE; cc++)
            for (int r = 0; r < M; r++)
                y_a[r][n0 * TILE + cc] = par[cc * TILE + r];
    }
}

/* ============================================ KERNEL B (preloaded) ======== */
/*
 * Every W tile and every X tile already sits in DDR. BIAS is aliased onto the
 * Y window so the accumulator never leaves the device. The inner loop is five
 * 4-byte register pokes and nothing else.
 *
 * The START=0 is KEPT deliberately. Probe E3 showed three chained tiles are
 * correct without it -- but in that test each tile was separated by ~270 us of
 * bias/result traffic, which gave the accelerator time to finish. Here the next
 * poke is only ~65 us away, and without a clear, Y_DONE never deasserts, so the
 * poll would return a STALE completion and we would read the previous tile's
 * result. 65 us is the price of having a real completion signal.
 */
static void kernel_b(void)
{
    int32_t par[64];

    rw(REG_K_BYTES, INPUT_BYTES);
    rw(REG_K_TUSER, 0u);
    rw(REG_X_BYTES, INPUT_BYTES);
    rw(REG_X_TUSER, 0u);
    rw(REG_B_ADDR,  (uint32_t)Y_ADDR);      /* alias: accumulate in place */
    rw(REG_B_BYTES, OUTPUT_BYTES);
    rw(REG_B_TUSER, 0u);
    rw(REG_Y_ADDR,  (uint32_t)Y_ADDR);
    rw(REG_Y_BYTES, OUTPUT_BYTES);

    for (int n0 = 0; n0 < NT; n0++) {
        for (int cc = 0; cc < TILE; cc++)
            for (int r = 0; r < TILE; r++)
                par[cc * TILE + r] = bias[n0 * TILE + cc];
        dw(Y_ADDR, par, OUTPUT_WORDS);              /* seed, once per column */

        for (int k0 = 0; k0 < KT; k0++) {
            rw(REG_START, 0u);
            rw(REG_K_ADDR, (uint32_t)(K_ADDR + ((unsigned long)n0 * KT + k0) * 64));
            rw(REG_X_ADDR, (uint32_t)(X_ADDR + (unsigned long)k0 * 64));
            rw(REG_START, 1u);
            if (wait_done() < 0) die("kernel B timeout at n0=%d k0=%d", n0, k0);
        }

        dr(Y_ADDR, par, OUTPUT_WORDS);              /* harvest, once per column */
        for (int cc = 0; cc < TILE; cc++)
            for (int r = 0; r < M; r++)
                y_b[r][n0 * TILE + cc] = par[cc * TILE + r];
    }
}

/* ================================================================ main === */

int main(int argc, char **argv)
{
    const char *dev = DEV_DEFAULT;
    int run_a = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dev") && i + 1 < argc) dev = argv[++i];
        else if (!strcmp(argv[i], "--skip-a")) run_a = 0;
        else { fprintf(stderr, "usage: %s [--dev P] [--skip-a]\n", argv[0]); return 1; }
    }

    fd = open(dev, O_RDWR);
    if (fd < 0) die("cannot open '%s': %s", dev, strerror(errno));

    printf("=== TEAM43 GEMM KERNEL A/B ===\n");
    printf("shape X[%d,%d] * W[%d,%d] + bias, %d invocations per kernel\n\n",
           M, KDIM, KDIM, NDIM, KT * NT);

    build_operands();
    pack_all();

    long t0 = now_us();
    gemm_sw();
    printf("software INT32 GEMM on VEGA : %.3f s\n\n", (now_us() - t0) / 1e6);

    long a_us = 0, a_sys = 0;
    if (run_a) {
        printf("KERNEL A -- Milestone A inner loop (8 syscalls, 4 transfers)\n");
        g_syscalls = 0;
        t0 = now_us();
        kernel_a();
        a_us = now_us() - t0;
        a_sys = g_syscalls;
        printf("  wall time                    %.3f s\n", a_us / 1e6);
        printf("  per invocation               %.1f us\n", (double)a_us / (KT * NT));
        printf("  syscalls                     %ld (%.1f per invocation)\n",
               a_sys, (double)a_sys / (KT * NT));
        check(y_a, "vs software");
        printf("\n");
    }

    printf("KERNEL B -- preloaded operands, register-only inner loop\n");
    g_syscalls = 0;
    t0 = now_us();
    dw_bulk(X_ADDR, packX, sizeof(packX));
    dw_bulk(K_ADDR, packW, sizeof(packW));
    long up_us = now_us() - t0;
    long up_sys = g_syscalls;
    printf("  operand upload               %.3f s  (%ld bytes, %ld syscalls)\n",
           up_us / 1e6, (long)(sizeof(packX) + sizeof(packW)), up_sys);

    g_syscalls = 0;
    t0 = now_us();
    kernel_b();
    long b_us = now_us() - t0;
    long b_sys = g_syscalls;
    printf("  wall time (loop only)        %.3f s\n", b_us / 1e6);
    printf("  wall time (incl. upload)     %.3f s\n", (b_us + up_us) / 1e6);
    printf("  per invocation               %.1f us\n", (double)b_us / (KT * NT));
    printf("  syscalls                     %ld (%.1f per invocation)\n",
           b_sys, (double)b_sys / (KT * NT));
    int okb = check(y_b, "vs software");

    printf("\n--- RESULT ---\n");
    if (run_a) {
        printf("kernel A : %.3f s   %.1f us/inv\n", a_us / 1e6, (double)a_us / (KT * NT));
        printf("kernel B : %.3f s   %.1f us/inv   speedup %.2fx\n",
               (b_us + up_us) / 1e6, (double)b_us / (KT * NT),
               (double)a_us / (double)(b_us + up_us));
        printf("identical results : %s\n",
               memcmp(y_a, y_b, sizeof(y_a)) == 0 ? "yes" : "NO");
    }
    if (okb) {
        double per = (double)b_us / (KT * NT);
        printf("\nMiniLM projection at %.1f us/invocation:\n", per);
        printf("  ~178,000 invocations -> %.1f s per query\n", 178000.0 * per / 1e6);
    } else {
        printf("\nKernel B is WRONG -- do not adopt it. Keep the Milestone A loop.\n");
    }

    close(fd);
    return okb ? 0 : 1;
}
