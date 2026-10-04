/*
 * team43_probe.c  --  DVCon India 2026, Stage 3B, Team 43
 *
 * Driver micro-benchmark + three decisive hardware experiments.
 *
 * WHY: Milestone A measured 2,624 us per 8x8x8 invocation, of which only ~55 us
 * is accelerator time. Removing 11 register writes per tile bought 1.37x, far
 * less than predicted, which means the cost is somewhere else. This program
 * measures each driver primitive in isolation instead of inferring it, then
 * tests three specific ways to remove work from the inner loop.
 *
 * It does NOT change any Milestone A result. It is a measurement tool.
 *
 * Build (dev PC):
 *   cd /home/Team43/Desktop/DVCon_2026/Application
 *   docker run --rm -t -v "$PWD":/workspace -w /workspace --user $(id -u):$(id -g) \
 *     buildroot-2004_new \
 *     bash -c 'buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc \
 *         -std=gnu99 -O2 -static -Wall -Wextra -o team43_probe team43_probe.c -lrt'
 *
 * Run (board):
 *   /tmp/team43/team43_probe
 *
 * Runtime: a few seconds. Nothing is written outside the four DDR scratch
 * windows the organizer's own drv_test_128.c already uses.
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

#define K_ADDR    0xBF000000UL
#define X_ADDR    0xBF100000UL
#define BIAS_ADDR 0xBF200000UL
#define Y_ADDR    0xBF300000UL

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
#define INPUT_WORDS   16
#define OUTPUT_WORDS  64
#define INPUT_BYTES   64u
#define OUTPUT_BYTES  256u
/* Milestone A measured DONE polls = 1.0 per invocation, i.e. the accelerator
 * is always already finished by the first status read. A poll read costs ~90 us
 * on this driver, so 20,000 polls is ~1.8 s -- generous, and it means a failed
 * experiment reports in seconds instead of hanging for minutes. */
#define POLL_LIMIT    20000u

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

static void rw(off_t off, uint32_t v)
{
    if (pwrite(fd, &v, 1, off) < 0)
        die("register write 0x%02lx: %s", (unsigned long)off, strerror(errno));
}

static uint32_t rr(off_t off)
{
    uint32_t v = 0;
    if (pread(fd, &v, 1, off) < 0)
        die("register read 0x%02lx: %s", (unsigned long)off, strerror(errno));
    return v;
}

static void dw(unsigned long addr, const void *buf, int words)
{
    if (pwrite(fd, buf, words, (off_t)addr) < 0)
        die("DDR write 0x%lx (%d words): %s", addr, words, strerror(errno));
}

static void dr(unsigned long addr, void *buf, int words)
{
    if (pread(fd, buf, words, (off_t)addr) < 0)
        die("DDR read 0x%lx (%d words): %s", addr, words, strerror(errno));
}

static void write_constant_regs(void)
{
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
}

static int wait_done(void)
{
    unsigned int g;
    for (g = 0; g < POLL_LIMIT; g++)
        if (rr(REG_Y_DONE) & 1u) return 0;
    return -1;
}

/* ---------------------------------------------------------------- timing -- */

static long g_iters = 300;

#define TIME_BLOCK(label, stmt)                                   \
    do {                                                          \
        long _t0 = now_us();                                      \
        for (long _i = 0; _i < g_iters; _i++) { stmt; }           \
        long _d = now_us() - _t0;                                 \
        printf("  %-38s %8.1f us\n", label, (double)_d / g_iters);\
    } while (0)

static double time_stmt_us(void (*fn)(void))
{
    long t0 = now_us();
    for (long i = 0; i < g_iters; i++) fn();
    return (double)(now_us() - t0) / (double)g_iters;
}

static uint8_t bigbuf[4096];

static void op_lseek(void)   { lseek(fd, 0, SEEK_SET); }
static void op_regw(void)    { rw(REG_START, 0u); }
static void op_regr(void)    { (void)rr(REG_Y_DONE); }
static void op_w64k(void)    { dw(K_ADDR, bigbuf, 16); }
static void op_w64x(void)    { dw(X_ADDR, bigbuf, 16); }
static void op_w256b(void)   { dw(BIAS_ADDR, bigbuf, 64); }
static void op_r256y(void)   { dr(Y_ADDR, bigbuf, 64); }
static void op_r64k(void)    { dr(K_ADDR, bigbuf, 16); }

/* ------------------------------------------------------- tile scaffolding -- */

static int8_t  xt[64], wt[64];
static int32_t par[64], expect[64];

static void make_operands(int seed)
{
    for (int i = 0; i < 64; i++) {
        xt[i] = (int8_t)((i * (seed + 1)) % 7 - 3);
        wt[i] = (int8_t)((i * 3 + seed) % 11 - 5);
    }
}

/* software model of one invocation: par += X*W, using the section-4 packing */
static void tile_sw(const int8_t *x, const int8_t *w, int32_t *acc)
{
    for (int cc = 0; cc < TILE; cc++)
        for (int r = 0; r < TILE; r++) {
            int32_t a = acc[cc * TILE + r];
            for (int kk = 0; kk < TILE; kk++)
                a += (int32_t)x[kk * TILE + r] * (int32_t)w[kk * TILE + cc];
            acc[cc * TILE + r] = a;
        }
}

/* the exact Milestone A hoisted sequence, 8 syscalls */
static long baseline_tile(void)
{
    long t0 = now_us();
    rw(REG_START, 0u);
    dw(K_ADDR, wt, INPUT_WORDS);
    dw(X_ADDR, xt, INPUT_WORDS);
    dw(BIAS_ADDR, par, OUTPUT_WORDS);
    rw(REG_START, 1u);
    if (wait_done() < 0) die("baseline tile timed out");
    dr(Y_ADDR, par, OUTPUT_WORDS);
    rw(REG_START, 0u);
    return now_us() - t0;
}

static int cmp64(const int32_t *a, const int32_t *b, const char *what)
{
    int bad = 0;
    for (int i = 0; i < 64; i++) if (a[i] != b[i]) bad++;
    printf("    %-34s %s (%d/64 match)\n", what,
           bad ? "*** MISMATCH ***" : "match", 64 - bad);
    return bad == 0;
}

/* ============================================================== main ====== */

int main(int argc, char **argv)
{
    const char *dev = DEV_DEFAULT;
    int do_exp = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dev") && i + 1 < argc) dev = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) g_iters = atol(argv[++i]);
        else if (!strcmp(argv[i], "--no-experiments")) do_exp = 0;
        else { fprintf(stderr, "usage: %s [--dev P] [--iters N] [--no-experiments]\n", argv[0]); return 1; }
    }

    fd = open(dev, O_RDWR);
    if (fd < 0) die("cannot open '%s': %s", dev, strerror(errno));
    memset(bigbuf, 0x5a, sizeof(bigbuf));
    write_constant_regs();

    printf("=== TEAM43 DRIVER PROBE ===\n");
    printf("device %s, %ld iterations per measurement\n\n", dev, g_iters);

    /* -------------------------------------------------- 1. primitives --- */
    printf("1. DRIVER PRIMITIVE COST\n");
    double t_lseek = time_stmt_us(op_lseek);
    double t_regw  = time_stmt_us(op_regw);
    double t_regr  = time_stmt_us(op_regr);
    double t_w64k  = time_stmt_us(op_w64k);
    double t_w64x  = time_stmt_us(op_w64x);
    double t_w256  = time_stmt_us(op_w256b);
    double t_r256  = time_stmt_us(op_r256y);
    double t_r64   = time_stmt_us(op_r64k);

    printf("  %-38s %8.1f us   (syscall floor, no device work)\n", "lseek()", t_lseek);
    printf("  %-38s %8.1f us\n", "pwrite 1 word  (register)",  t_regw);
    printf("  %-38s %8.1f us\n", "pread  1 word  (register)",  t_regr);
    printf("  %-38s %8.1f us\n", "pwrite 16 words (64 B)  -> K",  t_w64k);
    printf("  %-38s %8.1f us\n", "pwrite 16 words (64 B)  -> X",  t_w64x);
    printf("  %-38s %8.1f us\n", "pread  16 words (64 B)  <- K",  t_r64);
    printf("  %-38s %8.1f us\n", "pwrite 64 words (256 B) -> BIAS", t_w256);
    printf("  %-38s %8.1f us\n", "pread  64 words (256 B) <- Y",    t_r256);

    /* -------------------------------------------------- 2. size sweep --- */
    printf("\n2. TRANSFER COST vs SIZE (X window)\n");
    printf("  %8s %12s %12s %14s\n", "words", "write us", "read us", "write us/word");
    static const int sizes[] = {1, 4, 16, 32, 64, 128, 256};
    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        long t0 = now_us();
        int failed = 0;
        for (long i = 0; i < g_iters; i++)
            if (pwrite(fd, bigbuf, n, (off_t)X_ADDR) < 0) { failed = 1; break; }
        double tw = failed ? -1.0 : (double)(now_us() - t0) / g_iters;
        t0 = now_us();
        for (long i = 0; i < g_iters; i++)
            if (pread(fd, bigbuf, n, (off_t)X_ADDR) < 0) { failed = 1; break; }
        double tr = failed ? -1.0 : (double)(now_us() - t0) / g_iters;
        if (tw < 0) printf("  %8d %12s %12s\n", n, "unsupported", "unsupported");
        else printf("  %8d %12.1f %12.1f %14.2f\n", n, tw, tr, tw / n);
    }

    /* -------------------------------------------------- 3. baseline ----- */
    printf("\n3. BASELINE INVOCATION (the Milestone A hoisted sequence)\n");
    make_operands(1);
    memset(par, 0, sizeof(par));
    long tb = 0;
    for (long i = 0; i < g_iters; i++) { memset(par, 0, sizeof(par)); tb += baseline_tile(); }
    double t_base = (double)tb / g_iters;
    double t_model = 3 * t_regw + t_regr + 2 * t_w64k + t_w256 + t_r256;
    printf("  measured                               %8.1f us\n", t_base);
    printf("  sum of primitives                      %8.1f us\n", t_model);
    printf("  unexplained (accelerator + packing)    %8.1f us\n", t_base - t_model);
    printf("  Milestone A measured 2624.3 us for this same sequence.\n");

    if (!do_exp) { close(fd); return 0; }

    /* ============================================== 4. EXPERIMENTS ====== */
    printf("\n4. EXPERIMENTS -- can we remove work from the inner loop?\n");

    /* ---- E1: does operand DDR content persist between invocations? ---- */
    printf("\n  E1  Skip re-writing X and K when they are unchanged\n");
    make_operands(2);
    memset(par, 0, sizeof(par));
    memset(expect, 0, sizeof(expect));
    tile_sw(xt, wt, expect);

    rw(REG_START, 0u);
    dw(K_ADDR, wt, INPUT_WORDS);
    dw(X_ADDR, xt, INPUT_WORDS);
    dw(BIAS_ADDR, par, OUTPUT_WORDS);
    rw(REG_START, 1u);
    if (wait_done() < 0) die("E1 first tile timed out");
    dr(Y_ADDR, par, OUTPUT_WORDS);
    rw(REG_START, 0u);
    int e1a = cmp64(par, expect, "first tile vs software");

    int32_t zero[64];
    memset(zero, 0, sizeof(zero));
    rw(REG_START, 0u);
    dw(BIAS_ADDR, zero, OUTPUT_WORDS);      /* bias only, no X, no K */
    rw(REG_START, 1u);
    if (wait_done() < 0) die("E1 second tile timed out");
    dr(Y_ADDR, par, OUTPUT_WORDS);
    rw(REG_START, 0u);
    int e1b = cmp64(par, expect, "re-run without rewriting X/K");
    printf("    VERDICT: %s\n", (e1a && e1b)
        ? "operands PERSIST in DDR -- safe to write X once per k-tile"
        : "operands do NOT persist -- must rewrite every tile");

    /* ---- E2: alias BIAS_ADDR to Y_ADDR so accumulation stays in DDR --- */
    printf("\n  E2  Point BIAS at the Y window (accumulate in DDR, no round trip)\n");
    make_operands(3);
    memset(zero, 0, sizeof(zero));
    memset(expect, 0, sizeof(expect));
    tile_sw(xt, wt, expect);            /* one pass  */
    tile_sw(xt, wt, expect);            /* two passes -> 2*X*W */

    rw(REG_B_ADDR, (uint32_t)Y_ADDR);   /* the aliasing */
    rw(REG_START, 0u);
    dw(K_ADDR, wt, INPUT_WORDS);
    dw(X_ADDR, xt, INPUT_WORDS);
    dw(Y_ADDR, zero, OUTPUT_WORDS);     /* bias := 0, in the Y window */
    rw(REG_START, 1u);
    if (wait_done() < 0) die("E2 first tile timed out");
    rw(REG_START, 0u);

    /* Rewrite the operands so E2's verdict does not depend on E1's. */
    dw(K_ADDR, wt, INPUT_WORDS);
    dw(X_ADDR, xt, INPUT_WORDS);
    rw(REG_START, 1u);                  /* run again, no bias traffic at all */
    if (wait_done() < 0) die("E2 second tile timed out");
    rw(REG_START, 0u);
    dr(Y_ADDR, par, OUTPUT_WORDS);
    int e2 = cmp64(par, expect, "two chained tiles vs 2*X*W");
    rw(REG_B_ADDR, (uint32_t)BIAS_ADDR);   /* restore */
    printf("    VERDICT: %s\n", e2
        ? "DDR-resident accumulation WORKS -- removes a 256 B write AND a 256 B read per tile"
        : "aliasing does NOT accumulate correctly -- keep the host round trip");

    /* ---- E3: are the two START=0 clears necessary? -------------------- */
    printf("\n  E3  Drop the START=0 clears (2 register writes per tile)\n");
    int e3 = 1;
    memset(par, 0, sizeof(par));
    memset(expect, 0, sizeof(expect));
    for (int t = 0; t < 3; t++) {
        make_operands(10 + t);
        tile_sw(xt, wt, expect);
        dw(K_ADDR, wt, INPUT_WORDS);
        dw(X_ADDR, xt, INPUT_WORDS);
        dw(BIAS_ADDR, par, OUTPUT_WORDS);
        rw(REG_START, 1u);                 /* no leading clear */
        if (wait_done() < 0) { printf("    tile %d timed out\n", t); e3 = 0; break; }
        dr(Y_ADDR, par, OUTPUT_WORDS);     /* no trailing clear */
    }
    rw(REG_START, 0u);
    if (e3) e3 = cmp64(par, expect, "3 chained tiles, no clears");
    printf("    VERDICT: %s\n", e3
        ? "clears are NOT required -- saves 2 register writes per tile"
        : "clears ARE required -- keep them");

    /* -------------------------------------------------- 5. projection -- */
    printf("\n5. PROJECTED PER-INVOCATION COST\n");
    double hw = t_base - t_model; if (hw < 0) hw = 0;
    double cur = t_base;
    double clears = e3 ? 0.0 : 2 * t_regw;

    /* The two candidate loop orders are mutually exclusive, so cost them
     * separately rather than pretending every saving composes.
     *
     * PLAN A -- keep n0 outer / k0 inner (Milestone A's order).
     *   One accumulator per column tile lives at Y_ADDR for a whole 48-tile
     *   k-chain, so with E2 the 256 B write and read happen once per chain.
     *   X and K are both rewritten every tile.
     *
     * PLAN B -- k0 outer / n0 inner.
     *   The X tile is constant across the inner loop, so with E1 it is written
     *   48 times instead of 2,304. Each of the 48 column accumulators lives in
     *   its own DDR slot, which costs 2 address-register writes per tile but,
     *   with E2, moves the 256 B traffic out of the loop entirely.
     */
    double planA = clears + t_regw + t_regr + 2 * t_w64k + hw
                 + (e2 ? (t_w256 + t_r256) / 48.0 : (t_w256 + t_r256));
    double planB = clears + t_regw + t_regr + t_w64k + hw
                 + (e1a && e1b ? t_w64k / 48.0 : t_w64k)
                 + (e2 ? 2 * t_regw + (t_w256 + t_r256) / 48.0
                       : t_w256 + t_r256);

    printf("  current (Milestone A order, no changes)      %8.1f us\n", cur);
    printf("  PLAN A  n0 outer, DDR-resident accumulator   %8.1f us   (%.2fx)\n",
           planA, cur / planA);
    printf("  PLAN B  k0 outer, X hoisted, 48 accumulators %8.1f us   (%.2fx)\n",
           planB, cur / planB);
    double best = planA < planB ? planA : planB;
    printf("  -> implement PLAN %s\n", planA < planB ? "A (simpler, and cheaper here)" : "B");

    printf("\n  MiniLM runtime budget, ~178,000 invocations:\n");
    printf("    current   %6.1f s\n", 178000.0 * cur / 1e6);
    printf("    best plan %6.1f s\n", 178000.0 * best / 1e6);
    printf("\n  VEGA software INT32 GEMM ran the same 2,304-invocation projection\n");
    printf("  in 7.54 s; the accelerator did it in 6.05 s (1.25x).\n");
    printf("  At the projected rate it would take %.2f s (%.2fx vs the CPU).\n",
           2304.0 * best / 1e6, 7.54 / (2304.0 * best / 1e6));
    printf("\n  These are PROJECTIONS from measured primitives, not measurements.\n");

    close(fd);
    printf("\nPROBE COMPLETE.\n");
    return 0;
}
