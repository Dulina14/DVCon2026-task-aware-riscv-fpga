/* team43_kdepth.c -- can one invocation accumulate the whole K dimension?
 *
 * WHY THIS MATTERS
 *
 * Every GEMM in the pipeline is issued as (M/8) x (N/8) x (K/8) invocations.
 * The k0 loop is by far the largest factor: measured on the deployed graph the
 * mean is ~38 K-tiles per output tile, so 320 px costs 2,339,064 invocations
 * for only 62,196 output tiles. At ~50 us per system call and 5 calls per
 * invocation that is 587 s of the 713 s query, and essentially none of it is
 * arithmetic.
 *
 * The RTL says this loop does not have to exist. In axis_sa the accumulator is
 * reset by m_first, which follows s_last; in top_axi_int s_last is
 * s_k_tlast & s_x_tlast, which the DMA asserts at the end of a descriptor; and
 * in dma_controller the descriptor length is simply the MM2S_n_BYTES register,
 * a full 32-bit field fed straight through. Nothing anywhere fixes K at 8.
 * Stream 8*K bytes instead of 64 and the array should accumulate all K in one
 * invocation and emit the same 8x8 tile.
 *
 * The packed buffers already suit this. gemm_run puts the X tile for (m0,k0)
 * at (m0*KT + k0)*64 and the W tile for (n0,k0) at (n0*KT + k0)*64, so for a
 * fixed m0 (or n0) the whole K run is already contiguous and in k-major order,
 * which is exactly the beat order the array wants.
 *
 * THE HAZARD THIS ALSO TESTS
 *
 * wait_done() polls Y_DONE but nothing ever clears it, so after the first tile
 * it is permanently high and the "poll" is really just one syscall's worth of
 * delay -- about 50 us. That is comfortably longer than a K=8 tile, which is
 * why the pipeline is correct today. A K=1024 tile streams 1024 beats at
 * 50 MHz, about 20.5 us, plus DMA latency. Still under 50 us, but the margin is
 * now a guess rather than an accident, and it shrinks as K grows.
 *
 * So three methods are measured, not two:
 *
 *   A  KT invocations of K=8, seed and harvest Y around them   (what ships)
 *   B  one invocation of the full K, same one-read wait        (fast, racy?)
 *   C  one invocation of the full K, Y_DONE cleared then polled properly
 *
 * Correctness is decided against a software reference computed here, never
 * against method A, so a wrong A cannot make B or C look right.
 *
 * Build:  riscv64-...-gcc -std=gnu99 -O2 -static -Wall -Wextra \
 *             -o team43_kdepth team43_kdepth.c -lrt
 * Run:    ./team43_kdepth                 (add --iters N, default 20)
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

#define DEVICE_PATH  "/dev/accelerator_test"
#define K_ADDR       0xBF000000UL
#define X_ADDR       0xBF100000UL
#define Y_ADDR       0xBF300000UL

#define REG_START    0x00
#define REG_K_ADDR   0x08
#define REG_K_BYTES  0x0C
#define REG_K_TUSER  0x10
#define REG_X_ADDR   0x18
#define REG_X_BYTES  0x1C
#define REG_X_TUSER  0x20
#define REG_B_ADDR   0x28
#define REG_B_BYTES  0x2C
#define REG_B_TUSER  0x30
#define REG_Y_DONE   0x34
#define REG_Y_ADDR   0x38
#define REG_Y_BYTES  0x3C

#define TILE          8
#define OUTPUT_BYTES  256u
#define OUTPUT_WORDS  64
#define KMAX          1024              /* largest K this probe will try */
#define POLL_LIMIT    2000000u

static int  fd = -1;
static long g_sys = 0;

static void die(const char *m) { fprintf(stderr, "FAIL: %s\n", m); exit(1); }

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
static void dwr(unsigned long a, const void *b, long w)
{
    g_sys++;
    if (pwrite(fd, b, w, (off_t)a) < 0) { perror("DDR write"); exit(1); }
}
static void drd(unsigned long a, void *b, long w)
{
    g_sys++;
    if (pread(fd, b, w, (off_t)a) < 0) { perror("DDR read"); exit(1); }
}

/* the wait the pipeline actually uses: one read, and Y_DONE is never cleared,
 * so after the first tile this is just a syscall's worth of delay */
static void wait_loose(void) { (void)rr(REG_Y_DONE); }

/* clear Y_DONE first, then poll it for real */
static void wait_strict(void)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++)
        if (rr(REG_Y_DONE) & 1u) return;
    die("Y_DONE never asserted");
}

/* ---------------- operands ------------------------------------------- */
/* X is k-major: byte k*8 + r.  W is k-major: byte k*8 + c.  That is exactly
 * one AXI beat per k, which is what sx_data[R] and sk_data[C] consume. */
static int8_t  XB[KMAX * TILE], WB[KMAX * TILE];
static int32_t REF[64], GOT[64], SEED[64];

static void build(int K, unsigned salt)
{
    for (int k = 0; k < K; k++)
        for (int i = 0; i < TILE; i++) {
            XB[k*TILE + i] = (int8_t)(((k*37 + i*11 + salt*5) % 251) - 125);
            WB[k*TILE + i] = (int8_t)(((k*53 + i*17 + salt*3) % 241) - 120);
        }
    for (int i = 0; i < 64; i++) SEED[i] = (int32_t)((i * 7 - 100) + (int)salt);

    /* reference: par[c*8+r] = seed + sum_k X[k*8+r] * W[k*8+c] */
    for (int c = 0; c < TILE; c++)
        for (int r = 0; r < TILE; r++) {
            int32_t a = SEED[c*TILE + r];
            for (int k = 0; k < K; k++)
                a += (int32_t)XB[k*TILE + r] * (int32_t)WB[k*TILE + c];
            REF[c*TILE + r] = a;
        }
}

static void upload(int K)
{
    dwr(K_ADDR, WB, (long)K * TILE / 4);   /* counts are in 32-bit words */
    dwr(X_ADDR, XB, (long)K * TILE / 4);
}

static void static_regs(uint32_t kbytes)
{
    rw(REG_K_BYTES, kbytes); rw(REG_K_TUSER, 0u);
    rw(REG_X_BYTES, kbytes); rw(REG_X_TUSER, 0u);
    rw(REG_B_ADDR, (uint32_t)Y_ADDR); rw(REG_B_BYTES, OUTPUT_BYTES);
    rw(REG_B_TUSER, 0u);
    rw(REG_Y_ADDR, (uint32_t)Y_ADDR); rw(REG_Y_BYTES, OUTPUT_BYTES);
}

/* A: the shipping method -- KT invocations of K=8 */
static void run_A(int K)
{
    int KT = K / TILE;
    static_regs(64u);
    dwr(Y_ADDR, SEED, OUTPUT_WORDS);
    for (int k0 = 0; k0 < KT; k0++) {
        rw(REG_START, 0u);
        rw(REG_K_ADDR, (uint32_t)(K_ADDR + (unsigned long)k0 * 64));
        rw(REG_X_ADDR, (uint32_t)(X_ADDR + (unsigned long)k0 * 64));
        rw(REG_START, 1u);
        wait_loose();
    }
    drd(Y_ADDR, GOT, OUTPUT_WORDS);
}

/* B: one invocation over the whole K, same loose wait */
static void run_B(int K)
{
    static_regs((uint32_t)(K * TILE));
    dwr(Y_ADDR, SEED, OUTPUT_WORDS);
    rw(REG_START, 0u);
    rw(REG_K_ADDR, (uint32_t)K_ADDR);
    rw(REG_X_ADDR, (uint32_t)X_ADDR);
    rw(REG_START, 1u);
    wait_loose();
    drd(Y_ADDR, GOT, OUTPUT_WORDS);
}

/* C: one invocation, Y_DONE cleared then genuinely polled */
static void run_C(int K)
{
    static_regs((uint32_t)(K * TILE));
    dwr(Y_ADDR, SEED, OUTPUT_WORDS);
    rw(REG_Y_DONE, 0u);
    rw(REG_START, 0u);
    rw(REG_K_ADDR, (uint32_t)K_ADDR);
    rw(REG_X_ADDR, (uint32_t)X_ADDR);
    rw(REG_START, 1u);
    wait_strict();
    drd(Y_ADDR, GOT, OUTPUT_WORDS);
}

static int compare(long *worst)
{
    long w = 0;
    for (int i = 0; i < 64; i++) {
        long d = (long)GOT[i] - (long)REF[i];
        if (d < 0) d = -d;
        if (d > w) w = d;
    }
    *worst = w;
    return w == 0;
}

int main(int argc, char **argv)
{
    int iters = 20;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);

    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) { perror("open " DEVICE_PATH); return 1; }

    const int KS[] = { 8, 16, 64, 128, 256, 512, 1024 };
    const int NK = (int)(sizeof(KS)/sizeof(KS[0]));

    printf("=== K-depth probe: does one invocation accumulate all of K? ===\n");
    printf("device : %s\n", DEVICE_PATH);
    printf("iters  : %d random operand sets per K\n", iters);
    printf("\nA = %d-tile loop (shipping)   B = one invocation, loose wait\n", TILE);
    printf("C = one invocation, Y_DONE cleared and polled\n\n");
    printf("%6s %10s %10s %10s   %10s %10s %10s   %8s\n",
           "K", "A us", "B us", "C us", "A", "B", "C", "speedup");
    printf("%6s %10s %10s %10s   %10s %10s %10s   %8s\n",
           "-----", "--------", "--------", "--------",
           "--------", "--------", "--------", "-------");

    int all_B_ok = 1, all_C_ok = 1;

    for (int ki = 0; ki < NK; ki++) {
        int K = KS[ki];
        long wA = 0, wB = 0, wC = 0;
        int okA = 1, okB = 1, okC = 1;
        long tA = 0, tB = 0, tC = 0;

        for (int it = 0; it < iters; it++) {
            long w; long t0;

            build(K, (unsigned)(it + 1));
            upload(K);

            t0 = now_us(); run_A(K); tA += now_us() - t0;
            if (!compare(&w)) okA = 0;
            if (w > wA) wA = w;

            t0 = now_us(); run_B(K); tB += now_us() - t0;
            if (!compare(&w)) okB = 0;
            if (w > wB) wB = w;

            t0 = now_us(); run_C(K); tC += now_us() - t0;
            if (!compare(&w)) okC = 0;
            if (w > wC) wC = w;
        }

        char sA[16], sB[16], sC[16];
        snprintf(sA, sizeof sA, okA ? "EXACT" : "ERR %ld", wA);
        snprintf(sB, sizeof sB, okB ? "EXACT" : "ERR %ld", wB);
        snprintf(sC, sizeof sC, okC ? "EXACT" : "ERR %ld", wC);

        printf("%6d %10.1f %10.1f %10.1f   %10s %10s %10s   %7.1fx\n",
               K, (double)tA/iters, (double)tB/iters, (double)tC/iters,
               sA, sB, sC, tC ? (double)tA / (double)tC : 0.0);
        fflush(stdout);

        if (!okA) {
            printf("\n  method A is wrong at K=%d, which should not happen.\n"
                   "  Stop here: the reference or the operand layout is at fault,\n"
                   "  not the K-depth idea.\n", K);
            close(fd);
            return 2;
        }
        if (!okB) all_B_ok = 0;
        if (!okC) all_C_ok = 0;
    }

    printf("\n--- verdict ---\n");
    if (all_C_ok) {
        printf("  C is exact at every K. One invocation can accumulate the whole\n"
               "  K dimension. gemm_run can drop its k0 loop, which on the\n"
               "  measured graph is 27-30x fewer system calls.\n");
        if (all_B_ok)
            printf("  B is exact too, so the loose one-read wait still covers the\n"
                   "  longer transfer. Use C anyway: it costs one syscall per tile\n"
                   "  and does not depend on a timing coincidence.\n");
        else
            printf("  B is NOT exact, which is the expected failure: the single-read\n"
                   "  wait no longer outlasts the transfer. This is exactly why C\n"
                   "  clears Y_DONE and polls. Use C.\n");
    } else {
        printf("  C failed. Long descriptors do not accumulate correctly.\n"
               "  Look at the largest K that passed and cap --kdepth there;\n"
               "  a partial collapse still removes most of the system calls.\n");
    }
    printf("\n  syscalls issued by this probe: %ld\n", g_sys);

    close(fd);
    return all_C_ok ? 0 : 1;
}
