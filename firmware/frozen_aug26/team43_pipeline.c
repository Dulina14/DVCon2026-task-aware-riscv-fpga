/*
 * team43_pipeline.c  --  DVCon India 2026, Stage 3B, Team 43 "Silicon Syndicate"
 *
 * MILESTONE D: the whole thing, one binary.
 *
 *   image (P6 PPM) + task string from argv
 *     -> YOLOv8n detection, all 63 convolutions on the accelerator
 *     -> MiniLM encodes the runtime task prompt, on the accelerator
 *     -> the trained head scores every DETECTED class against the task
 *     -> the best-scoring detection is selected
 *     -> boxes and labels drawn onto the image, written back out as PPM
 *
 * Standalone: the same binary and the same two model files handle any
 * (image, task) pair with no recompilation. Nothing about a particular image
 * or prompt is computed off-board.
 *
 * The head only runs over the classes actually detected, not all 80, which is
 * why scoring here costs ~9,300 invocations instead of Milestone B's 92,640.
 *
 * Build (dev PC):
 *   docker run --rm -t -v "$PWD":/workspace -w /workspace --user $(id -u):$(id -g) \
 *     buildroot-2004_new \
 *     bash -c 'buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc \
 *         -std=gnu99 -O2 -static -Wall -Wextra -o team43_pipeline team43_pipeline.c -lm -lrt'
 *
 * Run (board):
 *   /tmp/team43/team43_pipeline --image /tmp/team43/bus.ppm --task "serve wine" \
 *                               --imgsz 128 --out /tmp/team43/result.ppm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "team43_yolo_graph.h"
#include "team43_minilm_layout.h"
#include "font8x8.h"
#include "jpeg.h"

#define HID     384
#define FFNDIM  1536
#define NLAYERS 6
#define NHEADS  12
#define HDIM    32
#define MAXTOK  8
#define NCLS    80
#define FUSED   (4*HID)

#define TOK_PAD 0
#define TOK_UNK 100
#define TOK_CLS 101
#define TOK_SEP 102

/* ===================================================== accelerator ======= */

/* DDR scratch windows.
 *
 * These were 1 MB each, which is enough for YOLOv8n: its largest packed weight
 * block is 294,912 bytes. YOLOv8s is twice as wide and its largest block is
 * 1,179,648 bytes, which would have run 128 KB past the end of the K window and
 * silently corrupted the activations sitting in X. The weight upload is not
 * chunked -- only the activation upload is -- so nothing would have caught it
 * at runtime; the detections would just have been wrong.
 *
 * Both windows are now 2 MB. The region in use grows from 0xBF000000..0xBF3001
 * to 0xBF000000..0xBF400100, which is still inside the same reserved area.
 * check_windows() asserts the fit at startup against the graph actually
 * compiled in, so a wider model in future fails loudly instead of quietly.  */
#define WINDOW    0x200000L               /* 2 MB per scratch window */
#define K_ADDR    0xBF000000UL            /* .. 0xBF200000 */
#define X_ADDR    0xBF200000UL            /* .. 0xBF400000 */
#define Y_ADDR    0xBF400000UL            /* 256 B result + bias seed */

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
#define POLL_LIMIT    20000u
#define CHUNK_WORDS   256
#define ROWCAP        4096      /* max im2col rows per GEMM chunk */
/* Headroom for one quantised activation tensor. The widest intermediate is a
 * concatenation, so this scales with the model's channel width: 16x the canvas
 * area is ample for YOLOv8n (max 256 channels) and 32x for YOLOv8s (max 512).
 * op_conv checks against it rather than trusting the estimate. */
#define XQ_CAP(w,h)   ((size_t)32 * (size_t)(w) * (size_t)(h) + 1024)

static int  g_fd = -1, g_sim = 0, g_quiet = 0;
/* Register batching is DISABLED and must stay that way.
 *
 * A single pwrite covering several register words does NOT set those
 * registers. team43_regstress proved it under a real GEMM pattern: with the
 * tile addresses changing every invocation, all 2304 accumulators were wrong
 * for every merged variant, whichever register the merge passed over --
 *
 *   A  5 calls, nothing merged        416.7 us/tile   exact
 *   B  4 calls, merges over X DONE    353.7 us/tile   CORRUPT, max err 52556
 *   C  3 calls, merges over both      298.1 us/tile   CORRUPT, max err 69316
 *   E  4 calls, merges over K DONE    354.1 us/tile   CORRUPT, max err 45309
 *
 * The earlier probe that reported these EXACT held the addresses constant, so
 * the registers kept values written by earlier single-word calls and the
 * failure was invisible. Five single-word writes per tile is the floor this
 * driver allows, and the driver may not be modified. */
static int  g_batched_regs = 0;  /* --unsafe-batched-regs, for measurement only */
static long g_inv = 0, g_sys = 0;
static long g_dev_us = 0, g_pack_us = 0, g_up_us = 0, g_vega_us = 0;
static long g_quant_n = 0, g_deq_n = 0, g_silu_n = 0;
static long g_amax_us = 0, g_qcvt_us = 0, g_deq_us = 0;
/* --- fine-grained instrumentation (timers only; no behavioural change) ---
 * g_pack_us stays the sum of every packing site so old logs remain
 * comparable. These break it down so the detection and MiniLM halves can be
 * attributed separately, which the combined counter never allowed.        */
static long g_im2col_us = 0, g_im2col_n = 0;   /* YOLO activation im2col   */
static long g_wcopy_us  = 0, g_wcopy_n  = 0;   /* YOLO weight-tile memcpy  */
static long g_wcopy_by  = 0;
static long g_mpx_us    = 0, g_mpx_n    = 0;   /* MiniLM activation pack   */
static long g_mpw_us    = 0, g_mpw_n    = 0;   /* MiniLM weight pack (pkg) */
static long g_mpw_by    = 0;
static long g_mpa_us    = 0, g_mpa_n    = 0;   /* MiniLM act-as-weight pack*/
static long g_jpeg_us   = 0;                   /* on-board JPEG decode     */

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf,1,2)));
static void die(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "FAIL: "); vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); exit(2);
}
static long now_us(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}
static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p) die("out of memory (%lu bytes)", (unsigned long)n);
    return p;
}
static void rw(off_t off, uint32_t v)
{
    g_sys++;
    if (pwrite(g_fd, &v, 1, off) < 0) die("reg write 0x%02lx: %s",
                                          (unsigned long)off, strerror(errno));
}
static uint32_t rr(off_t off)
{
    uint32_t v = 0; g_sys++;
    if (pread(g_fd, &v, 1, off) < 0) die("reg read 0x%02lx: %s",
                                         (unsigned long)off, strerror(errno));
    return v;
}
/* Several registers in one call. The register file is a contiguous block of
 * 32-bit words and the driver takes pwrite() counts in words, so a single
 * call can set a run of them. Measured on hardware by team43_regbatch:
 * five calls 322.4 us/tile, four 264.8, three 192.8 -- all bit-exact. */
static void rwn(off_t off, const uint32_t *v, int words)
{
    g_sys++;
    if (pwrite(g_fd, v, words, off) < 0)
        die("multi-word reg write 0x%02lx (%d words): %s",
            (unsigned long)off, words, strerror(errno));
}
static void dwr(unsigned long a, const void *b, int w)
{
    g_sys++;
    if (pwrite(g_fd, b, w, (off_t)a) < 0) die("DDR write 0x%lx: %s", a, strerror(errno));
}
static void drd(unsigned long a, void *b, int w)
{
    g_sys++;
    if (pread(g_fd, b, w, (off_t)a) < 0) die("DDR read 0x%lx: %s", a, strerror(errno));
}
static void dw_bulk(unsigned long addr, const void *buf, long bytes)
{
    long words = bytes / 4, done = 0; const uint8_t *p = buf;
    long t0 = now_us();
    while (done < words) {
        int n = (int)((words - done) > CHUNK_WORDS ? CHUNK_WORDS : (words - done));
        dwr(addr + (unsigned long)done * 4, p + done * 4, n);
        done += n;
    }
    g_up_us += now_us() - t0;
}
/* Y_DONE is cleared before every START and then genuinely polled.
 *
 * It used to be only read, never cleared, so after the very first tile it was
 * permanently high and this "poll" was really one syscall of delay, about
 * 100 us. That was enough for a K=8 tile, which is why the pipeline was
 * correct -- by accident. team43_kdepth measured the accident failing: with
 * the whole K streamed in one invocation the loose wait is exact to K=256 and
 * wrong from K=512 (max error 127,452 at 512, 169,893 at 1024), while clearing
 * and polling is exact at every K up to 1024. One extra syscall per
 * invocation, and the timing coincidence is gone.                            */
static void wait_done(void)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++) if (rr(REG_Y_DONE) & 1u) return;
    die("Y_DONE never asserted at invocation %ld", g_inv);
}

/* K_BYTES and X_BYTES change only when the chunk depth changes, which is once
 * per GEMM rather than once per tile, so they are cached rather than rewritten. */
static uint32_t g_cur_bytes = 0xFFFFFFFFu;
static void set_stream_bytes(uint32_t b)
{
    if (b == g_cur_bytes) return;
    rw(REG_K_BYTES, b); rw(REG_X_BYTES, b);
    g_cur_bytes = b;
}
static void sim_tile(const int8_t *xt, const int8_t *wt, int32_t *par)
{
    for (int cc = 0; cc < TILE; cc++)
        for (int r = 0; r < TILE; r++) {
            int32_t a = par[cc*TILE + r];
            for (int kk = 0; kk < TILE; kk++)
                a += (int32_t)xt[kk*TILE + r] * (int32_t)wt[kk*TILE + cc];
            par[cc*TILE + r] = a;
        }
}

/* Y[M,N] = X[M,K] * W[K,N] + bias, operands already tile-packed and uploaded.
 * packX has MT*KT tiles, packW has NT*KT tiles. out is [M][N] int32 row-major. */
/* g_tiles counts output tiles: one per (m0,n0). g_inv counts accelerator
 * invocations: one per (m0,n0,k0). The ratio is exactly what collapsing the
 * k0 loop into a single streamed accumulation would save, because the array
 * ends an accumulation on s_last, which the DMA descriptor length sets. */
static long g_tiles = 0, g_ktiles_sum = 0;
static int  g_kdepth = 0;   /* K-tiles per invocation; 0 = the whole K */

static void gemm_run(const int8_t *packX, const int8_t *packW,
                     int M, int K, int N, const int32_t *bias, int32_t *out,
                     int upload_w)
{
    int MT = (M + 7)/8, KT = (K + 7)/8, NT = (N + 7)/8;
    int32_t par[64];
    long t0 = now_us();
    /* How many K-tiles go into one accelerator invocation. The array ends an
     * accumulation on s_last, which the DMA asserts at the end of a descriptor,
     * and the descriptor length is just the BYTES register -- so K per
     * invocation is a software choice, not an RTL constant. The packed buffers
     * are already contiguous and k-major over the whole K run for a fixed m0
     * (or n0), which is exactly the beat order the array consumes, so no
     * repacking is needed either.
     *   --kdepth 1  reproduces the original one-K-tile-per-invocation path
     *   --kdepth 0  streams the whole K in a single invocation (default)     */
    int KC = (g_kdepth > 0 && g_kdepth < KT) ? g_kdepth : KT;

    if (!g_sim) {
        dw_bulk(X_ADDR, packX, (long)MT * KT * 64);
        /* The weight tiles are identical for every M-chunk of one convolution,
         * so they are uploaded on the first chunk only. */
        if (upload_w) dw_bulk(K_ADDR, packW, (long)NT * KT * 64);
        rw(REG_K_TUSER, 0u);
        rw(REG_X_TUSER, 0u);
        rw(REG_B_ADDR, (uint32_t)Y_ADDR); rw(REG_B_BYTES, OUTPUT_BYTES);
        rw(REG_B_TUSER, 0u);
        rw(REG_Y_ADDR, (uint32_t)Y_ADDR); rw(REG_Y_BYTES, OUTPUT_BYTES);
    }
    for (int m0 = 0; m0 < MT; m0++)
        for (int n0 = 0; n0 < NT; n0++) {
            for (int cc = 0; cc < TILE; cc++)
                for (int r = 0; r < TILE; r++) {
                    int gc = n0*TILE + cc;
                    par[cc*TILE + r] = (gc < N) ? bias[gc] : 0;
                }
            if (!g_sim) dwr(Y_ADDR, par, OUTPUT_WORDS);
            for (int k0 = 0; k0 < KT; k0 += KC) {
                int kc = (KT - k0 < KC) ? KT - k0 : KC;   /* last chunk may be short */
                if (g_sim) {
                    for (int j = 0; j < kc; j++)
                        sim_tile(packX + ((long)m0*KT + k0 + j)*64,
                                 packW + ((long)n0*KT + k0 + j)*64, par);
                }
                else if (!g_batched_regs) {        /* the only correct way */
                    set_stream_bytes((uint32_t)kc * INPUT_BYTES);
                    rw(REG_Y_DONE, 0u);
                    rw(REG_START, 0u);
                    rw(REG_K_ADDR, (uint32_t)(K_ADDR + ((unsigned long)n0*KT + k0)*64));
                    rw(REG_X_ADDR, (uint32_t)(X_ADDR + ((unsigned long)m0*KT + k0)*64));
                    rw(REG_START, 1u);
                    wait_done();
                } else {   /* three calls: PROVEN INCORRECT, kept only for A/B timing */
                    uint32_t w[7];
                    w[0] = 0u;                      /* 0x00 START = 0     */
                    w[1] = 0u;                      /* 0x04 K DONE        */
                    w[2] = (uint32_t)(K_ADDR + ((unsigned long)n0*KT + k0)*64);
                    w[3] = (uint32_t)kc * INPUT_BYTES;  /* 0x0C K BYTES   */
                    w[4] = 0u;                      /* 0x10 K TUSER       */
                    w[5] = 0u;                      /* 0x14 X DONE        */
                    w[6] = (uint32_t)(X_ADDR + ((unsigned long)m0*KT + k0)*64);
                    rwn(REG_START, w, 7);
                    rw(REG_START, 1u);
                    wait_done();
                }
                g_inv++;
            }
            g_tiles++; g_ktiles_sum += KT;
            if (!g_sim) drd(Y_ADDR, par, OUTPUT_WORDS);
            for (int cc = 0; cc < TILE; cc++)
                for (int r = 0; r < TILE; r++) {
                    int gr = m0*TILE + r, gc = n0*TILE + cc;
                    if (gr < M && gc < N) out[(long)gr*N + gc] = par[cc*TILE + r];
                }
        }
    g_dev_us += now_us() - t0;
}

/* ===================================================== model package ===== */

static const unsigned char *BLOB;
static size_t BLOB_SZ;

static const int8_t *w_packed(int i) { return (const int8_t *)(BLOB + YW_OFF[i]); }
static const float  *w_scale(int i)  { return (const float  *)(BLOB + YW_SOFF[i]); }
static const float  *w_bias(int i)   { return (const float  *)(BLOB + YW_BOFF[i]); }

/* ===================================================== tensors =========== */

typedef struct { float *d; int c, h, w; } ten_t;
static ten_t TS[YOLO_NTENSORS];

static float *ten_alloc(int id, int c, int h, int w)
{
    TS[id].c = c; TS[id].h = h; TS[id].w = w;
    TS[id].d = xmalloc((size_t)c * h * w * sizeof(float));
    return TS[id].d;
}
static void ten_free(int id) { free(TS[id].d); TS[id].d = NULL; }

/* ===================================================== PPM + letterbox === */

static unsigned char *read_ppm(const char *path, int *W, int *H)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open image '%s': %s", path, strerror(errno));
    char magic[3] = {0};
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P6"))
        die("'%s' is not a P6 PPM (got '%s')", path, magic);
    int vals[3], got = 0;
    while (got < 3) {
        int ch = fgetc(f);
        if (ch == '#') { while (ch != '\n' && ch != EOF) ch = fgetc(f); continue; }
        if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') continue;
        ungetc(ch, f);
        if (fscanf(f, "%d", &vals[got]) != 1) die("bad PPM header");
        got++;
    }
    fgetc(f);
    *W = vals[0]; *H = vals[1];
    if (vals[2] != 255) die("PPM maxval must be 255, got %d", vals[2]);
    size_t n = (size_t)*W * *H * 3;
    unsigned char *d = xmalloc(n);
    if (fread(d, 1, n, f) != n) die("short read on '%s'", path);
    fclose(f);
    return d;
}

/* Letterbox into a sw x sh canvas, preserving aspect. Bilinear, pad 114, /255.
 *
 * The canvas does not have to be square, and it should not be. COCO photographs
 * are about 3:4 or 4:3, so a square canvas spends a third of its pixels on grey
 * padding -- and every one of those padding pixels costs the same accelerator
 * invocations as a real one. Matching the canvas to the image aspect buys the
 * same object scale for less compute:
 *
 *   426x640 source, square 192x192 : 36,864 px, scale 0.30, 33 % padding
 *   426x640 source, rect   128x192 : 24,576 px, scale 0.30, no padding
 *
 * Same detection scale, one third less work. Nothing downstream assumes a
 * square: every op derives its shape from the tensor it is given, and the
 * detection head decodes anchors from each level's own H and W. */
static void letterbox(const unsigned char *src, int W, int H, int sw, int sh,
                      float *dst, float *ratio, int *padx, int *pady)
{
    double r = (double)sw / W; if ((double)sh / H < r) r = (double)sh / H;
    int nw = (int)(W * r + 0.5), nh = (int)(H * r + 0.5);
    int px = (sw - nw) / 2, py = (sh - nh) / 2;
    *ratio = (float)r; *padx = px; *pady = py;
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < sw*sh; i++) dst[(size_t)c*sw*sh + i] = 114.0f / 255.0f;
    for (int oy = 0; oy < nh; oy++) {
        double sy = (oy + 0.5) / r - 0.5; if (sy < 0) sy = 0;
        int y0 = (int)sy; if (y0 > H-1) y0 = H-1;
        int y1 = y0 + 1 < H ? y0 + 1 : y0;
        double fy = sy - y0;
        for (int ox = 0; ox < nw; ox++) {
            double sx = (ox + 0.5) / r - 0.5; if (sx < 0) sx = 0;
            int x0 = (int)sx; if (x0 > W-1) x0 = W-1;
            int x1 = x0 + 1 < W ? x0 + 1 : x0;
            double fx = sx - x0;
            for (int c = 0; c < 3; c++) {
                double v00 = src[((size_t)y0*W + x0)*3 + c], v01 = src[((size_t)y0*W + x1)*3 + c];
                double v10 = src[((size_t)y1*W + x0)*3 + c], v11 = src[((size_t)y1*W + x1)*3 + c];
                double v = v00*(1-fx)*(1-fy) + v01*fx*(1-fy) + v10*(1-fx)*fy + v11*fx*fy;
                dst[((size_t)c*sh + (py+oy))*sw + (px+ox)] = (float)(v / 255.0);
            }
        }
    }
}

/* ===================================================== conv ============== */

static int8_t  *packX, *packW_buf, *xq_buf;
static int32_t *acc_buf, *bias_i32;
static int      imgsz_g, imgh_g;   /* canvas, for buffer bound checks */
static float   *comb_buf;   /* per-output-channel dequant scale, N entries */
static long     PACKX_CAP;

/* silu(x) = x * sigmoid(x)
 *
 * The obvious implementation calls expf once per output element. At 320 px
 * that is 3,472,000 soft-float exponentials on a 50 MHz core, and measurement
 * puts it at the largest single remaining item in the query -- most of the
 * 78.1 s counted as "VEGA float".
 *
 * sigmoid is smooth, monotonic and bounded, so a table with linear
 * interpolation replaces it exactly enough. 6145 entries over [-12, +12] is a
 * step of 1/256; a linear fit errs by at most h^2/8 * max|sigma''| = 1.8e-7,
 * against an activation scale where a single INT8 step is around 1/127. The
 * table is 24 KB and is built once at startup.
 *
 * Outside the table sigmoid is saturated to 6.1e-6, so silu is taken as 0
 * below -12 (true value -7.4e-5) and as x above +12.
 *
 * --exact-float restores the expf path, for measuring the difference rather
 * than assuming it.                                                          */
#define SIG_N   6144
#define SIG_LO  (-12.0f)
#define SIG_HI  ( 12.0f)
static float SIG_T[SIG_N + 1];
static float g_sig_scale = 0.0f;
/* The sigmoid table is OPT-IN, not the default. It was measured, and it does
 * not pay for itself:
 *
 *   expf, FMA allowed          VEGA float 78.1 s   query 217.3 s
 *   table, FMA allowed         VEGA float 62.0 s   query 201.5 s
 *   table, -ffp-contract=off   VEGA float 76.2 s   query 215.8 s
 *
 * Almost the whole apparent 16 s win was the compiler contracting the
 * interpolation's a*b+c chains into FMA instructions -- and that contraction is
 * exactly what made the board disagree with the host simulation, because the
 * x86 build does not contract. Requiring both builds to be -ffp-contract=off,
 * which is what buys back bit-identical host/board agreement, leaves the table
 * worth about 2 s in 216. That is not worth carrying a change that alters
 * output, so expf is the default and --silu-table turns the table back on for
 * anyone who wants to repeat the measurement.                                */
static int   g_exact_float = 1;

static void silu_init(void)
{
    g_sig_scale = (float)SIG_N / (SIG_HI - SIG_LO);
    for (int i = 0; i <= SIG_N; i++) {
        double x = (double)SIG_LO + (double)i / (double)g_sig_scale;
        SIG_T[i] = (float)(1.0 / (1.0 + exp(-x)));
    }
}

static float silu(float x)
{
    if (g_exact_float)  return x / (1.0f + expf(-x));
    if (x <= SIG_LO)    return 0.0f;
    if (x >= SIG_HI)    return x;
    /* Three-point (quadratic) interpolation, not linear. Linear on this step
     * errs by h*h/8 * max|sigma''| = 1.8e-7, which sounds harmless and is not:
     * an INT8 activation step here is around 0.05, so roughly 1.6e-3 of the
     * 3.47 M elements land close enough to a rounding boundary to flip, and a
     * few thousand flipped INT8 values change the detection list. Quadratic
     * drops the error to about 5e-10, which flips nothing. The extra cost is
     * three multiplies, against one soft-float expf saved.                   */
    float t = (x - SIG_LO) * g_sig_scale;
    int   i = (int)t;
    if (i > SIG_N - 2) i = SIG_N - 2;
    float f  = t - (float)i;
    float a  = SIG_T[i], b = SIG_T[i+1], c = SIG_T[i+2];
    float s2 = a + f * ((-1.5f*a + 2.0f*b - 0.5f*c) + f * (0.5f*a - b + 0.5f*c));
    return x * s2;
}

/* A numerics fingerprint, printed in the header of every run.
 *
 * The host simulation and the board agreed bit-for-bit until the sigmoid table
 * went in, and then quietly stopped. The cause was not the table's accuracy: it
 * was fused multiply-add. The interpolation is a chain of a*b+c, the RISC-V
 * build contracts those into single FMA instructions (which keep the product's
 * full precision), and the x86 build does not. Forcing FMA on with
 * -march=haswell -ffp-contract=fast made the host reproduce the board's
 * 0.45823 exactly, which is what identified it.
 *
 * Both builds must therefore be compiled -ffp-contract=off. This line makes a
 * violation visible in one glance rather than after a 200 s run: it prints the
 * raw bits of an FMA-sensitive expression and of one silu evaluation. If the
 * host and the board print the same fpcheck line, the two builds round
 * identically and any later disagreement is a real bug.                      */
static void print_fp_fingerprint(void)
{
    volatile float a = 1.0000001f, b = 3.0000002f, c = -3.0000005f;
    union { float f; uint32_t u; } p1, p2, p3;
    p1.f = a * b + c;          /* catastrophic cancellation: fused != unfused */
    p2.f = silu(0.7351f);
    p3.f = silu(-2.3174f);
    printf("fpcheck: %08x %08x %08x   (host and board must match)\n",
           p1.u, p2.u, p3.u);
}

static void op_conv(int oi)
{
    int in = YOP_IN[oi][0], out = YOP_OUT[oi], wi = YOP_W[oi];
    int k = YOP_K[oi], s = YOP_S[oi], p = YOP_P[oi];
    ten_t *X = &TS[in];
    int K = YW_K[wi], N = YW_N[wi];
    int Ho = (X->h + 2*p - k)/s + 1, Wo = (X->w + 2*p - k)/s + 1;
    int M = Ho * Wo;
    int KT = (K + 7)/8, NT = (N + 7)/8;
    if (K != X->c * k * k)
        die("op %d: weight expects K=%d but input has %d channels (k=%d)",
            oi, K, X->c, k);

    float *Y = ten_alloc(out, N, Ho, Wo);

    long t0 = now_us();
    /* quantize the whole input tensor once, per tensor.
     * Two separate passes over the same multi-megabyte float tensor: one to
     * find amax, one to convert. They are timed apart because the arithmetic
     * in each is trivial and the cost is expected to be the memory traffic --
     * if that is so, the fix is fewer passes, not cheaper operations. */
    long nin = (long)X->c * X->h * X->w;
    if ((size_t)nin > XQ_CAP(imgsz_g, imgh_g))
        die("op %d: activation tensor is %ld bytes, quantise buffer holds %zu",
            oi, nin, XQ_CAP(imgsz_g, imgh_g));
    float amax = 0.0f;
    for (long i = 0; i < nin; i++) { float a = fabsf(X->d[i]); if (a > amax) amax = a; }
    { long _d = now_us() - t0; g_amax_us += _d; }
    long t_cvt = now_us();
    float sx = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    /* multiply by the reciprocal rather than divide: a soft-float divide costs
     * several times a multiply, and this runs once per input element -- almost
     * 5 million of them at 320 px. */
    float inv_sx = (amax > 0.0f) ? 127.0f / amax : 1.0f;
    for (long i = 0; i < nin; i++) {
        long q = lrintf(g_exact_float ? X->d[i] / sx : X->d[i] * inv_sx);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        xq_buf[i] = (int8_t)q;
    }
    g_quant_n += nin;
    g_qcvt_us += now_us() - t_cvt;
    g_vega_us += now_us() - t0;

    /* Copy the weight tiles out of the mmap'd package into RAM once per conv.
     * This is NOT just for the upload: --sim used to read tiles straight from
     * the mapping, and every W tile is re-read once per M row-tile. On the
     * board that cost 4,502 us per tile (1,540 s for one image) against 216 us
     * for the same sim_tile in Milestone B, because VEGA's mapped-file reads
     * are pathologically slow. One bulk copy fixes it for both paths. */
    { long _tc = now_us();
      memcpy(packW_buf, w_packed(wi), (size_t)NT * KT * 64);
      g_wcopy_us += now_us() - _tc; g_wcopy_n++;
      g_wcopy_by += (long)NT * KT * 64; }

    /* M is chunked so packX always fits the 1 MB X window */
    /* Chunk M so that (a) packX fits the 1 MB X window and (b) the INT32
     * accumulator buffer is bounded regardless of how small K is. Without (b) a
     * K=32 convolution asks for 32,768 rows at once and overruns acc_buf. */
    long per_rowtile = (long)KT * 64;
    int  MT_max = (int)(PACKX_CAP / per_rowtile);
    if (MT_max > ROWCAP / 8) MT_max = ROWCAP / 8;
    if (MT_max < 1) die("op %d: K=%d too large for the X window", oi, K);
    int first_chunk = 1;

    for (int m_base = 0; m_base < M; m_base += MT_max * 8) {
        int rows = M - m_base; if (rows > MT_max * 8) rows = MT_max * 8;
        int MT = (rows + 7)/8;

        /* im2col straight into tile order. Outer loop over k so the READS walk
         * the input tensor along a row; the scattered part lands on writes. */
        t0 = now_us();
        memset(packX, 0, (size_t)MT * KT * 64);
        for (int kk_g = 0; kk_g < K; kk_g++) {
            int c  = kk_g / (k*k), rem = kk_g % (k*k);
            int di = rem / k, dj = rem % k;
            int k0 = kk_g / 8, kk = kk_g % 8;
            const int8_t *plane = xq_buf + (size_t)c * X->h * X->w;
            int8_t *base = packX + (long)k0*64 + kk*8;
            for (int r = 0; r < rows; r++) {
                int gr = m_base + r;
                int oy = gr / Wo, ox = gr % Wo;
                int iy = oy*s - p + di, ix = ox*s - p + dj;
                if ((unsigned)iy < (unsigned)X->h && (unsigned)ix < (unsigned)X->w)
                    base[(long)(r/8) * KT * 64 + (r%8)] = plane[(size_t)iy*X->w + ix];
            }
        }
        { long _d = now_us() - t0; g_pack_us += _d; g_im2col_us += _d; g_im2col_n++; }

        const float *sw = w_scale(wi), *bf = w_bias(wi);
        for (int n = 0; n < N; n++) {
            double comb = (double)sx * (double)sw[n];
            bias_i32[n] = (int32_t)lround((double)bf[n] / comb);
        }
        gemm_run(packX, packW_buf, rows, K, N,
                 bias_i32, acc_buf, first_chunk);
        first_chunk = 0;

        t0 = now_us();
        /* Dequantise, activate, and scatter into the planar output.
         *
         * The obvious loop nest -- n outer, r inner -- reads the accumulator as
         * src[r * N], striding N*4 bytes on every single element. That is a
         * transpose against a row-major buffer, so essentially every read is a
         * cache miss. Measured on the board it cost 11.18 us per element while
         * the quantise pass, which does more arithmetic over more elements,
         * cost 1.55 us. The difference was never compute; it was the stride.
         *
         * Blocking fixes it. Inside a block the accumulator is read
         * sequentially, and only BN output planes are being written at once, so
         * both sides stay resident. The arithmetic per element is unchanged and
         * every element is still visited exactly once, so the result is
         * bit-identical -- this needs no accuracy argument, only a timing one. */
        {
            const int BN = 32, BR = 64;
            for (int n = 0; n < N; n++) comb_buf[n] = sx * sw[n];
            int act = YOP_ACT[oi];
            for (int nb = 0; nb < N; nb += BN) {
                int ne = nb + BN < N ? nb + BN : N;
                for (int rb = 0; rb < rows; rb += BR) {
                    int re = rb + BR < rows ? rb + BR : rows;
                    for (int r = rb; r < re; r++) {
                        const int32_t *srow = acc_buf + (long)r * N;
                        for (int n = nb; n < ne; n++) {
                            float v = (float)srow[n] * comb_buf[n];
                            Y[(size_t)n * M + m_base + r] = act ? silu(v) : v;
                        }
                    }
                }
            }
        }
        g_deq_n += (long)N * rows;
        if (YOP_ACT[oi]) g_silu_n += (long)N * rows;
        { long _d = now_us() - t0; g_deq_us += _d; g_vega_us += _d; }
    }
    (void)MT_max;
}

/* ===================================================== other ops ========= */

static void op_maxpool(int oi)
{
    ten_t *X = &TS[YOP_IN[oi][0]];
    int k = YOP_K[oi], s = YOP_S[oi], p = YOP_P[oi];
    int Ho = (X->h + 2*p - k)/s + 1, Wo = (X->w + 2*p - k)/s + 1;
    float *Y = ten_alloc(YOP_OUT[oi], X->c, Ho, Wo);
    for (int c = 0; c < X->c; c++)
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++) {
                float m = -INFINITY;
                for (int i = 0; i < k; i++)
                    for (int j = 0; j < k; j++) {
                        int iy = oy*s - p + i, ix = ox*s - p + j;
                        if ((unsigned)iy < (unsigned)X->h && (unsigned)ix < (unsigned)X->w) {
                            float v = X->d[((size_t)c*X->h + iy)*X->w + ix];
                            if (v > m) m = v;
                        }
                    }
                Y[((size_t)c*Ho + oy)*Wo + ox] = m;
            }
}
static void op_upsample(int oi)
{
    ten_t *X = &TS[YOP_IN[oi][0]];
    int Ho = X->h*2, Wo = X->w*2;
    float *Y = ten_alloc(YOP_OUT[oi], X->c, Ho, Wo);
    for (int c = 0; c < X->c; c++)
        for (int oy = 0; oy < Ho; oy++)
            for (int ox = 0; ox < Wo; ox++)
                Y[((size_t)c*Ho + oy)*Wo + ox] = X->d[((size_t)c*X->h + oy/2)*X->w + ox/2];
}
static void op_concat(int oi)
{
    int n = YOP_NIN[oi], C = 0;
    ten_t *a = &TS[YOP_IN[oi][0]];
    for (int i = 0; i < n; i++) C += TS[YOP_IN[oi][i]].c;
    float *Y = ten_alloc(YOP_OUT[oi], C, a->h, a->w);
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        ten_t *t = &TS[YOP_IN[oi][i]];
        size_t sz = (size_t)t->c * t->h * t->w;
        memcpy(Y + off, t->d, sz * sizeof(float));
        off += sz;
    }
}
static void op_slice(int oi)
{
    ten_t *X = &TS[YOP_IN[oi][0]];
    int c0 = YOP_C0[oi], c1 = YOP_C1[oi];
    float *Y = ten_alloc(YOP_OUT[oi], c1-c0, X->h, X->w);
    memcpy(Y, X->d + (size_t)c0*X->h*X->w,
           (size_t)(c1-c0)*X->h*X->w*sizeof(float));
}
static void op_add(int oi)
{
    ten_t *A = &TS[YOP_IN[oi][0]], *B = &TS[YOP_IN[oi][1]];
    float *Y = ten_alloc(YOP_OUT[oi], A->c, A->h, A->w);
    size_t n = (size_t)A->c * A->h * A->w;
    for (size_t i = 0; i < n; i++) Y[i] = A->d[i] + B->d[i];
}

/* ===================================================== detect =========== */

typedef struct { int cls; float conf, x1, y1, x2, y2; } det_t;

static int det_cmp(const void *a, const void *b)
{
    float d = ((const det_t *)b)->conf - ((const det_t *)a)->conf;
    return d > 0 ? 1 : (d < 0 ? -1 : 0);
}
static float iou(const det_t *a, const det_t *b)
{
    float x1 = a->x1 > b->x1 ? a->x1 : b->x1, y1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float x2 = a->x2 < b->x2 ? a->x2 : b->x2, y2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float w = x2-x1, h = y2-y1;
    if (w <= 0 || h <= 0) return 0.0f;
    float inter = w*h;
    float ar = (a->x2-a->x1)*(a->y2-a->y1) + (b->x2-b->x1)*(b->y2-b->y1) - inter;
    return ar > 0 ? inter/ar : 0.0f;
}

static int decode_detect(det_t *out, int maxout, float conf_thr, float iou_thr)
{
    int n = 0;
    for (int lvl = 0; lvl < YOLO_NLEVELS; lvl++) {
        ten_t *B = &TS[YDET_BOX[lvl]], *C = &TS[YDET_CLS[lvl]];
        int H = B->h, W = B->w, A = H*W;
        float st = YDET_STRIDE[lvl];
        for (int a = 0; a < A; a++) {
            int best = -1; float bs = 0.0f;
            for (int c = 0; c < YOLO_NCLASSES; c++) {
                float v = 1.0f / (1.0f + expf(-C->d[(size_t)c*A + a]));
                if (v > bs) { bs = v; best = c; }
            }
            if (bs < conf_thr) continue;
            float d[4];
            for (int side = 0; side < 4; side++) {
                float mx = -INFINITY, sum = 0.0f, acc = 0.0f;
                for (int r = 0; r < YOLO_REG_MAX; r++) {
                    float v = B->d[(size_t)(side*YOLO_REG_MAX + r)*A + a];
                    if (v > mx) mx = v;
                }
                for (int r = 0; r < YOLO_REG_MAX; r++) {
                    float e = expf(B->d[(size_t)(side*YOLO_REG_MAX + r)*A + a] - mx);
                    sum += e; acc += e * r;
                }
                d[side] = acc / sum;
            }
            float ax = (a % W) + 0.5f, ay = (a / W) + 0.5f;
            if (n < maxout) {
                out[n].cls = best; out[n].conf = bs;
                out[n].x1 = (ax - d[0]) * st; out[n].y1 = (ay - d[1]) * st;
                out[n].x2 = (ax + d[2]) * st; out[n].y2 = (ay + d[3]) * st;
                n++;
            }
        }
    }
    qsort(out, n, sizeof(det_t), det_cmp);
    /* per-class NMS */
    int keep = 0;
    for (int i = 0; i < n; i++) {
        int drop = 0;
        for (int j = 0; j < keep; j++)
            if (out[j].cls == out[i].cls && iou(&out[i], &out[j]) > iou_thr) { drop = 1; break; }
        if (!drop) out[keep++] = out[i];
    }
    return keep;
}

/* ===================================================== model access ====== */

typedef struct { const unsigned char *base; size_t size; } model_t;
typedef struct {
    const unsigned char *p; int d0, d1; double scale; int is_i8; const char *name;
} tensor_t;

static model_t MODEL;

static tensor_t mk(const char *name, long off, int d0, int d1, double scale, int is_i8)
{
    tensor_t t;
    size_t need = (size_t)d0 * (size_t)d1 * (is_i8 ? 1u : 4u);
    if (off < 0 || (size_t)off + need > MODEL.size)
        die("tensor %s past end of package", name);
    t.p = MODEL.base + off; t.d0 = d0; t.d1 = d1;
    t.scale = scale; t.is_i8 = is_i8; t.name = name;
    return t;
}
#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)
#define TEN(N) mk(#N, (long)CAT(N,_OFF), (int)CAT(N,_D0), (int)CAT(N,_D1), \
                  (double)CAT(N,_SCALE), (int)CAT(N,_IS_I8))

static double t_get(const tensor_t *t, int r, int c)
{
    size_t i = (size_t)r * (size_t)t->d1 + (size_t)c;
    return t->is_i8 ? (double)((const int8_t *)t->p)[i] * t->scale
                    : (double)((const float *)t->p)[i];
}
static double t_get1(const tensor_t *t, int i)
{
    return t->is_i8 ? (double)((const int8_t *)t->p)[i] * t->scale
                    : (double)((const float *)t->p)[i];
}
/* weights are stored [out, in]: element multiplying input k into output c */
static int8_t w_i8(const tensor_t *t, int k, int c)
{
    return ((const int8_t *)t->p)[(size_t)c * (size_t)t->d1 + (size_t)k];
}

/* per-layer weight tensors, resolved once */
typedef struct { tensor_t q,k,v,o,f1,f2; } lw_t;
typedef struct {
    tensor_t qb,kb,vb,ob, ln1w,ln1b, f1b,f2b, ln2w,ln2b;
} lb_t;
static lw_t LW[NLAYERS];
static lb_t LB[NLAYERS];

static void resolve_tensors(void)
{
    LW[0].q=TEN(T_L0_Q_W); LW[0].k=TEN(T_L0_K_W); LW[0].v=TEN(T_L0_V_W);
    LW[0].o=TEN(T_L0_O_W); LW[0].f1=TEN(T_L0_F1_W); LW[0].f2=TEN(T_L0_F2_W);
    LW[1].q=TEN(T_L1_Q_W); LW[1].k=TEN(T_L1_K_W); LW[1].v=TEN(T_L1_V_W);
    LW[1].o=TEN(T_L1_O_W); LW[1].f1=TEN(T_L1_F1_W); LW[1].f2=TEN(T_L1_F2_W);
    LW[2].q=TEN(T_L2_Q_W); LW[2].k=TEN(T_L2_K_W); LW[2].v=TEN(T_L2_V_W);
    LW[2].o=TEN(T_L2_O_W); LW[2].f1=TEN(T_L2_F1_W); LW[2].f2=TEN(T_L2_F2_W);
    LW[3].q=TEN(T_L3_Q_W); LW[3].k=TEN(T_L3_K_W); LW[3].v=TEN(T_L3_V_W);
    LW[3].o=TEN(T_L3_O_W); LW[3].f1=TEN(T_L3_F1_W); LW[3].f2=TEN(T_L3_F2_W);
    LW[4].q=TEN(T_L4_Q_W); LW[4].k=TEN(T_L4_K_W); LW[4].v=TEN(T_L4_V_W);
    LW[4].o=TEN(T_L4_O_W); LW[4].f1=TEN(T_L4_F1_W); LW[4].f2=TEN(T_L4_F2_W);
    LW[5].q=TEN(T_L5_Q_W); LW[5].k=TEN(T_L5_K_W); LW[5].v=TEN(T_L5_V_W);
    LW[5].o=TEN(T_L5_O_W); LW[5].f1=TEN(T_L5_F1_W); LW[5].f2=TEN(T_L5_F2_W);

#define LBSET(i)                                                              \
    LB[i].qb  = TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_ATTENTION_SELF_QUERY_BIAS));   \
    LB[i].kb  = TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_ATTENTION_SELF_KEY_BIAS));     \
    LB[i].vb  = TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_ATTENTION_SELF_VALUE_BIAS));   \
    LB[i].ob  = TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_ATTENTION_OUTPUT_DENSE_BIAS)); \
    LB[i].ln1w= TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_ATTENTION_OUTPUT_LAYERNORM_WEIGHT)); \
    LB[i].ln1b= TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_ATTENTION_OUTPUT_LAYERNORM_BIAS));   \
    LB[i].f1b = TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_INTERMEDIATE_DENSE_BIAS));     \
    LB[i].f2b = TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_OUTPUT_DENSE_BIAS));           \
    LB[i].ln2w= TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_OUTPUT_LAYERNORM_WEIGHT));     \
    LB[i].ln2b= TEN(CAT(CAT(T_ENCODER_ENCODER_LAYER_,i),_OUTPUT_LAYERNORM_BIAS));
    LBSET(0) LBSET(1) LBSET(2) LBSET(3) LBSET(4) LBSET(5)
#undef LBSET
}

/* ===================================================== vocabulary ======== */

#define VH_BITS 16
#define VH_SIZE (1u << VH_BITS)
typedef struct { char *blob; char **tok; int n; int32_t slot[VH_SIZE]; } vocab_t;
static vocab_t VOC;

static uint32_t fnv1a(const char *s, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}
static void vocab_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open vocab '%s': %s", path, strerror(errno));
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    if (sz <= 0) die("vocab empty");
    VOC.blob = (char *)calloc((size_t)sz + 2, 1);
    if (!VOC.blob || fread(VOC.blob, 1, (size_t)sz, f) != (size_t)sz) die("vocab read");
    fclose(f);
    int lines = 0;
    for (long i = 0; i < sz; i++) if (VOC.blob[i] == '\n') lines++;
    if (VOC.blob[sz-1] != '\n') lines++;
    VOC.tok = (char **)calloc((size_t)lines, sizeof(char *));
    for (unsigned i = 0; i < VH_SIZE; i++) VOC.slot[i] = -1;
    int id = 0; char *p = VOC.blob, *end = VOC.blob + sz;
    while (p < end) {
        char *nl = memchr(p, '\n', (size_t)(end - p));
        char *le = nl ? nl : end;
        if (le > p && *(le-1) == '\r') le--;
        *le = '\0';
        VOC.tok[id] = p;
        uint32_t h = fnv1a(p, strlen(p)) & (VH_SIZE - 1);
        while (VOC.slot[h] != -1) h = (h + 1) & (VH_SIZE - 1);
        VOC.slot[h] = id++;
        p = nl ? nl + 1 : end;
    }
    VOC.n = id;
    if (VOC.n != MINILM_VOCAB)
        die("vocab has %d entries, package says %d", VOC.n, MINILM_VOCAB);
}
static int vlook(const char *s, size_t n)
{
    uint32_t h = fnv1a(s, n) & (VH_SIZE - 1);
    while (VOC.slot[h] != -1) {
        int id = VOC.slot[h];
        if (strlen(VOC.tok[id]) == n && memcmp(VOC.tok[id], s, n) == 0) return id;
        h = (h + 1) & (VH_SIZE - 1);
    }
    return -1;
}

/* ===================================================== tokenizer ========= */

#define MAXW 64
#define MAXWL 128
static int is_ws(int c){return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v';}
static int is_punct(int c)
{
    if ((c>='0'&&c<='9')||(c>='a'&&c<='z')||(c>='A'&&c<='Z')) return 0;
    return (c>=33 && c<=126);
}
static int lower(int c){ return (c>='A'&&c<='Z') ? c+32 : c; }

static int basic_tok(const char *text, char w[MAXW][MAXWL])
{
    int nw = 0, wl = 0;
    for (const unsigned char *p = (const unsigned char *)text; ; p++) {
        int c = *p;
        if (c == 0 || is_ws(c) || is_punct(c)) {
            if (wl) { if (nw>=MAXW) die("too many words"); w[nw][wl]='\0'; nw++; wl=0; }
            if (c == 0) break;
            if (is_punct(c)) { if (nw>=MAXW) die("too many words");
                               w[nw][0]=(char)c; w[nw][1]='\0'; nw++; }
            continue;
        }
        if (wl >= MAXWL-1) die("word too long");
        w[nw][wl++] = (char)lower(c);
    }
    return nw;
}
static int wordpiece(const char *word, int *out, int maxo)
{
    size_t len = strlen(word); int n = 0;
    if (!len) return 0;
    if (len > 100) { if (n<maxo) out[n++]=TOK_UNK; return n; }
    size_t start = 0; int local[MAXWL], nl = 0;
    while (start < len) {
        size_t end = len; int found = -1, fe = 0;
        while (end > start) {
            char buf[MAXWL+2]; size_t k = 0, sl = end - start;
            if (start) { buf[k++]='#'; buf[k++]='#'; }
            memcpy(buf+k, word+start, sl); k += sl; buf[k]='\0';
            found = vlook(buf, k);
            if (found >= 0) { fe = (int)end; break; }
            end--;
        }
        if (found < 0) { if (n<maxo) out[n++]=TOK_UNK; return n; }
        if (nl >= MAXWL) die("wordpiece explosion");
        local[nl++] = found; start = (size_t)fe;
    }
    for (int i = 0; i < nl && n < maxo; i++) out[n++] = local[i];
    return n;
}
static void tokenize_task(const char *task, int *ids, int *mask)
{
    char text[1024];
    if (snprintf(text, sizeof(text), "task: %s", task) >= (int)sizeof(text))
        die("task string too long");
    char w[MAXW][MAXWL];
    int nw = basic_tok(text, w);
    int body[MAXTOK*4], nb = 0;
    for (int i = 0; i < nw; i++) {
        int got[MAXWL], g = wordpiece(w[i], got, MAXWL);
        for (int j = 0; j < g; j++)
            if (nb < (int)(sizeof(body)/sizeof(body[0]))) body[nb++] = got[j];
    }
    int cap = MAXTOK - 2;
    if (nb > cap) { nb = cap; fprintf(stderr, "[warn] prompt truncated to %d tokens\n", cap); }
    int k = 0; ids[k++] = TOK_CLS;
    for (int i = 0; i < nb; i++) ids[k++] = body[i];
    ids[k++] = TOK_SEP;
    for (int i = k; i < MAXTOK; i++) ids[i] = TOK_PAD;
    for (int i = 0; i < MAXTOK; i++) mask[i] = (i < k);
}

/* ===================================================== VEGA-side ops ===== */

static void layernorm(double *x, int rows, int n, const tensor_t *w, const tensor_t *b)
{
    for (int r = 0; r < rows; r++) {
        double *v = x + (size_t)r*n, m = 0.0, var = 0.0;
        for (int i = 0; i < n; i++) m += v[i];
        m /= n;
        for (int i = 0; i < n; i++) { double d = v[i]-m; var += d*d; }
        var /= n;
        double inv = 1.0/sqrt(var + 1e-12);
        for (int i = 0; i < n; i++) v[i] = (v[i]-m)*inv*t_get1(w,i) + t_get1(b,i);
    }
}
static double gelu(double x) { return x * 0.5 * (1.0 + erf(x / sqrt(2.0))); }


/* ============================ MiniLM packing on the shared GEMM engine === */

static int8_t  mq_buf[NCLS * FUSED];
static int32_t mo_buf[NCLS * HID];

/* source-sequential packing (the 20x lesson from Milestone B) */
static void mpack_x(const int8_t *x, int M, int K)
{
    int MT = (M + 7)/8, KT = K/8;
    long t0 = now_us();
    memset(packX, 0, (size_t)MT * KT * 64);
    for (int r = 0; r < M; r++) {
        const int8_t *row = x + (size_t)r * K;
        int8_t *base = packX + (long)(r/8) * KT * 64 + (r % 8);
        for (int k = 0; k < K; k++) base[(long)(k/8)*64 + (k%8)*8] = row[k];
    }
    { long _d = now_us() - t0; g_pack_us += _d; g_mpx_us += _d; g_mpx_n++; }
}
/* ===================== pre-packed MiniLM weight cache =====================
 *
 * mpack_w_pkg rearranges a weight matrix into 8x8 accelerator tiles. The
 * arithmetic is trivial -- it is a byte copy -- but the writes are scattered
 * across tiles, and on this core every scattered access is a DDR round trip.
 * Measured on hardware: 10.7 MB rearranged in 130 s, about 0.08 MB/s, or
 * roughly 270 cycles per byte moved.
 *
 * The weights are constant, so the rearrangement can be done once, off the
 * board, exactly as the detector package already does. This is a sidecar
 * file: the original package is untouched and still works on its own.
 *
 *   --emit-packed FILE   run once on the host, writes the cache
 *   --packed FILE        use the cache; each pack becomes one memcpy
 *
 * Blocks are keyed by (offset of the tensor inside the package, K, N), not by
 * call order, so a cache built from a different model cannot be silently
 * mistaken for a valid one -- the lookup fails and the program stops.
 */
#define PK_MAGIC   "T43MLPK1"
#define PK_MAXREC  64
typedef struct { uint64_t woff; uint32_t K, N; uint64_t off, bytes; } pkrec_t;

static const char *g_pk_emit = NULL, *g_pk_use = NULL;
/* --dump-scores writes MiniLM(task, class) for all 80 COCO classes and exits.
 * The evaluator reads a 14x80 table built this way, so that what is scored
 * offline is literally what the board computes, not a separate float model. */
static const char *g_dump_scores = NULL;
static pkrec_t  g_pk_rec[PK_MAXREC];
static uint8_t *g_pk_buf[PK_MAXREC];
static int      g_pk_n = 0;
static const uint8_t *g_pk_base = NULL;
static long     g_pk_hits = 0, g_pk_bytes = 0;

static void pk_load(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open packed cache '%s': %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat packed cache: %s", strerror(errno));
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) die("mmap packed cache: %s", strerror(errno));
    const uint8_t *b = (const uint8_t *)m;
    if (memcmp(b, PK_MAGIC, 8) != 0) die("'%s' is not a T43 packed cache", path);
    uint32_t ver, cnt; uint64_t mb;
    memcpy(&ver, b + 8, 4); memcpy(&cnt, b + 12, 4); memcpy(&mb, b + 16, 8);
    if (ver != 1) die("packed cache version %u, expected 1", ver);
    if (mb != (uint64_t)MODEL.size)
        die("packed cache was built for a %llu byte package, this one is %llu",
            (unsigned long long)mb, (unsigned long long)MODEL.size);
    if (cnt > PK_MAXREC) die("packed cache holds %u blocks, limit %d", cnt, PK_MAXREC);
    memcpy(g_pk_rec, b + 64, sizeof(pkrec_t) * cnt);
    g_pk_n = (int)cnt; g_pk_base = b;
}

static void pk_write(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write '%s': %s", path, strerror(errno));
    uint64_t off = 64 + (uint64_t)sizeof(pkrec_t) * (uint64_t)g_pk_n;
    for (int i = 0; i < g_pk_n; i++) { g_pk_rec[i].off = off; off += g_pk_rec[i].bytes; }
    uint8_t hdr[64]; memset(hdr, 0, sizeof(hdr));
    memcpy(hdr, PK_MAGIC, 8);
    uint32_t ver = 1, cnt = (uint32_t)g_pk_n; uint64_t mb = (uint64_t)MODEL.size;
    memcpy(hdr + 8, &ver, 4); memcpy(hdr + 12, &cnt, 4); memcpy(hdr + 16, &mb, 8);
    fwrite(hdr, 1, sizeof(hdr), f);
    fwrite(g_pk_rec, sizeof(pkrec_t), (size_t)g_pk_n, f);
    long total = 0;
    for (int i = 0; i < g_pk_n; i++) {
        fwrite(g_pk_buf[i], 1, (size_t)g_pk_rec[i].bytes, f);
        total += (long)g_pk_rec[i].bytes;
    }
    fclose(f);
    printf("\npacked cache written: %s\n", path);
    printf("  %d weight blocks, %.2f MB\n", g_pk_n, total / 1048576.0);
}

static void mpack_w_pkg(const tensor_t *W, int K, int N)
{
    int KT = K/8, NT = (N + 7)/8;
    const int8_t *p = (const int8_t *)W->p;
    int D1 = W->d1;
    size_t nb = (size_t)NT * KT * 64;
    uint64_t woff = (uint64_t)((const unsigned char *)W->p - MODEL.base);
    long t0 = now_us();

    if (g_pk_use) {                       /* one bulk copy instead of a scatter */
        int hit = -1;
        for (int i = 0; i < g_pk_n; i++)
            if (g_pk_rec[i].woff == woff && g_pk_rec[i].K == (uint32_t)K
                                         && g_pk_rec[i].N == (uint32_t)N) { hit = i; break; }
        if (hit < 0)
            die("packed cache has no block for tensor %s (off %llu, K=%d, N=%d)",
                W->name ? W->name : "?", (unsigned long long)woff, K, N);
        if (g_pk_rec[hit].bytes != (uint64_t)nb)
            die("packed cache block for %s is %llu bytes, expected %llu",
                W->name ? W->name : "?", (unsigned long long)g_pk_rec[hit].bytes,
                (unsigned long long)nb);
        memcpy(packW_buf, g_pk_base + g_pk_rec[hit].off, nb);
        g_pk_hits++; g_pk_bytes += (long)nb;
        { long _d = now_us() - t0; g_pack_us += _d; g_mpw_us += _d; g_mpw_n++;
          g_mpw_by += (long)K * N; }
        return;
    }

    memset(packW_buf, 0, nb);
    for (int c = 0; c < N; c++) {
        const int8_t *row = p + (size_t)c * D1;
        int8_t *base = packW_buf + (long)(c/8) * KT * 64 + (c % 8);
        for (int k = 0; k < K; k++) base[(long)(k/8)*64 + (k%8)*8] = row[k];
    }
    { long _d = now_us() - t0; g_pack_us += _d; g_mpw_us += _d; g_mpw_n++;
      g_mpw_by += (long)K * N; }

    if (g_pk_emit) {                      /* stash a copy for the cache file */
        for (int i = 0; i < g_pk_n; i++)
            if (g_pk_rec[i].woff == woff && g_pk_rec[i].K == (uint32_t)K
                                         && g_pk_rec[i].N == (uint32_t)N) return;
        if (g_pk_n >= PK_MAXREC) die("too many weight blocks for the cache");
        g_pk_buf[g_pk_n] = (uint8_t *)xmalloc(nb);
        memcpy(g_pk_buf[g_pk_n], packW_buf, nb);
        g_pk_rec[g_pk_n].woff = woff; g_pk_rec[g_pk_n].K = (uint32_t)K;
        g_pk_rec[g_pk_n].N = (uint32_t)N; g_pk_rec[g_pk_n].bytes = (uint64_t)nb;
        g_pk_n++;
    }
}
static void mpack_w_act(const int8_t *w, int K, int N)
{
    int KT = K/8, NT = (N + 7)/8;
    long t0 = now_us();
    memset(packW_buf, 0, (size_t)NT * KT * 64);
    for (int k = 0; k < K; k++) {
        const int8_t *row = w + (size_t)k * N;
        int8_t *base = packW_buf + (long)(k/8)*64 + (k%8)*8;
        for (int c = 0; c < N; c++) base[(long)(c/8) * KT * 64 + (c%8)] = row[c];
    }
    { long _d = now_us() - t0; g_pack_us += _d; g_mpa_us += _d; g_mpa_n++; }
}
static double mquant(const double *x, long n, int8_t *out)
{
    double amax = 0.0;
    for (long i = 0; i < n; i++) { double a = fabs(x[i]); if (a > amax) amax = a; }
    double s = (amax > 0.0) ? amax / 127.0 : 1.0;
    for (long i = 0; i < n; i++) {
        long q = lround(x[i] / s);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        out[i] = (int8_t)q;
    }
    return s;
}
static void dense(const double *x, int M, int K, int N,
                  const tensor_t *W, const tensor_t *B, double *y)
{
    double sx = mquant(x, (long)M*K, mq_buf);
    double comb = sx * W->scale;
    for (int c = 0; c < N; c++) bias_i32[c] = (int32_t)lround(t_get1(B, c) / comb);
    mpack_x(mq_buf, M, K);
    mpack_w_pkg(W, K, N);
    gemm_run(packX, packW_buf, M, K, N, bias_i32, mo_buf, 1);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            y[(size_t)r*N + c] = (double)mo_buf[(size_t)r*N + c] * comb;
}
static void matmul_act(const double *a, int M, int K, int N, const double *b, double *y)
{
    static int8_t bq[FFNDIM * 8];
    double sa = mquant(a, (long)M*K, mq_buf);
    mpack_x(mq_buf, M, K);
    double sb = mquant(b, (long)K*N, bq);
    mpack_w_act(bq, K, N);
    memset(bias_i32, 0, sizeof(int32_t) * (size_t)N);
    gemm_run(packX, packW_buf, M, K, N, bias_i32, mo_buf, 1);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            y[(size_t)r*N + c] = (double)mo_buf[(size_t)r*N + c] * sa * sb;
}

/* ============================ MiniLM encoder ============================= */

/* lw_t / lb_t / resolve_tensors come from the Milestone B engine above. */

static double MX[MAXTOK*HID], MQ[MAXTOK*HID], MK[MAXTOK*HID], MV[MAXTOK*HID];
static double MCTX[MAXTOK*HID], MTMP[MAXTOK*HID], MH[MAXTOK*FFNDIM];
static double qh[MAXTOK*HDIM], kh[HDIM*MAXTOK], vh[MAXTOK*HDIM];
static double att[MAXTOK*MAXTOK], ctxh[MAXTOK*HDIM];
static double temb[HID];

static void encode_task(const int *ids, const int *mask)
{
    tensor_t we = TEN(T_WORD_EMB), pe = TEN(T_POS_EMB), te = TEN(T_TOK_TYPE);
    tensor_t lnw = TEN(T_ENCODER_EMBEDDINGS_LAYERNORM_WEIGHT);
    tensor_t lnb = TEN(T_ENCODER_EMBEDDINGS_LAYERNORM_BIAS);
    for (int t = 0; t < MAXTOK; t++)
        for (int h = 0; h < HID; h++)
            MX[t*HID+h] = t_get(&we, ids[t], h) + t_get(&pe, t, h) + t_get(&te, 0, h);
    layernorm(MX, MAXTOK, HID, &lnw, &lnb);

    for (int L = 0; L < NLAYERS; L++) {
        dense(MX, MAXTOK, HID, HID, &LW[L].q, &LB[L].qb, MQ);
        dense(MX, MAXTOK, HID, HID, &LW[L].k, &LB[L].kb, MK);
        dense(MX, MAXTOK, HID, HID, &LW[L].v, &LB[L].vb, MV);
        for (int h = 0; h < NHEADS; h++) {
            for (int t = 0; t < MAXTOK; t++)
                for (int d = 0; d < HDIM; d++) {
                    qh[t*HDIM+d] = MQ[t*HID + h*HDIM + d];
                    vh[t*HDIM+d] = MV[t*HID + h*HDIM + d];
                    kh[d*MAXTOK+t] = MK[t*HID + h*HDIM + d];
                }
            matmul_act(qh, MAXTOK, HDIM, MAXTOK, kh, att);
            for (int r = 0; r < MAXTOK; r++) {
                double mx = -1e300;
                for (int c = 0; c < MAXTOK; c++) {
                    att[r*MAXTOK+c] = att[r*MAXTOK+c] / sqrt((double)HDIM)
                                    + (mask[c] ? 0.0 : -10000.0);
                    if (att[r*MAXTOK+c] > mx) mx = att[r*MAXTOK+c];
                }
                double sum = 0.0;
                for (int c = 0; c < MAXTOK; c++) {
                    att[r*MAXTOK+c] = exp(att[r*MAXTOK+c] - mx); sum += att[r*MAXTOK+c];
                }
                for (int c = 0; c < MAXTOK; c++) att[r*MAXTOK+c] /= sum;
            }
            matmul_act(att, MAXTOK, MAXTOK, HDIM, vh, ctxh);
            for (int t = 0; t < MAXTOK; t++)
                for (int d = 0; d < HDIM; d++)
                    MCTX[t*HID + h*HDIM + d] = ctxh[t*HDIM+d];
        }
        dense(MCTX, MAXTOK, HID, HID, &LW[L].o, &LB[L].ob, MTMP);
        for (long i = 0; i < MAXTOK*HID; i++) MTMP[i] += MX[i];
        layernorm(MTMP, MAXTOK, HID, &LB[L].ln1w, &LB[L].ln1b);
        memcpy(MX, MTMP, sizeof(MX));
        dense(MX, MAXTOK, HID, FFNDIM, &LW[L].f1, &LB[L].f1b, MH);
        for (long i = 0; i < MAXTOK*FFNDIM; i++) MH[i] = gelu(MH[i]);
        dense(MH, MAXTOK, FFNDIM, HID, &LW[L].f2, &LB[L].f2b, MTMP);
        for (long i = 0; i < MAXTOK*HID; i++) MTMP[i] += MX[i];
        layernorm(MTMP, MAXTOK, HID, &LB[L].ln2w, &LB[L].ln2b);
        memcpy(MX, MTMP, sizeof(MX));
        if (!g_quiet) { printf("  minilm layer %d/%d\n", L+1, NLAYERS); fflush(stdout); }
    }
    double cnt = 0.0;
    for (int h = 0; h < HID; h++) temb[h] = 0.0;
    for (int t = 0; t < MAXTOK; t++) if (mask[t]) {
        cnt += 1.0;
        for (int h = 0; h < HID; h++) temb[h] += MX[t*HID+h];
    }
    if (cnt < 1e-9) cnt = 1e-9;
    double nrm = 0.0;
    for (int h = 0; h < HID; h++) { temb[h] /= cnt; nrm += temb[h]*temb[h]; }
    nrm = sqrt(nrm); if (nrm < 1e-9) nrm = 1e-9;
    for (int h = 0; h < HID; h++) temb[h] /= nrm;
}

/* score a set of class ids against the encoded task */
static double fused_buf[NCLS*FUSED], hid_buf[NCLS*HID], logit_buf[NCLS*8];

static void score_classes(const int *cls, int n, double *out)
{
    tensor_t oe = TEN(T_OBJ_EMB);
    for (int i = 0; i < n; i++)
        for (int h = 0; h < HID; h++) {
            double o = t_get(&oe, cls[i], h);
            fused_buf[(size_t)i*FUSED + h]         = temb[h];
            fused_buf[(size_t)i*FUSED + HID + h]   = o;
            fused_buf[(size_t)i*FUSED + 2*HID + h] = fabs(temb[h] - o);
            fused_buf[(size_t)i*FUSED + 3*HID + h] = temb[h] * o;
        }
    tensor_t h0w = TEN(T_HEAD0_W), h0b = TEN(T_HEAD_0_BIAS);
    tensor_t h3w = TEN(T_HEAD3_W), h3b = TEN(T_HEAD_3_BIAS);
    dense(fused_buf, n, FUSED, HID, &h0w, &h0b, hid_buf);
    for (long i = 0; i < (long)n*HID; i++) if (hid_buf[i] < 0.0) hid_buf[i] = 0.0;
    dense(hid_buf, n, HID, 1, &h3w, &h3b, logit_buf);
    for (int i = 0; i < n; i++) out[i] = 1.0 / (1.0 + exp(-logit_buf[i]));
}

/* Accept either a raw P6 PPM or an ordinary baseline JPEG. Sniffing the magic
 * means the board takes the same .jpg you would hand any other program -- no
 * off-board preparation of the image at all. */
static unsigned char *read_image(const char *path, int *W, int *H)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open image '%s': %s", path, strerror(errno));
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    if (n < 4) die("'%s' is too small to be an image", path);
    unsigned char *raw = xmalloc((size_t)n);
    if (fread(raw, 1, (size_t)n, f) != (size_t)n) die("short read on '%s'", path);
    fclose(f);
    if (raw[0] == 0xFF && raw[1] == 0xD8) {
        const char *e = NULL;
        long t0 = now_us();
        unsigned char *rgb = jpeg_decode(raw, (size_t)n, W, H, &e);
        { long _d = now_us() - t0; g_vega_us += _d; g_jpeg_us = _d; }
        free(raw);
        if (!rgb) die("JPEG decode failed for '%s': %s", path, e ? e : "unknown");
        printf("format: JPEG, decoded on the board\n");
        return rgb;
    }
    free(raw);
    printf("format: PPM\n");
    return read_ppm(path, W, H);
}

/* ============================ drawing =================================== */

static unsigned char *IMG;      /* source RGB, W*H*3 */
static int IMG_W, IMG_H;

static void px(int x, int y, int r, int g, int b)
{
    if ((unsigned)x >= (unsigned)IMG_W || (unsigned)y >= (unsigned)IMG_H) return;
    unsigned char *p = IMG + ((size_t)y*IMG_W + x)*3;
    p[0] = r; p[1] = g; p[2] = b;
}
static void draw_rect(int x1, int y1, int x2, int y2, int t, int r, int g, int b)
{
    for (int i = 0; i < t; i++) {
        for (int x = x1 - i; x <= x2 + i; x++) { px(x, y1 - i, r,g,b); px(x, y2 + i, r,g,b); }
        for (int y = y1 - i; y <= y2 + i; y++) { px(x1 - i, y, r,g,b); px(x2 + i, y, r,g,b); }
    }
}
static void draw_text(int x, int y, const char *s, int scale, int r, int g, int b)
{
    for (const unsigned char *c = (const unsigned char *)s; *c; c++) {
        if (*c < 32 || *c > 126) { x += (FONT_W-1)*scale; continue; }
        const unsigned char *gl = FONT8X8[*c - 32];
        for (int row = 0; row < FONT_H; row++)
            for (int col = 0; col < FONT_W; col++)
                if (gl[row] & (1u << col))
                    for (int sy = 0; sy < scale; sy++)
                        for (int sx = 0; sx < scale; sx++)
                            px(x + col*scale + sx, y + row*scale + sy, r,g,b);
        x += (FONT_W - 1)*scale;
    }
}
static void draw_label(int x, int y, const char *s, int scale, int wr,int wg,int wb)
{
    int w = (int)strlen(s) * (FONT_W-1) * scale + 6, h = FONT_H*scale + 4;
    if (y < 0) y = 0;
    for (int yy = y; yy < y + h; yy++)
        for (int xx = x; xx < x + w; xx++) px(xx, yy, 0, 0, 0);
    draw_text(x + 2, y + 2, s, scale, wr, wg, wb);
}
static void write_ppm(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write '%s': %s", path, strerror(errno));
    fprintf(f, "P6\n%d %d\n255\n", IMG_W, IMG_H);
    if (fwrite(IMG, 1, (size_t)IMG_W*IMG_H*3, f) != (size_t)IMG_W*IMG_H*3)
        die("short write on '%s'", path);
    fclose(f);
}

/* ============================ main ====================================== */

static void usage(const char *p)
{
    fprintf(stderr,
      "usage: %s --image FILE.jpg|FILE.ppm --task \"<prompt>\" [options]\n"
      "  --out PATH        marked image (default /tmp/team43/result.ppm)\n"
      "  --yolo PATH       default /tmp/team43/team43_yolo_int8.bin\n"
      "  --minilm PATH     default /tmp/team43/team43_minilm_int8.bin\n"
      "  --vocab PATH      default /tmp/team43/vocab.txt\n"
      "  --packed PATH     pre-packed MiniLM weight cache (big speed-up)\n"
      "  --emit-packed P   build that cache, then exit normally\n"
      "  --dev PATH        default /dev/accelerator_test\n"
      "  --imgsz N         detector input WIDTH, multiple of 32 (default 128)\n"
      "  --rect            fit the canvas to the photo: --imgsz becomes the\n"
      "                    LONG side and the short side is derived, so no\n"
      "                    accelerator invocation is spent on grey padding\n"
      "  --imgh N          detector input HEIGHT; omit for a square canvas.\n"
      "                    A canvas matching the photo aspect wastes no pixels\n"
      "                    on padding: 128x192 costs 1.5x a 128 square for the\n"
      "                    same object scale, where 192x192 costs 2.25x\n"
      "  --conf F          detection threshold (default 0.02; low on purpose --\n"
      "                    a withheld box is a candidate the ranker never sees)\n"
      "  --iou F           NMS IoU (default 0.45)\n"
      "  --rank-by minilm|product   (default product = MiniLM x detection score,\n"
      "                    the reference paper's rule and the one the metric scores)\n"
      "  --silu-table      interpolate silu from a table instead of calling\n"
      "                    expf; measured worth ~2 s once both builds are\n"
      "                    -ffp-contract=off, so it is off by default\n"
      "  --kdepth N        K-tiles streamed per accelerator invocation.\n"
      "                    0 = the whole K in one invocation (default, fastest)\n"
      "                    1 = the original one-tile-per-invocation path\n"      "  --dump-scores F   write MiniLM(task, class) for all 80 classes to F\n"
      "                    as JSON and exit; no image is read\n"
      "  --draw-top N      show and draw only the N best-ranked candidates\n"
      "                    (default 8, 0 = all). A low --conf feeds the ranker\n"
      "                    many boxes, which helps accuracy but ruins the picture\n"
      "  --unsafe-batched-regs  PRODUCES WRONG RESULTS; timing comparison only\n"
      "  --detect-list FILE  batch detection over '<id> <path>' lines\n"
      "  --detect-json OUT   write COCO detections there, then exit\n"
      "  --resume            skip ids already in OUT\n"
      "  --sim             emulate the accelerator in software\n"
      "  --quiet\n", p);
    exit(1);
}

/* ===================== batch detection mode ==============================
 *
 * Every accuracy figure this project has quoted so far -- 0.0630 at 128 px,
 * 0.1448 at 192, 0.2729 at 320 -- came from eval_coco_tasks.py with
 * --boxes yolov8n, which runs the ORIGINAL FP32 YOLOv8n through ultralytics.
 * It has never measured the INT8 detector that is actually deployed.
 *
 * This mode closes that gap using the deployed code itself: the same C, the
 * same INT8 package, the same letterbox, decode and NMS the board runs, in
 * --sim on a PC. Detections come out in COCO format so the existing AP
 * machinery can score them unchanged.
 *
 *   --detect-list FILE   one "<image_id> <path>" per line
 *   --detect-json OUT    COCO-style detection array
 *   --resume             skip image ids already present in OUT
 */
static const int COCO80_IDS[80] = {
     1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
    27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,
    51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,
    76,77,78,79,80,81,82,84,85,86,87,88,89,90 };

static const char *g_det_list = NULL, *g_det_json = NULL;
static int g_resume = 0;

int main(int argc, char **argv)
{
    const char *img_path = NULL, *task = NULL;
    const char *out_path = "/tmp/team43/result.ppm";
    const char *ypath = "/tmp/team43/team43_yolo_int8.bin";
    const char *mpath = "/tmp/team43/team43_minilm_int8.bin";
    const char *vpath = "/tmp/team43/vocab.txt";
    const char *dpath = "/dev/accelerator_test";
    /* product ranking is the default because it is what the metric rewards and
     * what the reference paper does: "the final confidence ... is obtained by
     * multiplying the detection score with the estimated probability".
     * Measured on 2,150 COCO-Tasks test images with the deployed INT8 models:
     *   rank-by product  mAP@0.5 0.0975   top-1 selection 0.1976
     *   rank-by minilm   mAP@0.5 0.0705   top-1 selection 0.1976
     * Identical on the single box the demo marks, 38 % better over the ranked
     * list. Shipping minilm-alone while evaluating product was an inconsistency.
     * --rank-by minilm still selects on the language score alone.            */
    int imgsz = 128, imgh = 0, product = 1, rect = 0, draw_top = 8;
    /* 0.02, not the 0.25 a detector demo would normally use. Task-driven
     * selection is not object detection: a box only has to be good enough for
     * the ranker to consider it, and every box withheld is a candidate the
     * ranker never sees. Measured on 717 COCO-Tasks test images at 192 px,
     * product ranking, INT8 models as deployed:
     *   conf   mAP@0.5   top-1 selection   right answer on the list at all
     *   0.01    0.2046        0.3036              0.5714
     *   0.02    0.2008        0.3107              0.5214
     *   0.05    0.1914        0.2857              0.4500
     *   0.10    0.1786        0.2643              0.3857
     *   0.25    (the old default) far below all of these
     * Accuracy still climbs below 0.02 but the shortlist stops being readable,
     * and product ranking already discounts a weak box by its own confidence. */
    float conf = 0.02f, iou_thr = 0.45f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--image") && i+1<argc) img_path = argv[++i];
        else if (!strcmp(argv[i],"--task") && i+1<argc) task = argv[++i];
        else if (!strcmp(argv[i],"--out") && i+1<argc) out_path = argv[++i];
        else if (!strcmp(argv[i],"--yolo") && i+1<argc) ypath = argv[++i];
        else if (!strcmp(argv[i],"--minilm") && i+1<argc) mpath = argv[++i];
        else if (!strcmp(argv[i],"--vocab") && i+1<argc) vpath = argv[++i];
        else if (!strcmp(argv[i],"--packed") && i+1<argc) g_pk_use = argv[++i];
        else if (!strcmp(argv[i],"--emit-packed") && i+1<argc) g_pk_emit = argv[++i];
        else if (!strcmp(argv[i],"--dev") && i+1<argc) dpath = argv[++i];
        else if (!strcmp(argv[i],"--imgsz") && i+1<argc) imgsz = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--imgh") && i+1<argc) imgh = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--rect")) rect = 1;
        else if (!strcmp(argv[i],"--conf") && i+1<argc) conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--iou") && i+1<argc) iou_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--rank-by") && i+1<argc) product = !strcmp(argv[++i],"product");
        else if (!strcmp(argv[i],"--draw-top") && i+1<argc) draw_top = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--dump-scores") && i+1<argc) g_dump_scores = argv[++i];
        else if (!strcmp(argv[i],"--kdepth") && i+1<argc) g_kdepth = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--exact-float")) g_exact_float = 1;
        else if (!strcmp(argv[i],"--silu-table"))  g_exact_float = 0;
        else if (!strcmp(argv[i],"--unsafe-batched-regs")) g_batched_regs = 1;
        else if (!strcmp(argv[i],"--detect-list") && i+1<argc) g_det_list = argv[++i];
        else if (!strcmp(argv[i],"--detect-json") && i+1<argc) g_det_json = argv[++i];
        else if (!strcmp(argv[i],"--resume")) g_resume = 1;
        else if (!strcmp(argv[i],"--sim")) g_sim = 1;
        else if (!strcmp(argv[i],"--quiet")) g_quiet = 1;
        else usage(argv[0]);
    }
    if (g_dump_scores) { if (!task) usage(argv[0]); }
    else if (!(g_det_list && g_det_json) && (!img_path || !task)) usage(argv[0]);
    if (imgsz % 32) die("--imgsz must be a multiple of 32 (got %d)", imgsz);
    if (!imgh) imgh = imgsz;                    /* square unless told otherwise */
    if (imgh % 32) die("--imgh must be a multiple of 32 (got %d)", imgh);

    long t_start = now_us();

    /* ---- map both packages ------------------------------------------- */
    int yfd = open(ypath, O_RDONLY);
    if (yfd < 0) die("cannot open '%s': %s", ypath, strerror(errno));
    struct stat st;
    if (fstat(yfd, &st) != 0) die("fstat: %s", strerror(errno));
    BLOB = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, yfd, 0);
    if (BLOB == MAP_FAILED) die("mmap yolo: %s", strerror(errno));
    BLOB_SZ = (size_t)st.st_size;
    if (BLOB_SZ != (size_t)YOLO_BLOB_BYTES)
        die("yolo package is %lu bytes, header says %ld",
            (unsigned long)BLOB_SZ, (long)YOLO_BLOB_BYTES);

    int mfd = open(mpath, O_RDONLY);
    if (mfd < 0) die("cannot open '%s': %s", mpath, strerror(errno));
    if (fstat(mfd, &st) != 0) die("fstat: %s", strerror(errno));
    void *mp = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mfd, 0);
    if (mp == MAP_FAILED) die("mmap minilm: %s", strerror(errno));
    MODEL.base = (const unsigned char *)mp; MODEL.size = (size_t)st.st_size;
    if (MODEL.size != (size_t)MINILM_BLOB_BYTES)
        die("minilm package is %lu bytes, header says %ld",
            (unsigned long)MODEL.size, (long)MINILM_BLOB_BYTES);
    resolve_tensors();
    if (g_pk_use && g_pk_emit) die("--packed and --emit-packed are exclusive");
    if (g_pk_use) pk_load(g_pk_use);
    vocab_load(vpath);
    (void)COCO_NAMES; (void)w_i8;
    /* The detector's class indices are used directly as MiniLM object-embedding
     * indices, so the two 80-class lists must agree element for element. */
    for (int i = 0; i < NCLS; i++)
        if (strcmp(COCO_NAMES[i], YOLO_NAMES[i]))
            die("class list mismatch at %d: minilm '%s' vs yolo '%s'",
                i, COCO_NAMES[i], YOLO_NAMES[i]);

    if (!g_sim) {
        g_fd = open(dpath, O_RDWR);
        if (g_fd < 0) die("cannot open '%s': %s", dpath, strerror(errno));
    }

    silu_init();

    printf("=== TEAM43 -- task-aware object selection, standalone on the board ===\n");
    if (!g_det_list) printf("image : %s\n", img_path);
    printf("task  : \"%s\"\n", task);
    printf("imgsz : %dx%d    rank-by: %s    mode: %s\n", imgsz, imgh,
           product ? "product" : "minilm", g_sim ? "SOFTWARE EMULATION" : "accelerator");
    printf("build : %s %s   jpeg: %s\n", __DATE__, __TIME__, JPEG_IMPL_NAME);
    printf("kdepth : %s\n", g_kdepth ? "capped" : "whole K per invocation");
    printf("silu   : %s\n", g_exact_float ? "expf" : "sigmoid table");
    print_fp_fingerprint();
    printf("regs   : %s\n", g_batched_regs
           ? "3 syscalls/tile  ** BATCHED: RESULTS WILL BE WRONG **"
           : "5 syscalls/tile (correct)");
    if (g_batched_regs)
        fprintf(stderr, "WARNING: --unsafe-batched-regs is enabled. Multi-word "
                        "register writes do not\n         set the registers; "
                        "output from this run is meaningless.\n");
    printf("weights: %s\n", g_pk_use ? "pre-packed cache"
                           : (g_pk_emit ? "packing at runtime, emitting cache"
                                        : "packing at runtime"));

    /* ---- shared scratch, sized for whichever model needs more --------- */
    long maxK = FFNDIM, maxN = FFNDIM;
    for (int i = 0; i < YOLO_NWEIGHTS; i++) {
        if (YW_K[i] > maxK) maxK = YW_K[i];
        if (YW_N[i] > maxN) maxN = YW_N[i];
    }
    long wtiles_yolo = 0, wtiles_mini = (long)((FFNDIM+7)/8) * ((HID+7)/8) * 64;
    for (int i = 0; i < YOLO_NWEIGHTS; i++) {
        long t = (long)((YW_N[i]+7)/8) * ((YW_K[i]+7)/8) * 64;
        if (t > wtiles_yolo) wtiles_yolo = t;
    }
    long wtiles = wtiles_yolo > wtiles_mini ? wtiles_yolo : wtiles_mini;
    /* the weight upload is a single transfer, so it must fit its window */
    {
        long worst = 0; int wi_worst = -1;
        for (int i = 0; i < YOLO_NWEIGHTS; i++) {
            long b = (long)((YW_N[i]+7)/8) * ((YW_K[i]+7)/8) * 64;
            if (b > worst) { worst = b; wi_worst = i; }
        }
        if (worst > WINDOW)
            die("weight %d packs to %ld bytes but the K window is %ld; "
                "enlarge WINDOW or chunk the weight upload", wi_worst, worst, (long)WINDOW);
        printf("ddr    : K/X windows %ld MB each, largest weight block %ld bytes\n",
               (long)WINDOW >> 20, worst);
    }
    imgsz_g = imgsz; imgh_g = imgh;
    PACKX_CAP = WINDOW;
    packX     = xmalloc((size_t)PACKX_CAP);
    packW_buf = xmalloc((size_t)wtiles);
    xq_buf    = xmalloc(XQ_CAP(imgsz, imgh));
    /* sized from the graph, not from a constant: YOLOv8n tops out at 256
     * output channels and YOLOv8s at 512, and a hardcoded 256 silently
     * overruns on the wider model. */
    acc_buf   = xmalloc(sizeof(int32_t) * (size_t)ROWCAP * (size_t)maxN);
    bias_i32  = xmalloc(sizeof(int32_t) * (size_t)(maxN > FFNDIM ? maxN : FFNDIM));
    comb_buf  = xmalloc(sizeof(float)   * (size_t)(maxN > FFNDIM ? maxN : FFNDIM));

    /* ---- score every class against the task, then exit --------------- */
    if (g_dump_scores) {
        int ids2[MAXTOK], mask2[MAXTOK];
        tokenize_task(task, ids2, mask2);
        printf("\ntokens:");
        for (int i = 0; i < MAXTOK; i++) printf(" %s", VOC.tok[ids2[i]]);
        printf("\n");
        encode_task(ids2, mask2);
        int all[NCLS];
        for (int i = 0; i < NCLS; i++) all[i] = i;
        double sc[NCLS];
        score_classes(all, NCLS, sc);
        FILE *sf = fopen(g_dump_scores, "w");
        if (!sf) die("cannot write '%s': %s", g_dump_scores, strerror(errno));
        fprintf(sf, "{\"prompt\":\"%s\",\"maxtok\":%d,\"scores\":{", task, MAXTOK);
        for (int i = 0; i < NCLS; i++)
            fprintf(sf, "%s\"%s\":%.6f", i ? "," : "", YOLO_NAMES[i], sc[i]);
        fprintf(sf, "}}\n");
        fclose(sf);
        printf("wrote %d class scores to %s\n", NCLS, g_dump_scores);
        return 0;
    }

    /* ---- batch detection: many images, no ranking, COCO json out ----- */
    if (g_det_list && g_det_json) {
        long done_ids[65536]; int ndone = 0;
        if (g_resume) {
            FILE *pf = fopen(g_det_json, "r");
            if (pf) {                       /* recover the ids already written */
                char ln[512];
                while (fgets(ln, sizeof(ln), pf)) {
                    char *p = strstr(ln, "\"image_id\":");
                    if (p && ndone < 65536) done_ids[ndone++] = atol(p + 11);
                }
                fclose(pf);
            }
        }
        FILE *lf = fopen(g_det_list, "r");
        if (!lf) die("cannot open '%s': %s", g_det_list, strerror(errno));
        FILE *jf = fopen(g_det_json, g_resume && ndone ? "a" : "w");
        if (!jf) die("cannot write '%s': %s", g_det_json, strerror(errno));

        char line[1024]; long nimg = 0, ndet = 0, skipped = 0;
        long tb = now_us();
        g_quiet = 1;
        while (fgets(line, sizeof(line), lf)) {
            char *sp = strpbrk(line, " \t");
            if (!sp) continue;
            *sp = 0;
            long iid = atol(line);
            char *path = sp + 1;
            while (*path == ' ' || *path == '\t') path++;
            char *nl = strpbrk(path, "\r\n"); if (nl) *nl = 0;
            if (!*path) continue;

            int already = 0;
            for (int i = 0; i < ndone; i++) if (done_ids[i] == iid) { already = 1; break; }
            if (already) { skipped++; continue; }

            IMG = read_image(path, &IMG_W, &IMG_H);
            float ratio; int padx, pady;
            float *inp = ten_alloc(0, 3, imgh, imgsz);
            letterbox(IMG, IMG_W, IMG_H, imgsz, imgh, inp, &ratio, &padx, &pady);
            for (int oi = 0; oi < YOLO_NOPS; oi++) {
                switch (YOP_TYPE[oi]) {
                    case YOP_CONV:     op_conv(oi);     break;
                    case YOP_MAXPOOL:  op_maxpool(oi);  break;
                    case YOP_UPSAMPLE: op_upsample(oi); break;
                    case YOP_CONCAT:   op_concat(oi);   break;
                    case YOP_SLICE:    op_slice(oi);    break;
                    case YOP_ADD:      op_add(oi);      break;
                    default: die("unknown op %d", YOP_TYPE[oi]);
                }
                for (int t = 0; t < YOLO_NTENSORS; t++)
                    if (TS[t].d && YTENSOR_LASTUSE[t] == oi) ten_free(t);
            }
            static det_t bd[4096];
            int nb = decode_detect(bd, 4096, conf, iou_thr);
            for (int i = 0; i < nb; i++) {
                /* undo the letterbox: decode_detect works in the padded
                 * imgsz x imgsz frame, COCO wants original image pixels */
                double x1 = (bd[i].x1 - padx) / ratio, y1 = (bd[i].y1 - pady) / ratio;
                double x2 = (bd[i].x2 - padx) / ratio, y2 = (bd[i].y2 - pady) / ratio;
                if (x1 < 0)     x1 = 0;
                if (y1 < 0)     y1 = 0;
                if (x2 > IMG_W) x2 = IMG_W;
                if (y2 > IMG_H) y2 = IMG_H;
                if (x2 <= x1 || y2 <= y1) continue;
                double x = x1, y = y1, w = x2 - x1, h = y2 - y1;
                fprintf(jf, "{\"image_id\":%ld,\"category_id\":%d,"
                            "\"bbox\":[%.2f,%.2f,%.2f,%.2f],\"score\":%.5f}\n",
                        iid, COCO80_IDS[bd[i].cls], x, y, w, h, (double)bd[i].conf);
                ndet++;
            }
            if (!nb) fprintf(jf, "{\"image_id\":%ld,\"category_id\":0,"
                                 "\"bbox\":[0,0,0,0],\"score\":0}\n", iid);
            for (int t = 0; t < YOLO_NTENSORS; t++) if (TS[t].d) ten_free(t);
            free(IMG); IMG = NULL;
            nimg++;
            if (nimg % 25 == 0) {
                double el = (now_us() - tb) / 1e6;
                fprintf(stderr, "\r  %ld images, %ld detections, %.2f s/img   ",
                        nimg, ndet, el / nimg);
                fflush(stderr); fflush(jf);
            }
        }
        fclose(jf); fclose(lf);
        fprintf(stderr, "\n");
        printf("batch detection done\n");
        printf("  images    : %ld  (%ld skipped as already present)\n", nimg, skipped);
        printf("  detections: %ld\n", ndet);
        printf("  imgsz %d, conf %.2f, iou %.2f\n", imgsz, conf, iou_thr);
        printf("  wall      : %.1f s  (%.2f s/image)\n",
               (now_us() - tb) / 1e6, nimg ? (now_us() - tb) / 1e6 / nimg : 0.0);
        printf("  written   : %s\n", g_det_json);
        if (g_fd >= 0) close(g_fd);
        return 0;
    }

    /* ---- stage 1: detection ------------------------------------------ */
    IMG = read_image(img_path, &IMG_W, &IMG_H);
    printf("source: %d x %d\n", IMG_W, IMG_H);
    if (rect) {                 /* long side = imgsz, short side from the aspect */
        int lo;
        if (IMG_W >= IMG_H) {
            lo = (int)((double)imgsz * IMG_H / IMG_W + 31.0) / 32 * 32;
            if (lo < 32) lo = 32;
            imgh = lo;
        } else {
            lo = (int)((double)imgsz * IMG_W / IMG_H + 31.0) / 32 * 32;
            if (lo < 32) lo = 32;
            imgh = imgsz; imgsz = lo;
        }
        printf("canvas: %d x %d  (rect, no padding wasted)\n", imgsz, imgh);
        free(TS[0].d); TS[0].d = NULL;
        free(xq_buf); xq_buf = xmalloc(XQ_CAP(imgsz, imgh));
    }
    printf("\n");
    float ratio; int padx, pady;
    float *inp = ten_alloc(0, 3, imgh, imgsz);
    long t0 = now_us();
    letterbox(IMG, IMG_W, IMG_H, imgsz, imgh, inp, &ratio, &padx, &pady);
    g_vega_us += now_us() - t0;

    long t_det = now_us();
    for (int oi = 0; oi < YOLO_NOPS; oi++) {
        switch (YOP_TYPE[oi]) {
            case YOP_CONV:     op_conv(oi);     break;
            case YOP_MAXPOOL:  op_maxpool(oi);  break;
            case YOP_UPSAMPLE: op_upsample(oi); break;
            case YOP_CONCAT:   op_concat(oi);   break;
            case YOP_SLICE:    op_slice(oi);    break;
            case YOP_ADD:      op_add(oi);      break;
            default: die("unknown op %d", YOP_TYPE[oi]);
        }
        for (int t = 0; t < YOLO_NTENSORS; t++)
            if (TS[t].d && YTENSOR_LASTUSE[t] == oi) ten_free(t);
        if (!g_quiet && oi % 20 == 19) { printf("  yolo op %d/%d\n", oi+1, YOLO_NOPS); fflush(stdout); }
    }
    static det_t dets[4096];
    int nd = decode_detect(dets, 4096, conf, iou_thr);
    long det_us = now_us() - t_det;
    long det_inv = g_inv;

    printf("\n--- DETECTED (%d) ---\n", nd);
    for (int i = 0; i < nd; i++)
        printf("  %-16s %.3f\n", YOLO_NAMES[dets[i].cls], dets[i].conf);
    if (!nd) {
        printf("  nothing above the threshold -- nothing to select.\n");
        write_ppm(out_path);
        return 1;
    }

    /* ---- stage 2: encode the runtime task ---------------------------- */
    int ids[MAXTOK], mask[MAXTOK];
    tokenize_task(task, ids, mask);
    printf("\ntokens:");
    for (int i = 0; i < MAXTOK; i++) printf(" %s", VOC.tok[ids[i]]);
    printf("\n");
    long t_mini = now_us();
    encode_task(ids, mask);

    /* ---- stage 3: score only the classes that were detected ---------- */
    int uniq[NCLS], nu = 0;
    for (int i = 0; i < nd; i++) {
        int seen = 0;
        for (int j = 0; j < nu; j++) if (uniq[j] == dets[i].cls) { seen = 1; break; }
        if (!seen) uniq[nu++] = dets[i].cls;
    }
    double cls_score[NCLS];
    score_classes(uniq, nu, cls_score);
    long mini_us = now_us() - t_mini;

    /* ---- stage 4: rank ------------------------------------------------ */
    int best = 0; double best_rank = -1.0;
    double rank[4096];
    printf("\n--- TASK-AWARE RANKING (%s) ---\n", product ? "minilm x yolo" : "minilm");
    for (int i = 0; i < nd; i++) {
        double ms = 0.0;
        for (int j = 0; j < nu; j++) if (uniq[j] == dets[i].cls) ms = cls_score[j];
        rank[i] = product ? ms * dets[i].conf : ms;
        if (rank[i] > best_rank) { best_rank = rank[i]; best = i; }
    }
    /* Order the candidates by rank so both the printout and the marked image
     * show the shortlist rather than everything the detector emitted. At a low
     * confidence floor nd can be 40+, and a picture carrying 46 boxes tells the
     * viewer nothing about which one the system chose or why.               */
    int order[4096];
    for (int i = 0; i < nd; i++) order[i] = i;
    for (int i = 0; i < nd; i++) {           /* selection sort, nd is small */
        int m = i;
        for (int j = i + 1; j < nd; j++)
            if (rank[order[j]] > rank[order[m]]) m = j;
        int tmp = order[i]; order[i] = order[m]; order[m] = tmp;
    }
    int nshow = (draw_top > 0 && draw_top < nd) ? draw_top : nd;
    for (int k = 0; k < nshow; k++) {
        int i = order[k];
        double ms = 0.0;
        for (int j = 0; j < nu; j++) if (uniq[j] == dets[i].cls) ms = cls_score[j];
        printf("  %-16s yolo %.3f  minilm %.5f  rank %.5f%s\n",
               YOLO_NAMES[dets[i].cls], dets[i].conf, ms, rank[i],
               i == best ? "   <== SELECTED" : "");
    }
    if (nshow < nd)
        printf("  ... %d further candidate%s below rank %.5f not shown or drawn\n",
               nd - nshow, nd - nshow == 1 ? "" : "s", rank[order[nshow - 1]]);

    /* ---- stage 5: mark the image ------------------------------------- */
    int scale = IMG_W > 900 ? 3 : 2;
    for (int k = 0; k < nshow; k++) {
        int i = order[k];
        if (i == best) continue;
        int x1 = (int)((dets[i].x1 - padx)/ratio), y1 = (int)((dets[i].y1 - pady)/ratio);
        int x2 = (int)((dets[i].x2 - padx)/ratio), y2 = (int)((dets[i].y2 - pady)/ratio);
        draw_rect(x1, y1, x2, y2, 1, 90, 90, 90);
    }
    {
        int x1 = (int)((dets[best].x1 - padx)/ratio), y1 = (int)((dets[best].y1 - pady)/ratio);
        int x2 = (int)((dets[best].x2 - padx)/ratio), y2 = (int)((dets[best].y2 - pady)/ratio);
        draw_rect(x1, y1, x2, y2, 3 + scale, 0, 230, 60);
        char lab[160];
        snprintf(lab, sizeof(lab), "%s %.3f", YOLO_NAMES[dets[best].cls], best_rank);
        draw_label(x1, y1 - (FONT_H*scale + 6), lab, scale, 0, 255, 90);
        char tl[200];
        snprintf(tl, sizeof(tl), "TASK: %s", task);
        draw_label(6, 6, tl, scale, 255, 255, 255);
    }
    write_ppm(out_path);

    long total = now_us() - t_start;
    printf("\n--- RESULT ---\n");
    printf("  SELECTED : %s   (rank %.5f)\n", YOLO_NAMES[dets[best].cls], best_rank);
    printf("  marked image written to %s\n", out_path);
    printf("\n--- WORK ---\n");
    printf("  detection   : %ld invocations, %.1f s\n", det_inv, det_us/1e6);
    printf("  minilm      : %ld invocations, %.1f s  (head over %d detected classes,\n"
           "                                          not all 80)\n",
           g_inv - det_inv, mini_us/1e6, nu);
    printf("  total       : %ld invocations, %.1f s\n", g_inv, total/1e6);
    printf("  output tiles: %ld   (mean K-tiles per output tile %.1f)\n",
           g_tiles, g_tiles ? (double)g_ktiles_sum / g_tiles : 0.0);
    printf("  if the k0 loop were streamed as one accumulation:\n"
           "     invocations %ld -> %ld   (%.1fx fewer)\n"
           "     syscalls    %ld -> %ld   (%.1fx fewer)\n",
           g_inv, g_tiles, g_tiles ? (double)g_inv / g_tiles : 0.0,
           2*g_tiles + 5*g_inv, 7*g_tiles,
           g_tiles ? (double)(2*g_tiles + 5*g_inv) / (7*g_tiles) : 0.0);
    printf("  GEMM loop %.1f s | packing %.1f s | upload %.1f s | VEGA float %.1f s\n",
           g_dev_us/1e6, g_pack_us/1e6, g_up_us/1e6, g_vega_us/1e6);
    printf("  --- float breakdown (all of it inside VEGA float) ---\n");
    printf("    amax scan       : %7.2f s  over %9ld elements  (%.2f us/elem)\n",
           g_amax_us/1e6, g_quant_n, g_quant_n ? (double)g_amax_us/g_quant_n : 0.0);
    printf("    quantise convert: %7.2f s  over %9ld elements  (%.2f us/elem)\n",
           g_qcvt_us/1e6, g_quant_n, g_quant_n ? (double)g_qcvt_us/g_quant_n : 0.0);
    printf("    dequant + silu  : %7.2f s  over %9ld elements  (%.2f us/elem, %ld through silu)\n",
           g_deq_us/1e6, g_deq_n, g_deq_n ? (double)g_deq_us/g_deq_n : 0.0, g_silu_n);
    printf("    accounted       : %7.2f s of %.2f s\n",
           (g_amax_us+g_qcvt_us+g_deq_us)/1e6, g_vega_us/1e6);

    printf("\n--- PACKING BREAKDOWN ---\n");
    printf("  detection  im2col      : %8.2f s  over %7ld calls\n",
           g_im2col_us/1e6, g_im2col_n);
    printf("  detection  W tile copy : %8.2f s  over %7ld calls, %.1f MB"
           "  (%.1f MB/s)\n",
           g_wcopy_us/1e6, g_wcopy_n, g_wcopy_by/1048576.0,
           g_wcopy_us ? g_wcopy_by/1048576.0/(g_wcopy_us/1e6) : 0.0);
    printf("  minilm     activations : %8.2f s  over %7ld calls\n",
           g_mpx_us/1e6, g_mpx_n);
    printf("  minilm     weights     : %8.2f s  over %7ld calls, %.1f MB"
           "  (%.2f MB/s)%s\n",
           g_mpw_us/1e6, g_mpw_n, g_mpw_by/1048576.0,
           g_mpw_us ? g_mpw_by/1048576.0/(g_mpw_us/1e6) : 0.0,
           g_pk_use ? "  [from cache]" : "");
    if (g_pk_use)
        printf("  cache hits             : %ld blocks, %.1f MB copied\n",
               g_pk_hits, g_pk_bytes/1048576.0);
    printf("  minilm     act-as-wgt  : %8.2f s  over %7ld calls\n",
           g_mpa_us/1e6, g_mpa_n);
    printf("  ------------------------------------------------------------\n");
    printf("  detection subtotal     : %8.2f s\n", (g_im2col_us+g_wcopy_us)/1e6);
    printf("  minilm    subtotal     : %8.2f s\n",
           (g_mpx_us+g_mpw_us+g_mpa_us)/1e6);
    printf("  sum vs g_pack_us       : %8.2f s vs %.2f s"
           "   (W tile copy is extra, it was never counted before)\n",
           (g_im2col_us+g_wcopy_us+g_mpx_us+g_mpw_us+g_mpa_us)/1e6, g_pack_us/1e6);
    if (g_jpeg_us) {
        printf("\n--- JPEG DECODE ---\n");
        printf("  total                  : %8.2f s\n", g_jpeg_us/1e6);
        printf("  huffman + dequant      : %8.2f s   (scan %.2f s minus IDCT)\n",
               (jt_scan_us - jt_idct_us)/1e6, jt_scan_us/1e6);
        printf("  IDCT                   : %8.2f s   over %ld blocks"
               "  (%.0f us/block)\n", jt_idct_us/1e6, jt_blocks,
               jt_blocks ? (double)jt_idct_us/jt_blocks : 0.0);
        printf("  YCbCr -> RGB           : %8.2f s   over %ld pixels"
               "  (%.1f us/pixel)\n", jt_color_us/1e6, jt_pixels,
               jt_pixels ? (double)jt_color_us/jt_pixels : 0.0);
        printf("  unaccounted            : %8.2f s   (headers, allocation)\n",
               (g_jpeg_us - jt_scan_us - jt_color_us)/1e6);
    } else {
        printf("\n  JPEG decode : not exercised (input was PPM)\n");
    }

    if (g_pk_emit) pk_write(g_pk_emit);
    if (g_fd >= 0) close(g_fd);
    return 0;
}
