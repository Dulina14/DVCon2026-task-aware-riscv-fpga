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

#define K_ADDR    0xBF000000UL
#define X_ADDR    0xBF100000UL
#define Y_ADDR    0xBF300000UL
#define WINDOW    0x100000L               /* 1 MB per scratch window */

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

static int  g_fd = -1, g_sim = 0, g_quiet = 0;
static long g_inv = 0, g_sys = 0;
static long g_dev_us = 0, g_pack_us = 0, g_up_us = 0, g_vega_us = 0;

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
static void wait_done(void)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++) if (rr(REG_Y_DONE) & 1u) return;
    die("Y_DONE never asserted at invocation %ld", g_inv);
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
static void gemm_run(const int8_t *packX, const int8_t *packW,
                     int M, int K, int N, const int32_t *bias, int32_t *out,
                     int upload_w)
{
    int MT = (M + 7)/8, KT = (K + 7)/8, NT = (N + 7)/8;
    int32_t par[64];
    long t0 = now_us();
    if (!g_sim) {
        dw_bulk(X_ADDR, packX, (long)MT * KT * 64);
        /* The weight tiles are identical for every M-chunk of one convolution,
         * so they are uploaded on the first chunk only. */
        if (upload_w) dw_bulk(K_ADDR, packW, (long)NT * KT * 64);
        rw(REG_K_BYTES, INPUT_BYTES); rw(REG_K_TUSER, 0u);
        rw(REG_X_BYTES, INPUT_BYTES); rw(REG_X_TUSER, 0u);
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
            for (int k0 = 0; k0 < KT; k0++) {
                if (g_sim) sim_tile(packX + ((long)m0*KT + k0)*64,
                                    packW + ((long)n0*KT + k0)*64, par);
                else {
                    rw(REG_START, 0u);
                    rw(REG_K_ADDR, (uint32_t)(K_ADDR + ((unsigned long)n0*KT + k0)*64));
                    rw(REG_X_ADDR, (uint32_t)(X_ADDR + ((unsigned long)m0*KT + k0)*64));
                    rw(REG_START, 1u);
                    wait_done();
                }
                g_inv++;
            }
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

/* letterbox: preserve aspect, bilinear resize, pad to sz with 114, /255 */
static void letterbox(const unsigned char *src, int W, int H, int sz,
                      float *dst, float *ratio, int *padx, int *pady)
{
    double r = (double)sz / W; if ((double)sz / H < r) r = (double)sz / H;
    int nw = (int)(W * r + 0.5), nh = (int)(H * r + 0.5);
    int px = (sz - nw) / 2, py = (sz - nh) / 2;
    *ratio = (float)r; *padx = px; *pady = py;
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < sz*sz; i++) dst[(size_t)c*sz*sz + i] = 114.0f / 255.0f;
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
                dst[((size_t)c*sz + (py+oy))*sz + (px+ox)] = (float)(v / 255.0);
            }
        }
    }
}

/* ===================================================== conv ============== */

static int8_t  *packX, *packW_buf, *xq_buf;
static int32_t *acc_buf, *bias_i32;
static long     PACKX_CAP;

static float silu(float x) { return x / (1.0f + expf(-x)); }

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
    /* quantize the whole input tensor once, per tensor */
    long nin = (long)X->c * X->h * X->w;
    float amax = 0.0f;
    for (long i = 0; i < nin; i++) { float a = fabsf(X->d[i]); if (a > amax) amax = a; }
    float sx = (amax > 0.0f) ? amax / 127.0f : 1.0f;
    for (long i = 0; i < nin; i++) {
        long q = lrintf(X->d[i] / sx);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        xq_buf[i] = (int8_t)q;
    }
    g_vega_us += now_us() - t0;

    /* Copy the weight tiles out of the mmap'd package into RAM once per conv.
     * This is NOT just for the upload: --sim used to read tiles straight from
     * the mapping, and every W tile is re-read once per M row-tile. On the
     * board that cost 4,502 us per tile (1,540 s for one image) against 216 us
     * for the same sim_tile in Milestone B, because VEGA's mapped-file reads
     * are pathologically slow. One bulk copy fixes it for both paths. */
    memcpy(packW_buf, w_packed(wi), (size_t)NT * KT * 64);

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
        g_pack_us += now_us() - t0;

        const float *sw = w_scale(wi), *bf = w_bias(wi);
        for (int n = 0; n < N; n++) {
            double comb = (double)sx * (double)sw[n];
            bias_i32[n] = (int32_t)lround((double)bf[n] / comb);
        }
        gemm_run(packX, packW_buf, rows, K, N,
                 bias_i32, acc_buf, first_chunk);
        first_chunk = 0;

        t0 = now_us();
        for (int n = 0; n < N; n++) {
            float comb = sx * sw[n];
            float *dst = Y + (size_t)n * M + m_base;
            const int32_t *src = acc_buf + n;
            for (int r = 0; r < rows; r++) {
                float v = (float)src[(long)r * N] * comb;
                dst[r] = YOP_ACT[oi] ? silu(v) : v;
            }
        }
        g_vega_us += now_us() - t0;
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
    g_pack_us += now_us() - t0;
}
static void mpack_w_pkg(const tensor_t *W, int K, int N)
{
    int KT = K/8, NT = (N + 7)/8;
    const int8_t *p = (const int8_t *)W->p;
    int D1 = W->d1;
    long t0 = now_us();
    memset(packW_buf, 0, (size_t)NT * KT * 64);
    for (int c = 0; c < N; c++) {
        const int8_t *row = p + (size_t)c * D1;
        int8_t *base = packW_buf + (long)(c/8) * KT * 64 + (c % 8);
        for (int k = 0; k < K; k++) base[(long)(k/8)*64 + (k%8)*8] = row[k];
    }
    g_pack_us += now_us() - t0;
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
    g_pack_us += now_us() - t0;
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
        g_vega_us += now_us() - t0;
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
      "  --dev PATH        default /dev/accelerator_test\n"
      "  --imgsz N         detector input, multiple of 32 (default 128)\n"
      "  --conf F          detection threshold (default 0.25)\n"
      "  --iou F           NMS IoU (default 0.45)\n"
      "  --rank-by minilm|product\n"
      "  --sim             emulate the accelerator in software\n"
      "  --quiet\n", p);
    exit(1);
}

int main(int argc, char **argv)
{
    const char *img_path = NULL, *task = NULL;
    const char *out_path = "/tmp/team43/result.ppm";
    const char *ypath = "/tmp/team43/team43_yolo_int8.bin";
    const char *mpath = "/tmp/team43/team43_minilm_int8.bin";
    const char *vpath = "/tmp/team43/vocab.txt";
    const char *dpath = "/dev/accelerator_test";
    int imgsz = 128, product = 0;
    float conf = 0.25f, iou_thr = 0.45f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--image") && i+1<argc) img_path = argv[++i];
        else if (!strcmp(argv[i],"--task") && i+1<argc) task = argv[++i];
        else if (!strcmp(argv[i],"--out") && i+1<argc) out_path = argv[++i];
        else if (!strcmp(argv[i],"--yolo") && i+1<argc) ypath = argv[++i];
        else if (!strcmp(argv[i],"--minilm") && i+1<argc) mpath = argv[++i];
        else if (!strcmp(argv[i],"--vocab") && i+1<argc) vpath = argv[++i];
        else if (!strcmp(argv[i],"--dev") && i+1<argc) dpath = argv[++i];
        else if (!strcmp(argv[i],"--imgsz") && i+1<argc) imgsz = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--conf") && i+1<argc) conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--iou") && i+1<argc) iou_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--rank-by") && i+1<argc) product = !strcmp(argv[++i],"product");
        else if (!strcmp(argv[i],"--sim")) g_sim = 1;
        else if (!strcmp(argv[i],"--quiet")) g_quiet = 1;
        else usage(argv[0]);
    }
    if (!img_path || !task) usage(argv[0]);
    if (imgsz % 32) die("--imgsz must be a multiple of 32 (got %d)", imgsz);

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

    printf("=== TEAM43 -- task-aware object selection, standalone on the board ===\n");
    printf("image : %s\n", img_path);
    printf("task  : \"%s\"\n", task);
    printf("imgsz : %d    rank-by: %s    mode: %s\n", imgsz,
           product ? "product" : "minilm", g_sim ? "SOFTWARE EMULATION" : "accelerator");

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
    PACKX_CAP = WINDOW;
    packX     = xmalloc((size_t)PACKX_CAP);
    packW_buf = xmalloc((size_t)wtiles);
    xq_buf    = xmalloc((size_t)16 * imgsz * imgsz + 1024);
    acc_buf   = xmalloc(sizeof(int32_t) * (size_t)ROWCAP * 256);
    bias_i32  = xmalloc(sizeof(int32_t) * (size_t)(maxN > FFNDIM ? maxN : FFNDIM));

    /* ---- stage 1: detection ------------------------------------------ */
    IMG = read_image(img_path, &IMG_W, &IMG_H);
    printf("source: %d x %d\n\n", IMG_W, IMG_H);
    float ratio; int padx, pady;
    float *inp = ten_alloc(0, 3, imgsz, imgsz);
    long t0 = now_us();
    letterbox(IMG, IMG_W, IMG_H, imgsz, inp, &ratio, &padx, &pady);
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
    for (int i = 0; i < nd; i++) {
        double ms = 0.0;
        for (int j = 0; j < nu; j++) if (uniq[j] == dets[i].cls) ms = cls_score[j];
        printf("  %-16s yolo %.3f  minilm %.5f  rank %.5f%s\n",
               YOLO_NAMES[dets[i].cls], dets[i].conf, ms, rank[i],
               i == best ? "   <== SELECTED" : "");
    }

    /* ---- stage 5: mark the image ------------------------------------- */
    int scale = IMG_W > 900 ? 3 : 2;
    for (int i = 0; i < nd; i++) {
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
    printf("  GEMM loop %.1f s | packing %.1f s | upload %.1f s | VEGA float %.1f s\n",
           g_dev_us/1e6, g_pack_us/1e6, g_up_us/1e6, g_vega_us/1e6);

    if (g_fd >= 0) close(g_fd);
    return 0;
}
