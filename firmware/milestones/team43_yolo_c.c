/*
 * team43_yolo_c.c  --  DVCon India 2026, Stage 3B, Team 43 "Silicon Syndicate"
 *
 * MILESTONE C: YOLOv8n object detection on the Genesys 2, every convolution
 * executed as im2col GEMM on the 8x8x8 accelerator.
 *
 *   raw P6 image  ->  letterbox + normalise            (VEGA)
 *                 ->  103-op graph, 63 convolutions    (accelerator)
 *                 ->  SiLU / maxpool / upsample / concat / slice / add (VEGA)
 *                 ->  DFL softmax, box decode, sigmoid (VEGA)
 *                 ->  NMS                              (VEGA)
 *                 ->  detected COCO classes + confidences + boxes
 *
 * NOT YOLOv8x. This is YOLOv8n at reduced resolution, and it must never be
 * described as anything else. YOLOv8x at 640 is 128.9 G MAC = 251.8 M
 * invocations = ~25 hours per image at our measured 356 us; even a
 * zero-overhead interface leaves it at ~3.8 hours. YOLOv8n at 128 is 342,160
 * invocations (~122 s) and at 192 is 785,432 (~280 s).
 *
 * The image is supplied as raw P6 PPM because the board has no JPEG decoder.
 * Converting the container offline is not inference: the board still does the
 * letterbox, the normalisation, every convolution, the decode and the NMS.
 *
 * Build (dev PC):
 *   docker run --rm -t -v "$PWD":/workspace -w /workspace --user $(id -u):$(id -g) \
 *     buildroot-2004_new \
 *     bash -c 'buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc \
 *         -std=gnu99 -O2 -static -Wall -Wextra -o team43_yolo_c team43_yolo_c.c -lm -lrt'
 *
 * Run (board):
 *   /tmp/team43/team43_yolo_c --image /tmp/team43/bus.ppm --imgsz 128
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

/* ===================================================== main ============= */

static void usage(const char *p)
{
    fprintf(stderr,
      "usage: %s --image FILE.ppm [options]\n"
      "  --model PATH  default /tmp/team43/team43_yolo_int8.bin\n"
      "  --dev   PATH  default /dev/accelerator_test\n"
      "  --imgsz N     inference size, multiple of 32 (default 128)\n"
      "  --conf F      confidence threshold (default 0.25)\n"
      "  --iou  F      NMS IoU threshold (default 0.45)\n"
      "  --sim         emulate the accelerator in software\n"
      "  --quiet\n", p);
    exit(1);
}

int main(int argc, char **argv)
{
    const char *img_path = NULL;
    const char *mpath = "/tmp/team43/team43_yolo_int8.bin";
    const char *dpath = "/dev/accelerator_test";
    int imgsz = 128;
    float conf = 0.25f, iou_thr = 0.45f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--image") && i+1<argc) img_path = argv[++i];
        else if (!strcmp(argv[i],"--model") && i+1<argc) mpath = argv[++i];
        else if (!strcmp(argv[i],"--dev")   && i+1<argc) dpath = argv[++i];
        else if (!strcmp(argv[i],"--imgsz") && i+1<argc) imgsz = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--conf")  && i+1<argc) conf = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--iou")   && i+1<argc) iou_thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--sim")) g_sim = 1;
        else if (!strcmp(argv[i],"--quiet")) g_quiet = 1;
        else usage(argv[0]);
    }
    if (!img_path) usage(argv[0]);
    if (imgsz % 32) die("--imgsz must be a multiple of 32 (got %d)", imgsz);

    long t_start = now_us();

    int mfd = open(mpath, O_RDONLY);
    if (mfd < 0) die("cannot open '%s': %s", mpath, strerror(errno));
    struct stat st;
    if (fstat(mfd, &st) != 0) die("fstat: %s", strerror(errno));
    BLOB = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mfd, 0);
    if (BLOB == MAP_FAILED) die("mmap: %s", strerror(errno));
    BLOB_SZ = (size_t)st.st_size;
    if (BLOB_SZ != (size_t)YOLO_BLOB_BYTES)
        die("package is %lu bytes, header says %ld",
            (unsigned long)BLOB_SZ, (long)YOLO_BLOB_BYTES);

    if (!g_sim) {
        g_fd = open(dpath, O_RDWR);
        if (g_fd < 0) die("cannot open '%s': %s", dpath, strerror(errno));
    }

    printf("=== TEAM43 MILESTONE C -- YOLOv8n detection on the accelerator ===\n");
    printf("image : %s\n", img_path);
    printf("imgsz : %d    mode: %s\n", imgsz, g_sim ? "SOFTWARE EMULATION" : "accelerator");

    /* scratch sized for the largest conv at this resolution */
    long maxK = 0, maxN = 0;
    for (int i = 0; i < YOLO_NWEIGHTS; i++) {
        if (YW_K[i] > maxK) maxK = YW_K[i];
        if (YW_N[i] > maxN) maxN = YW_N[i];
    }
    PACKX_CAP = WINDOW;
    packX     = xmalloc((size_t)PACKX_CAP);
    packW_buf = xmalloc((size_t)((maxN+7)/8) * ((maxK+7)/8) * 64);
    xq_buf    = xmalloc((size_t)16 * imgsz * imgsz + 1024);
    acc_buf   = xmalloc(sizeof(int32_t) * (size_t)ROWCAP * (size_t)maxN);
    bias_i32  = xmalloc(sizeof(int32_t) * (size_t)maxN);

    int W, H;
    unsigned char *raw = read_ppm(img_path, &W, &H);
    printf("source: %d x %d\n", W, H);
    float ratio; int padx, pady;
    float *inp = ten_alloc(0, 3, imgsz, imgsz);
    long t0 = now_us();
    letterbox(raw, W, H, imgsz, inp, &ratio, &padx, &pady);
    g_vega_us += now_us() - t0;
    free(raw);

    for (int oi = 0; oi < YOLO_NOPS; oi++) {
        switch (YOP_TYPE[oi]) {
            case YOP_CONV:     op_conv(oi);     break;
            case YOP_MAXPOOL:  op_maxpool(oi);  break;
            case YOP_UPSAMPLE: op_upsample(oi); break;
            case YOP_CONCAT:   op_concat(oi);   break;
            case YOP_SLICE:    op_slice(oi);    break;
            case YOP_ADD:      op_add(oi);      break;
            default: die("unknown op type %d at %d", YOP_TYPE[oi], oi);
        }
        for (int t = 0; t < YOLO_NTENSORS; t++)
            if (TS[t].d && YTENSOR_LASTUSE[t] == oi) ten_free(t);
        if (!g_quiet && (oi % 20 == 19 || oi == YOLO_NOPS-1)) {
            printf("  op %3d/%d  (%ld invocations)\n", oi+1, YOLO_NOPS, g_inv);
            fflush(stdout);
        }
    }

    det_t dets[4096];
    int nd = decode_detect(dets, 4096, conf, iou_thr);
    long total = now_us() - t_start;

    printf("\n--- DETECTIONS (conf >= %.2f) ---\n", conf);
    for (int i = 0; i < nd; i++) {
        /* undo letterbox back to source pixels */
        float x1 = (dets[i].x1 - padx) / ratio, y1 = (dets[i].y1 - pady) / ratio;
        float x2 = (dets[i].x2 - padx) / ratio, y2 = (dets[i].y2 - pady) / ratio;
        printf("  %-16s %.3f   box [%.0f %.0f %.0f %.0f]\n",
               YOLO_NAMES[dets[i].cls], dets[i].conf, x1, y1, x2, y2);
    }
    if (!nd) printf("  (nothing above threshold)\n");

    printf("\n--- WORK ---\n");
    printf("  accelerator invocations : %ld\n", g_inv);
    printf("  syscalls                : %ld\n", g_sys);
    printf("  GEMM loop               : %.2f s\n", g_dev_us / 1e6);
    printf("  im2col + tile packing   : %.2f s\n", g_pack_us / 1e6);
    printf("  operand upload          : %.2f s\n", g_up_us / 1e6);
    printf("  VEGA float work         : %.2f s\n", g_vega_us / 1e6);
    printf("  total wall time         : %.2f s\n", total / 1e6);

    if (g_fd >= 0) close(g_fd);
    return 0;
}
