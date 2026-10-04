/*
 * team43_minilm_b.c  --  DVCon India 2026, Stage 3B, Team 43 "Silicon Syndicate"
 *
 * MILESTONE B: complete task-aware object selection on the board.
 *
 *   argv task string
 *     -> WordPiece tokenize                                        (VEGA)
 *     -> embeddings + LayerNorm                                    (VEGA)
 *     -> 6 x { Q,K,V | QK^T | softmax | A*V | O | LN | FFN1 | GELU | FFN2 | LN }
 *              ^accelerator^   ^VEGA^   ^acc^  ^acc^ ^VEGA^ ^acc^   ^VEGA^ ^acc^
 *     -> mean-pool + L2-norm                                       (VEGA)
 *     -> fuse with the 80 precomputed COCO label embeddings        (VEGA)
 *     -> head.0 -> ReLU -> head.3 -> sigmoid                       (accelerator)
 *     -> ranked list of COCO classes for this task
 *
 * The accelerator computes every GEMM, so the final ranking genuinely depends
 * on accelerator output. The task embedding is computed on the board from a
 * runtime string; only the 80 fixed label embeddings are precomputed offline,
 * which is legal because they do not depend on the runtime input.
 *
 * GEMM KERNEL: the "preloaded" kernel validated by team43_gemm2.c on 16 August
 * (2,304 invocations, EXACT, 236.9 us/invocation, 4.37x the Milestone A loop).
 * Every operand tile is uploaded to DDR once per GEMM in 256-word chunks; the
 * inner loop is five 4-byte register pokes and nothing else.
 *
 * EXPECTED OUTPUT for --task "serve wine". These are this program's own --sim
 * numbers, produced on the host from this same INT8 package, so the board must
 * reproduce them (to within libm rounding, see below):
 *      wine glass  0.970027     bottle      0.046360
 *      cup         0.933609     spoon       0.023431
 *      bowl        0.089686     knife       0.012295
 *      vase        0.079406     hair drier  0.007672
 *      task_emb[0..3] = +0.018800 -0.028369 +0.063699 -0.006796
 * The FP32 golden from export_minilm_int8.py was wine glass 0.969409 with the
 * same top-8 order, so the whole INT8 pipeline costs +0.0006 on the top score.
 *
 * Board vs host will not necessarily be bit-identical: exp/erf/sqrt may differ
 * in the last ULP between glibc builds, and a value sitting exactly on a
 * quantization boundary can round the other way. Agreement to ~1e-4 on the
 * scores and an EXACT top-8 ordering is the pass criterion.
 *
 * WORK: 259,104 invocations -- 166,464 for the 6 encoder layers and 92,640 for
 * the trained head. Handoff section 7 put the head at ~11,530; that figure is
 * 8x low (it divided the head's 47.2 M MACs by 4096 instead of 512).
 *
 * Build (dev PC):
 *   cd /home/Team43/Desktop/DVCon_2026/Application
 *   docker run --rm -t -v "$PWD":/workspace -w /workspace --user $(id -u):$(id -g) \
 *     buildroot-2004_new \
 *     bash -c 'buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc \
 *         -std=gnu99 -O2 -static -Wall -Wextra -o team43_minilm_b team43_minilm_b.c -lm -lrt'
 *
 * Run (board):
 *   /tmp/team43/team43_minilm_b --task "serve wine"
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

#include "team43_minilm_layout.h"

/* ===================================================== model geometry ==== */

#define HID        384
#define FFN        1536
#define NLAYERS    6
#define NHEADS     12
#define HDIM       32            /* HID / NHEADS */
#define MAXTOK     8
#define NCLS       80
#define FUSED      (4 * HID)     /* 1536 */

#define TOK_PAD 0
#define TOK_UNK 100
#define TOK_CLS 101
#define TOK_SEP 102

/* ===================================================== accelerator ======= */

#define K_ADDR    0xBF000000UL
#define X_ADDR    0xBF100000UL
#define BIAS_ADDR 0xBF200000UL
#define Y_ADDR    0xBF300000UL
#define WINDOW_BYTES 0x100000UL          /* 1 MB per scratch window */

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

/* largest packings needed: W for FFN1/FFN2/head0 = 192*48*64 = 589,824
 *                          X for head0 (M=80,K=1536) = 10*192*64 = 122,880 */
#define MAX_PACKW  (192 * 48 * 64)
#define MAX_PACKX  (10 * 192 * 64)

static int   g_fd    = -1;
static int   g_sim   = 0;
static int   g_quiet = 0;
static long  g_inv   = 0;
static long  g_sys   = 0;
static long  g_upload_us = 0;
static long  g_pack_us   = 0;   /* host-side gather of weights into tile order */
static long  g_dev_us    = 0;   /* the accelerator inner loop only             */
static long  g_seed_us   = 0;   /* per-block accumulator seed + harvest        */
/* Packing loop order. 'src' walks the SOURCE sequentially and scatters into the
 * tile buffer; 'dst' walks the DESTINATION sequentially and gathers from the
 * source with a stride of one matrix row (the original order).
 *
 * On the board, 'dst' cost 154.73 s -- about 13.4 us per element, roughly 30x
 * the per-element rate of a cache-resident loop. Every inner step jumped one
 * full row (384 or 1536 bytes) through a 23 MB mapping, so essentially no read
 * was ever served from cache. 'src' reads straight down the source and pays the
 * scattering on 8-byte-strided WRITES into a small buffer instead, which the
 * store path can absorb. Same output, byte for byte -- assert with --checkpack. */
static int   g_pack_src  = 1;

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

static void rw(off_t off, uint32_t v)
{
    g_sys++;
    if (pwrite(g_fd, &v, 1, off) < 0)
        die("register write 0x%02lx: %s", (unsigned long)off, strerror(errno));
}
static uint32_t rr(off_t off)
{
    uint32_t v = 0; g_sys++;
    if (pread(g_fd, &v, 1, off) < 0)
        die("register read 0x%02lx: %s", (unsigned long)off, strerror(errno));
    return v;
}
static void dw(unsigned long addr, const void *buf, int words)
{
    g_sys++;
    if (pwrite(g_fd, buf, words, (off_t)addr) < 0)
        die("DDR write 0x%lx (%d words): %s", addr, words, strerror(errno));
}
static void dr(unsigned long addr, void *buf, int words)
{
    g_sys++;
    if (pread(g_fd, buf, words, (off_t)addr) < 0)
        die("DDR read 0x%lx (%d words): %s", addr, words, strerror(errno));
}
static void dw_bulk(unsigned long addr, const void *buf, long bytes)
{
    long words = bytes / 4, done = 0;
    const uint8_t *p = (const uint8_t *)buf;
    long t0 = now_us();
    while (done < words) {
        int n = (int)((words - done) > CHUNK_WORDS ? CHUNK_WORDS : (words - done));
        dw(addr + (unsigned long)done * 4, p + done * 4, n);
        done += n;
    }
    g_upload_us += now_us() - t0;
}
static void wait_done(int m0, int n0, int k0)
{
    for (unsigned g = 0; g < POLL_LIMIT; g++)
        if (rr(REG_Y_DONE) & 1u) return;
    die("Y_DONE never asserted (m0=%d n0=%d k0=%d, invocation %ld)", m0, n0, k0, g_inv);
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

/* ===================================================== GEMM engine ======= */

static int8_t  packW[MAX_PACKW];
static int8_t  packX[MAX_PACKX];
static int32_t g_bias[FFN];

/* pack X[M,K] (INT8, row-major) into (m0,k0) tiles, column-major inside */
static void pack_x(const int8_t *x, int M, int K)
{
    long _t0 = now_us();
    int MT = (M + TILE - 1) / TILE, KT = K / TILE;
    if ((long)MT * KT * 64 > MAX_PACKX) die("packX overflow M=%d K=%d", M, K);
    if (g_pack_src) {
        memset(packX, 0, (size_t)MT * KT * 64);
        for (int r = 0; r < M; r++) {
            const int8_t *row = x + (size_t)r * K;
            int m0 = r / TILE, rr = r % TILE;
            int8_t *base = packX + (long)m0 * KT * 64 + rr;
            for (int k = 0; k < K; k++)
                base[(long)(k / TILE) * 64 + (k % TILE) * TILE] = row[k];
        }
        g_pack_us += now_us() - _t0;
        return;
    }
    for (int m0 = 0; m0 < MT; m0++)
        for (int k0 = 0; k0 < KT; k0++) {
            int8_t *t = packX + ((long)m0 * KT + k0) * 64;
            for (int kk = 0; kk < TILE; kk++)
                for (int r = 0; r < TILE; r++) {
                    int gr = m0*TILE + r;
                    t[kk*TILE + r] = (gr < M) ? x[(size_t)gr*K + k0*TILE + kk] : 0;
                }
        }
    g_pack_us += now_us() - _t0;
}
/* pack a packaged weight tensor (stored [out,in]) into (n0,k0) tiles */
static void pack_w_pkg(const tensor_t *W, int K, int N)
{
    long _t0 = now_us();
    int KT = K / TILE, NT = (N + TILE - 1) / TILE;
    if ((long)NT * KT * 64 > MAX_PACKW) die("packW overflow K=%d N=%d", K, N);
    if (g_pack_src) {
        const int8_t *p = (const int8_t *)W->p;
        int D1 = W->d1;                    /* stored [out, in]; D1 == in == K */
        memset(packW, 0, (size_t)NT * KT * 64);
        for (int c = 0; c < N; c++) {
            const int8_t *row = p + (size_t)c * D1;   /* one output channel */
            int n0 = c / TILE, cc = c % TILE;
            int8_t *base = packW + (long)n0 * KT * 64 + cc;
            for (int k = 0; k < K; k++)
                base[(long)(k / TILE) * 64 + (k % TILE) * TILE] = row[k];
        }
        g_pack_us += now_us() - _t0;
        return;
    }
    for (int n0 = 0; n0 < NT; n0++)
        for (int k0 = 0; k0 < KT; k0++) {
            int8_t *t = packW + ((long)n0 * KT + k0) * 64;
            for (int kk = 0; kk < TILE; kk++)
                for (int cc = 0; cc < TILE; cc++) {
                    int gc = n0*TILE + cc;
                    t[kk*TILE + cc] = (gc < N) ? w_i8(W, k0*TILE + kk, gc) : 0;
                }
        }
    g_pack_us += now_us() - _t0;
}
/* pack an activation-as-weight matrix given row-major [K][N] INT8 */
static void pack_w_act(const int8_t *w, int K, int N)
{
    long _t0 = now_us();
    int KT = K / TILE, NT = (N + TILE - 1) / TILE;
    if ((long)NT * KT * 64 > MAX_PACKW) die("packW overflow K=%d N=%d", K, N);
    if (g_pack_src) {
        memset(packW, 0, (size_t)NT * KT * 64);
        for (int k = 0; k < K; k++) {
            const int8_t *row = w + (size_t)k * N;
            int8_t *base = packW + (long)(k / TILE) * 64 + (k % TILE) * TILE;
            for (int c = 0; c < N; c++)
                base[(long)(c / TILE) * KT * 64 + (c % TILE)] = row[c];
        }
        g_pack_us += now_us() - _t0;
        return;
    }
    for (int n0 = 0; n0 < NT; n0++)
        for (int k0 = 0; k0 < KT; k0++) {
            int8_t *t = packW + ((long)n0 * KT + k0) * 64;
            for (int kk = 0; kk < TILE; kk++)
                for (int cc = 0; cc < TILE; cc++) {
                    int gc = n0*TILE + cc;
                    t[kk*TILE + cc] = (gc < N) ? w[(size_t)(k0*TILE+kk)*N + gc] : 0;
                }
        }
    g_pack_us += now_us() - _t0;
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

/* Y[M,N] += bias, using packX / packW already filled. out is [M][N] row-major. */
static void gemm_run(int M, int K, int N, const int32_t *bias_i32, int32_t *out)
{
    int MT = (M + TILE - 1) / TILE, KT = K / TILE, NT = (N + TILE - 1) / TILE;
    int32_t par[64];

    if (!g_sim) {
        dw_bulk(X_ADDR, packX, (long)MT * KT * 64);
        dw_bulk(K_ADDR, packW, (long)NT * KT * 64);
        rw(REG_K_BYTES, INPUT_BYTES);  rw(REG_K_TUSER, 0u);
        rw(REG_X_BYTES, INPUT_BYTES);  rw(REG_X_TUSER, 0u);
        rw(REG_B_ADDR, (uint32_t)Y_ADDR); rw(REG_B_BYTES, OUTPUT_BYTES);
        rw(REG_B_TUSER, 0u);
        rw(REG_Y_ADDR, (uint32_t)Y_ADDR); rw(REG_Y_BYTES, OUTPUT_BYTES);
    }

    for (int m0 = 0; m0 < MT; m0++)
        for (int n0 = 0; n0 < NT; n0++) {
            for (int cc = 0; cc < TILE; cc++)
                for (int r = 0; r < TILE; r++) {
                    int gc = n0*TILE + cc;
                    par[cc*TILE + r] = (gc < N) ? bias_i32[gc] : 0;
                }
            long _ts = now_us();
            if (!g_sim) dw(Y_ADDR, par, OUTPUT_WORDS);
            g_seed_us += now_us() - _ts;

            long _td = now_us();
            for (int k0 = 0; k0 < KT; k0++) {
                if (g_sim) {
                    sim_tile(packX + ((long)m0*KT + k0)*64,
                             packW + ((long)n0*KT + k0)*64, par);
                } else {
                    rw(REG_START, 0u);
                    rw(REG_K_ADDR, (uint32_t)(K_ADDR + ((unsigned long)n0*KT + k0)*64));
                    rw(REG_X_ADDR, (uint32_t)(X_ADDR + ((unsigned long)m0*KT + k0)*64));
                    rw(REG_START, 1u);
                    wait_done(m0, n0, k0);
                }
                g_inv++;
            }
            g_dev_us += now_us() - _td;
            _ts = now_us();
            if (!g_sim) dr(Y_ADDR, par, OUTPUT_WORDS);
            g_seed_us += now_us() - _ts;

            for (int cc = 0; cc < TILE; cc++)
                for (int r = 0; r < TILE; r++) {
                    int gr = m0*TILE + r, gc = n0*TILE + cc;
                    if (gr < M && gc < N) out[(size_t)gr*N + gc] = par[cc*TILE + r];
                }
        }
}

/* --------- convenience wrappers: quantize, run, dequantize -------------- */

static int8_t  qbuf[NCLS * FUSED];      /* biggest activation: 80 x 1536      */
static int32_t obuf[NCLS * HID];        /* biggest output:     80 x 384       */

static double quantize(const double *x, long n, int8_t *out)
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

/* y[M,N] = x[M,K] * W_pkg + bias_fp   (W from the package, [out,in]) */
static void dense(const double *x, int M, int K, int N,
                  const tensor_t *W, const tensor_t *B, double *y)
{
    double sx = quantize(x, (long)M*K, qbuf);
    double sw = W->scale, comb = sx * sw;
    for (int c = 0; c < N; c++) g_bias[c] = (int32_t)lround(t_get1(B, c) / comb);
    pack_x(qbuf, M, K);
    pack_w_pkg(W, K, N);
    gemm_run(M, K, N, g_bias, obuf);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            y[(size_t)r*N + c] = (double)obuf[(size_t)r*N + c] * comb;
}

/* y[M,N] = a[M,K] * b[K,N], both activations, no bias */
static void matmul_act(const double *a, int M, int K, int N,
                       const double *b, double *y)
{
    static int8_t bq[FFN * TILE];
    double sa = quantize(a, (long)M*K, qbuf);
    pack_x(qbuf, M, K);
    if ((long)K*N > (long)sizeof(bq)) die("matmul_act operand too large");
    double sb = quantize(b, (long)K*N, bq);
    pack_w_act(bq, K, N);
    memset(g_bias, 0, sizeof(int32_t) * (size_t)N);
    gemm_run(M, K, N, g_bias, obuf);
    for (int r = 0; r < M; r++)
        for (int c = 0; c < N; c++)
            y[(size_t)r*N + c] = (double)obuf[(size_t)r*N + c] * sa * sb;
}

/* Pack the same real weight tensor both ways and require byte-identical output.
 * Cheap insurance that the fast loop order is a pure reordering, not a rewrite
 * of the arithmetic. */
static int8_t packW_ref[MAX_PACKW];
static void checkpack(void)
{
    struct { const char *tag; tensor_t t; int K, N; } cases[] = {
        { "L0.q.w  [384,384]",  TEN(T_L0_Q_W),  HID, HID },
        { "L0.f1.w [384,1536]", TEN(T_L0_F1_W), HID, FFN },
        { "L0.f2.w [1536,384]", TEN(T_L0_F2_W), FFN, HID },
        { "head0.w [1536,384]", TEN(T_HEAD0_W), FFN, HID },
    };
    int bad = 0;
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        long bytes = (long)((cases[i].N + TILE - 1)/TILE) * (cases[i].K/TILE) * 64;
        g_pack_src = 0; pack_w_pkg(&cases[i].t, cases[i].K, cases[i].N);
        memcpy(packW_ref, packW, (size_t)bytes);
        g_pack_src = 1; pack_w_pkg(&cases[i].t, cases[i].K, cases[i].N);
        int same = memcmp(packW_ref, packW, (size_t)bytes) == 0;
        printf("  checkpack %-22s %ld bytes : %s\n",
               cases[i].tag, bytes, same ? "identical" : "*** DIFFERENT ***");
        if (!same) bad = 1;
    }
    g_pack_us = 0;
    if (bad) die("the two packing orders disagree -- do not use --pack src");
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

/* ===================================================== main ============== */

static double X[MAXTOK*HID], Q[MAXTOK*HID], K_[MAXTOK*HID], V[MAXTOK*HID];
static double CTX[MAXTOK*HID], TMP[MAXTOK*HID], H[MAXTOK*FFN];
static double qh[MAXTOK*HDIM], kh[HDIM*MAXTOK], vh[MAXTOK*HDIM];
static double sc[MAXTOK*MAXTOK], ctxh[MAXTOK*HDIM];
static double fused[NCLS*FUSED], hid[NCLS*HID], logit[NCLS*8];
static double temb[HID], oemb[NCLS][HID], score[NCLS];

static void usage(const char *p)
{
    fprintf(stderr,
      "usage: %s --task \"<prompt>\" [options]\n"
      "  --model PATH  default /tmp/team43/team43_minilm_int8.bin\n"
      "  --vocab PATH  default /tmp/team43/vocab.txt\n"
      "  --dev   PATH  default /dev/accelerator_test\n"
      "  --sim         emulate the accelerator in software (no device)\n"
      "  --pack src|dst  packing loop order (default src; dst is the old one)\n"
      "  --checkpack   assert both packing orders give identical bytes, then run\n"
      "  --topk N      how many classes to print (default 8)\n"
      "  --trace       print intermediate values after every layer\n"
      "  --quiet\n", p);
    exit(1);
}

int main(int argc, char **argv)
{
    const char *task = NULL;
    const char *mpath = "/tmp/team43/team43_minilm_int8.bin";
    const char *vpath = "/tmp/team43/vocab.txt";
    const char *dpath = "/dev/accelerator_test";
    int topk = 8, trace = 0, docheck = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--task") && i+1<argc) task = argv[++i];
        else if (!strcmp(argv[i],"--model") && i+1<argc) mpath = argv[++i];
        else if (!strcmp(argv[i],"--vocab") && i+1<argc) vpath = argv[++i];
        else if (!strcmp(argv[i],"--dev")   && i+1<argc) dpath = argv[++i];
        else if (!strcmp(argv[i],"--topk")  && i+1<argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--sim")) g_sim = 1;
        else if (!strcmp(argv[i],"--pack") && i+1<argc) {
            const char *m = argv[++i];
            if (!strcmp(m,"src")) g_pack_src = 1;
            else if (!strcmp(m,"dst")) g_pack_src = 0;
            else usage(argv[0]);
        }
        else if (!strcmp(argv[i],"--trace")) trace = 1;
        else if (!strcmp(argv[i],"--checkpack")) docheck = 1;
        else if (!strcmp(argv[i],"--quiet")) g_quiet = 1;
        else usage(argv[0]);
    }
    if (!task) usage(argv[0]);

    long t_start = now_us();

    int mfd = open(mpath, O_RDONLY);
    if (mfd < 0) die("cannot open model '%s': %s", mpath, strerror(errno));
    struct stat st;
    if (fstat(mfd, &st) != 0) die("fstat: %s", strerror(errno));
    void *mp = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mfd, 0);
    if (mp == MAP_FAILED) die("mmap: %s", strerror(errno));
    MODEL.base = (const unsigned char *)mp; MODEL.size = (size_t)st.st_size;
    if (MODEL.size != (size_t)MINILM_BLOB_BYTES)
        die("package is %lu bytes, header says %ld",
            (unsigned long)MODEL.size, (long)MINILM_BLOB_BYTES);
    resolve_tensors();
    vocab_load(vpath);
    if (docheck) { int keep = g_pack_src; checkpack(); g_pack_src = keep; }

    printf("=== TEAM43 MILESTONE B -- task-aware object selection ===\n");
    printf("task  : \"%s\"\n", task);
    printf("mode  : %s\n", g_sim ? "SOFTWARE EMULATION (no device)" : "accelerator");

    if (!g_sim) {
        g_fd = open(dpath, O_RDWR);
        if (g_fd < 0) die("cannot open '%s': %s", dpath, strerror(errno));
    }

    /* ---- tokenize the runtime prompt ---------------------------------- */
    int ids[MAXTOK], mask[MAXTOK];
    tokenize_task(task, ids, mask);
    printf("tokens: ");
    for (int i = 0; i < MAXTOK; i++) printf("%s%s", i?" ":"", VOC.tok[ids[i]]);
    printf("\nids   : [");
    for (int i = 0; i < MAXTOK; i++) printf("%s%d", i?", ":"", ids[i]);
    printf("]\n");

    /* ---- embeddings + LayerNorm --------------------------------------- */
    tensor_t we = TEN(T_WORD_EMB), pe = TEN(T_POS_EMB), te = TEN(T_TOK_TYPE);
    tensor_t lnw = TEN(T_ENCODER_EMBEDDINGS_LAYERNORM_WEIGHT);
    tensor_t lnb = TEN(T_ENCODER_EMBEDDINGS_LAYERNORM_BIAS);
    for (int t = 0; t < MAXTOK; t++)
        for (int h = 0; h < HID; h++)
            X[t*HID+h] = t_get(&we, ids[t], h) + t_get(&pe, t, h) + t_get(&te, 0, h);
    layernorm(X, MAXTOK, HID, &lnw, &lnb);

    /* ---- 6 encoder layers --------------------------------------------- */
    long t_enc = now_us();
    for (int L = 0; L < NLAYERS; L++) {
        dense(X, MAXTOK, HID, HID, &LW[L].q, &LB[L].qb, Q);
        dense(X, MAXTOK, HID, HID, &LW[L].k, &LB[L].kb, K_);
        dense(X, MAXTOK, HID, HID, &LW[L].v, &LB[L].vb, V);

        for (int h = 0; h < NHEADS; h++) {
            for (int t = 0; t < MAXTOK; t++)
                for (int d = 0; d < HDIM; d++) {
                    qh[t*HDIM+d] = Q[t*HID + h*HDIM + d];
                    vh[t*HDIM+d] = V[t*HID + h*HDIM + d];
                    kh[d*MAXTOK+t] = K_[t*HID + h*HDIM + d];   /* K^T [HDIM, T] */
                }
            matmul_act(qh, MAXTOK, HDIM, MAXTOK, kh, sc);      /* [T,T] */

            for (int r = 0; r < MAXTOK; r++) {
                double mx = -1e300;
                for (int c = 0; c < MAXTOK; c++) {
                    sc[r*MAXTOK+c] = sc[r*MAXTOK+c] / sqrt((double)HDIM)
                                   + (mask[c] ? 0.0 : -10000.0);
                    if (sc[r*MAXTOK+c] > mx) mx = sc[r*MAXTOK+c];
                }
                double sum = 0.0;
                for (int c = 0; c < MAXTOK; c++) {
                    sc[r*MAXTOK+c] = exp(sc[r*MAXTOK+c] - mx);
                    sum += sc[r*MAXTOK+c];
                }
                for (int c = 0; c < MAXTOK; c++) sc[r*MAXTOK+c] /= sum;
            }
            matmul_act(sc, MAXTOK, MAXTOK, HDIM, vh, ctxh);    /* [T,HDIM] */
            for (int t = 0; t < MAXTOK; t++)
                for (int d = 0; d < HDIM; d++)
                    CTX[t*HID + h*HDIM + d] = ctxh[t*HDIM+d];
        }

        dense(CTX, MAXTOK, HID, HID, &LW[L].o, &LB[L].ob, TMP);
        for (long i = 0; i < MAXTOK*HID; i++) TMP[i] += X[i];
        layernorm(TMP, MAXTOK, HID, &LB[L].ln1w, &LB[L].ln1b);
        memcpy(X, TMP, sizeof(X));

        dense(X, MAXTOK, HID, FFN, &LW[L].f1, &LB[L].f1b, H);
        for (long i = 0; i < MAXTOK*FFN; i++) H[i] = gelu(H[i]);
        dense(H, MAXTOK, FFN, HID, &LW[L].f2, &LB[L].f2b, TMP);
        for (long i = 0; i < MAXTOK*HID; i++) TMP[i] += X[i];
        layernorm(TMP, MAXTOK, HID, &LB[L].ln2w, &LB[L].ln2b);
        memcpy(X, TMP, sizeof(X));

        if (trace)
            printf("  [layer %d] x[0][0..3] = %+.6f %+.6f %+.6f %+.6f\n",
                   L, X[0], X[1], X[2], X[3]);
        else if (!g_quiet)
            printf("  layer %d/%d done (%ld invocations so far)\n", L+1, NLAYERS, g_inv);
        fflush(stdout);
    }
    long enc_us = now_us() - t_enc;

    /* ---- mean-pool over the mask, then L2 normalise -------------------- */
    double cnt = 0.0;
    for (int h = 0; h < HID; h++) temb[h] = 0.0;
    for (int t = 0; t < MAXTOK; t++) if (mask[t]) {
        cnt += 1.0;
        for (int h = 0; h < HID; h++) temb[h] += X[t*HID+h];
    }
    if (cnt < 1e-9) cnt = 1e-9;
    double nrm = 0.0;
    for (int h = 0; h < HID; h++) { temb[h] /= cnt; nrm += temb[h]*temb[h]; }
    nrm = sqrt(nrm); if (nrm < 1e-9) nrm = 1e-9;
    for (int h = 0; h < HID; h++) temb[h] /= nrm;
    printf("task_emb[0..3] : %+.6f %+.6f %+.6f %+.6f\n",
           temb[0], temb[1], temb[2], temb[3]);

    /* ---- fuse with the 80 precomputed label embeddings ----------------- */
    tensor_t oe = TEN(T_OBJ_EMB);
    for (int c = 0; c < NCLS; c++)
        for (int h = 0; h < HID; h++) oemb[c][h] = t_get(&oe, c, h);
    for (int c = 0; c < NCLS; c++)
        for (int h = 0; h < HID; h++) {
            fused[(size_t)c*FUSED + h]           = temb[h];
            fused[(size_t)c*FUSED + HID + h]     = oemb[c][h];
            fused[(size_t)c*FUSED + 2*HID + h]   = fabs(temb[h] - oemb[c][h]);
            fused[(size_t)c*FUSED + 3*HID + h]   = temb[h] * oemb[c][h];
        }

    /* ---- trained head on the accelerator ------------------------------ */
    tensor_t h0w = TEN(T_HEAD0_W), h0b = TEN(T_HEAD_0_BIAS);
    tensor_t h3w = TEN(T_HEAD3_W), h3b = TEN(T_HEAD_3_BIAS);
    dense(fused, NCLS, FUSED, HID, &h0w, &h0b, hid);
    for (long i = 0; i < NCLS*HID; i++) if (hid[i] < 0.0) hid[i] = 0.0;
    dense(hid, NCLS, HID, 1, &h3w, &h3b, logit);
    for (int c = 0; c < NCLS; c++) score[c] = 1.0 / (1.0 + exp(-logit[c]));

    /* ---- rank ---------------------------------------------------------- */
    int order[NCLS];
    for (int i = 0; i < NCLS; i++) order[i] = i;
    for (int i = 1; i < NCLS; i++) {
        int key = order[i], j = i - 1;
        while (j >= 0 && score[order[j]] < score[key]) { order[j+1] = order[j]; j--; }
        order[j+1] = key;
    }

    long total_us = now_us() - t_start;
    printf("\n--- RANKED OBJECTS FOR TASK \"%s\" ---\n", task);
    if (topk > NCLS) topk = NCLS;
    for (int i = 0; i < topk; i++)
        printf("  %2d. %-16s %.6f\n", i+1, COCO_NAMES[order[i]], score[order[i]]);

    printf("\n--- WORK ---\n");
    printf("  accelerator invocations : %ld\n", g_inv);
    printf("  syscalls                : %ld\n", g_sys);
    printf("  encoder time            : %.2f s\n", enc_us / 1e6);
    printf("  operand upload time     : %.2f s\n", g_upload_us / 1e6);
    printf("  weight packing (VEGA)   : %.2f s\n", g_pack_us / 1e6);
    printf("  accelerator inner loop  : %.2f s   (%.1f us/invocation)\n",
           g_dev_us / 1e6, g_inv ? (double)g_dev_us / (double)g_inv : 0.0);
    printf("  accumulator seed/harvest: %.2f s\n", g_seed_us / 1e6);
    printf("  everything else (VEGA)  : %.2f s   (LayerNorm, GELU, softmax,\n"
           "                                       quantize, dequantize, fuse)\n",
           (total_us - g_upload_us - g_pack_us - g_dev_us - g_seed_us) / 1e6);
    printf("  total wall time         : %.2f s\n", total_us / 1e6);
    if (g_inv) printf("  per invocation          : %.1f us\n",
                      (double)(total_us) / (double)g_inv);
    printf("\n  SELECTED OBJECT: %s\n", COCO_NAMES[order[0]]);

    if (g_fd >= 0) close(g_fd);
    return 0;
}
