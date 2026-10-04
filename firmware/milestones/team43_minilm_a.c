/*
 * team43_minilm_a.c  --  DVCon India 2026, Stage 3B, Team 43 "Silicon Syndicate"
 *
 * MILESTONE A: prove the on-board plumbing end to end, on real runtime input,
 * before building the full 6-layer encoder.
 *
 * What this program does, in order:
 *   1. mmap  the INT8 model package (team43_minilm_int8.bin)
 *   2. load  vocab.txt and WordPiece-tokenize the RUNTIME prompt from argv
 *   3. build the embedding row (word + position + token-type) and LayerNorm it
 *      -> X[8,384] FP32, then per-tensor symmetric INT8 quantization
 *   4. run  ONE GEMM on the accelerator: layer-0 Q projection,
 *      X[8,384] * Wq[384,384] + bq   =  2,304 physical 8x8x8 invocations
 *   5. compare the accelerator result against an EXACT software INT32 GEMM
 *      (must be 3072/3072) and against an FP32 reference (quantization error)
 *   6. report timing in two register modes so the "94% software overhead"
 *      claim in the handoff becomes a MEASURED number:
 *         --mode legacy : 11 constant register writes re-issued per tile
 *         --mode hoist  : those 11 writes done ONCE at program start
 *
 * STANDALONE: the task string is read at runtime from argv. Nothing about a
 * particular prompt is baked into this binary.
 *
 * Build (dev PC, CentOS 7):
 *   cd /home/Team43/Desktop/DVCon_2026/Application
 *   TOOL=$PWD/buildroot_dvcon/buildroot/output/host/bin
 *   $TOOL/riscv64-buildroot-linux-gnu-gcc -std=gnu99 -O2 -static -Wall -Wextra \
 *        -o team43_minilm_a team43_minilm_a.c -lm -lrt
 *
 * Run (board):
 *   /tmp/team43/team43_minilm_a --task "serve wine" --mode both
 *
 * ===========================================================================
 * THINGS THAT MUST BE CONFIRMED AGAINST YOUR ACTUAL FILES BEFORE TRUSTING THIS
 * ===========================================================================
 *  (1) The layout-header macro names in the "NAME MAP" block below. They follow
 *      section 6 of the handoff. If the compiler says a macro is undefined,
 *      fix it in that block ONLY -- nothing else in this file names a macro.
 *  (2) The TUSER register values. The handoff documents the register OFFSETS
 *      but not the tuser payloads. Copy them from Application/drv_test_128.c
 *      into T43_TUSER_K / _X / _B below, or pass --tuser K,X,B at runtime.
 *  (3) T43_W_LAYOUT_IN_OUT: whether the exporter stored the Q weight as
 *      [in,out] (1, default) or [out,in] (0). Milestone A's pass/fail does not
 *      depend on this (software and accelerator use the same accessor), but
 *      Milestone B's numeric agreement with the host golden does.
 * ===========================================================================
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
#include <sys/types.h>

/* ======================================================================== */
/* NAME MAP -- the ONLY place this file refers to team43_minilm_layout.h     */
/* ======================================================================== */

#include "team43_minilm_layout.h"

/* Each of these must resolve to a macro family <NAME>_OFF, _D0, _D1,
 * _SCALE, _IS_I8 in the generated header. Rename here if yours differ. */
#ifndef T43_WORD_EMB
#define T43_WORD_EMB   T_WORD_EMB          /* INT8  [vocab, 384]           */
#endif
#ifndef T43_POS_EMB
#define T43_POS_EMB    T_POS_EMB           /* INT8  [8, 384] (sliced 0..7) */
#endif
#ifndef T43_TOK_TYPE
#define T43_TOK_TYPE   T_TOK_TYPE          /* INT8  [2, 384]               */
#endif
#ifndef T43_EMB_LN_W
#define T43_EMB_LN_W   T_ENCODER_EMBEDDINGS_LAYERNORM_WEIGHT  /* FP32 [384] */
#endif
#ifndef T43_EMB_LN_B
#define T43_EMB_LN_B   T_ENCODER_EMBEDDINGS_LAYERNORM_BIAS    /* FP32 [384] */
#endif
#ifndef T43_L0_Q_W
#define T43_L0_Q_W     T_L0_Q_W            /* INT8  [384, 384]             */
#endif
#ifndef T43_L0_Q_B
#define T43_L0_Q_B     T_ENCODER_ENCODER_LAYER_0_ATTENTION_SELF_QUERY_BIAS
#endif

/* Weight storage order: 1 = [in,out] (index k*D1+c), 0 = [out,in] (c*D1+k).
 * CONFIRMED 0 for this package: export_minilm_int8.py stores raw torch
 * `.weight` tensors and its forward uses `x @ W.T`, i.e. [out, in]. The
 * manifest proves it -- L0.f1.w is [1536,384] for a 384->1536 layer, and
 * head0.w is [384,1536] for a 1536->384 layer. */
#ifndef T43_W_LAYOUT_IN_OUT
#define T43_W_LAYOUT_IN_OUT 0
#endif

/* ======================================================================== */
/* ACCELERATOR INTERFACE -- from handoff section 4 (Application/drv_test_128.c) */
/* ======================================================================== */

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
#define REG_DONE      0x34
#define REG_Y_ADDR    0x38
#define REG_Y_BYTES   0x3C

#define DDR_K   0xBF000000UL
#define DDR_X   0xBF100000UL
#define DDR_B   0xBF200000UL
#define DDR_Y   0xBF300000UL

/* pread/pwrite counts are in 32-BIT WORDS, not bytes. */
#define WORDS_I8_TILE   16      /* 8x8 INT8  = 64 bytes  */
#define WORDS_I32_TILE  64      /* 8x8 INT32 = 256 bytes */
#define BYTES_I8_TILE   64
#define BYTES_I32_TILE  256

/* CONFIRM THESE AGAINST drv_test_128.c, or override with --tuser K,X,B */
#ifndef T43_TUSER_K
#define T43_TUSER_K 0u
#endif
#ifndef T43_TUSER_X
#define T43_TUSER_X 0u
#endif
#ifndef T43_TUSER_B
#define T43_TUSER_B 0u
#endif

#define DONE_TIMEOUT_US 2000000L   /* 2 s per tile, then we call it a failure */

/* ======================================================================== */
/* MODEL GEOMETRY                                                           */
/* ======================================================================== */

#define HID         384
#define MAX_TOKENS  8       /* deployed reduction from inference.py's 32 */
#define TILE        8

#define TOK_PAD   0
#define TOK_UNK   100
#define TOK_CLS   101
#define TOK_SEP   102

/* ======================================================================== */
/* SMALL UTILITIES                                                          */
/* ======================================================================== */

static int g_quiet = 0;

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf,1,2)));
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

static long now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n, sz);
    if (!p) die("out of memory (%lu x %lu)", (unsigned long)n, (unsigned long)sz);
    return p;
}

/* ======================================================================== */
/* MODEL FILE + TENSOR VIEWS                                                */
/* ======================================================================== */

typedef struct {
    const unsigned char *base;
    size_t               size;
} model_t;

typedef struct {
    const unsigned char *p;       /* pointer to first element        */
    long                 off;     /* byte offset inside the package  */
    int                  d0, d1;
    double               scale;   /* dequant scale (1.0 for FP32)    */
    int                  is_i8;
    const char          *name;
} tensor_t;

static tensor_t make_tensor(const model_t *m, const char *name,
                            long off, int d0, int d1, double scale, int is_i8)
{
    tensor_t t;
    size_t elems = (size_t)d0 * (size_t)d1;
    size_t need  = elems * (is_i8 ? 1u : 4u);

    if (off < 0 || (size_t)off + need > m->size)
        die("tensor %s runs past end of model file "
            "(off=%ld need=%lu size=%lu)",
            name, off, (unsigned long)need, (unsigned long)m->size);

    t.p     = m->base + off;
    t.off   = off;
    t.d0    = d0;
    t.d1    = d1;
    t.scale = scale;
    t.is_i8 = is_i8;
    t.name  = name;
    return t;
}

#define T43_CAT_(a,b) a##b
#define T43_CAT(a,b)  T43_CAT_(a,b)

/* TEN(model, T43_L0_Q_W) -> make_tensor(model, "T43_L0_Q_W",
 *                              T_L0_Q_W_OFF, T_L0_Q_W_D0, ... ) */
#define TEN(m, N) make_tensor((m), #N,                       \
                              (long)T43_CAT(N,_OFF),         \
                              (int) T43_CAT(N,_D0),          \
                              (int) T43_CAT(N,_D1),          \
                              (double)T43_CAT(N,_SCALE),     \
                              (int) T43_CAT(N,_IS_I8))

/* dequantized element access, row-major [d0,d1] */
static double t_get(const tensor_t *t, int r, int c)
{
    size_t idx = (size_t)r * (size_t)t->d1 + (size_t)c;
    if (t->is_i8) return (double)((const int8_t *)t->p)[idx] * t->scale;
    return (double)((const float *)t->p)[idx];
}

/* 1-D vector access, tolerating [n,1], [1,n] or [n] shapes */
static int t_len1d(const tensor_t *t)
{
    if (t->d0 == 1) return t->d1;
    if (t->d1 == 1) return t->d0;
    return t->d0 * t->d1;
}
static double t_get1(const tensor_t *t, int i)
{
    if (t->is_i8) return (double)((const int8_t *)t->p)[i] * t->scale;
    return (double)((const float *)t->p)[i];
}

/* raw INT8 weight access with the layout switch applied:
 * w_i8(t, k, c) is the element multiplying input k to produce output c. */
static int8_t w_i8(const tensor_t *t, int k, int c)
{
    const int8_t *w = (const int8_t *)t->p;
#if T43_W_LAYOUT_IN_OUT
    return w[(size_t)k * (size_t)t->d1 + (size_t)c];
#else
    return w[(size_t)c * (size_t)t->d1 + (size_t)k];
#endif
}

/* ======================================================================== */
/* VOCABULARY (30522 lines) + HASH TABLE                                     */
/* ======================================================================== */

#define VOCAB_HASH_BITS 16
#define VOCAB_HASH_SIZE (1u << VOCAB_HASH_BITS)

typedef struct {
    char  *blob;         /* whole file, newlines replaced by NUL */
    size_t blob_len;
    char **tok;          /* tok[id] -> string                    */
    int    n;
    int32_t slot[VOCAB_HASH_SIZE];   /* -1 empty, else token id  */
} vocab_t;

static uint32_t fnv1a(const char *s, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static void vocab_load(vocab_t *v, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open vocab '%s': %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) die("seek failed on vocab");
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) die("vocab '%s' is empty", path);

    v->blob = (char *)xcalloc((size_t)sz + 2, 1);
    if (fread(v->blob, 1, (size_t)sz, f) != (size_t)sz)
        die("short read on vocab '%s'", path);
    fclose(f);
    v->blob_len = (size_t)sz;

    /* first pass: count lines */
    int lines = 0;
    for (long i = 0; i < sz; i++) if (v->blob[i] == '\n') lines++;
    if (sz > 0 && v->blob[sz - 1] != '\n') lines++;

    v->tok = (char **)xcalloc((size_t)lines, sizeof(char *));
    for (unsigned i = 0; i < VOCAB_HASH_SIZE; i++) v->slot[i] = -1;

    int id = 0;
    char *p = v->blob;
    char *end = v->blob + sz;
    while (p < end) {
        char *nl = memchr(p, '\n', (size_t)(end - p));
        char *lineend = nl ? nl : end;
        if (lineend > p && *(lineend - 1) == '\r') lineend--;
        *lineend = '\0';
        v->tok[id] = p;

        uint32_t h = fnv1a(p, strlen(p)) & (VOCAB_HASH_SIZE - 1);
        while (v->slot[h] != -1) h = (h + 1) & (VOCAB_HASH_SIZE - 1);
        v->slot[h] = id;

        id++;
        p = nl ? nl + 1 : end;
    }
    v->n = id;

    if (v->n < 1000) die("vocab '%s' only has %d entries", path, v->n);
    if (!g_quiet) printf("[vocab] %s : %d tokens\n", path, v->n);
}

static int vocab_lookup(const vocab_t *v, const char *s, size_t n)
{
    uint32_t h = fnv1a(s, n) & (VOCAB_HASH_SIZE - 1);
    while (v->slot[h] != -1) {
        int id = v->slot[h];
        const char *t = v->tok[id];
        if (strlen(t) == n && memcmp(t, s, n) == 0) return id;
        h = (h + 1) & (VOCAB_HASH_SIZE - 1);
    }
    return -1;
}

/* ======================================================================== */
/* TOKENIZER: BERT basic tokenizer + greedy longest-match WordPiece          */
/* ======================================================================== */

static int is_ws(int c)   { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
static int is_punct(int c)
{
    if (c >= '0' && c <= '9') return 0;
    if (c >= 'a' && c <= 'z') return 0;
    if (c >= 'A' && c <= 'Z') return 0;
    if (c >= 33 && c <= 126) return 1;   /* all remaining ASCII graphics */
    return 0;
}
static int lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

#define MAX_WORDS      64
#define MAX_WORDLEN    128

/* Split raw text into whitespace/punctuation-delimited, lowercased words. */
static int basic_tokenize(const char *text, char words[MAX_WORDS][MAX_WORDLEN])
{
    int nw = 0, wl = 0;
    for (const unsigned char *p = (const unsigned char *)text; ; p++) {
        int c = *p;
        int flush = (c == 0) || is_ws(c) || is_punct(c);
        if (flush) {
            if (wl > 0) {
                if (nw >= MAX_WORDS) die("prompt has too many words (max %d)", MAX_WORDS);
                words[nw][wl] = '\0';
                nw++; wl = 0;
            }
            if (c == 0) break;
            if (is_punct(c)) {
                if (nw >= MAX_WORDS) die("prompt has too many words (max %d)", MAX_WORDS);
                words[nw][0] = (char)c; words[nw][1] = '\0'; nw++;
            }
            continue;
        }
        if (c >= 0x80) {
            /* Non-ASCII: we do not do NFD accent stripping on the board.
             * Emit the byte; WordPiece will resolve it to [UNK]. */
        }
        if (wl >= MAX_WORDLEN - 1) die("word longer than %d chars", MAX_WORDLEN - 1);
        words[nw][wl++] = (char)lower(c);
    }
    return nw;
}

/* Greedy longest-match-first WordPiece. Returns number of ids appended. */
static int wordpiece(const vocab_t *v, const char *word, int *out, int max_out)
{
    size_t len = strlen(word);
    int n = 0;
    if (len == 0) return 0;
    if (len > 100) {                      /* max_input_chars_per_word */
        if (n < max_out) out[n++] = TOK_UNK;
        return n;
    }
    size_t start = 0;
    int local[MAX_WORDLEN];
    int nl = 0;
    while (start < len) {
        size_t end = len;
        int found = -1;
        size_t found_end = 0;
        while (end > start) {
            char buf[MAX_WORDLEN + 2];
            size_t sublen = end - start;
            size_t k = 0;
            if (start > 0) { buf[k++] = '#'; buf[k++] = '#'; }
            memcpy(buf + k, word + start, sublen);
            k += sublen;
            buf[k] = '\0';
            found = vocab_lookup(v, buf, k);
            if (found >= 0) { found_end = end; break; }
            end--;
        }
        if (found < 0) {                  /* unmatchable -> whole word is UNK */
            if (n < max_out) out[n++] = TOK_UNK;
            return n;
        }
        if (nl >= MAX_WORDLEN) die("wordpiece explosion on '%s'", word);
        local[nl++] = found;
        start = found_end;
    }
    for (int i = 0; i < nl && n < max_out; i++) out[n++] = local[i];
    return n;
}

/* Build "task: <prompt>" -> ids[MAX_TOKENS], mask[MAX_TOKENS] */
static void tokenize_task(const vocab_t *v, const char *task,
                          int *ids, int *mask, int *n_real)
{
    char text[1024];
    int len = snprintf(text, sizeof(text), "task: %s", task);
    if (len < 0 || (size_t)len >= sizeof(text)) die("task string too long");

    char words[MAX_WORDS][MAX_WORDLEN];
    int nw = basic_tokenize(text, words);

    int body[MAX_TOKENS * 4];
    int nb = 0;
    for (int i = 0; i < nw; i++) {
        int got[MAX_WORDLEN];
        int g = wordpiece(v, words[i], got, MAX_WORDLEN);
        for (int j = 0; j < g; j++) {
            if (nb < (int)(sizeof(body) / sizeof(body[0]))) body[nb++] = got[j];
        }
    }

    int cap = MAX_TOKENS - 2;             /* room for [CLS] and [SEP] */
    int truncated = 0;
    if (nb > cap) { nb = cap; truncated = 1; }

    int k = 0;
    ids[k++] = TOK_CLS;
    for (int i = 0; i < nb; i++) ids[k++] = body[i];
    ids[k++] = TOK_SEP;
    *n_real = k;
    for (int i = k; i < MAX_TOKENS; i++) ids[i] = TOK_PAD;
    for (int i = 0; i < MAX_TOKENS; i++) mask[i] = (i < k) ? 1 : 0;

    if (truncated)
        fprintf(stderr, "[warn] prompt truncated to %d content tokens "
                        "(MAX_TOKENS=%d)\n", cap, MAX_TOKENS);
}

/* ======================================================================== */
/* EMBEDDINGS + LAYERNORM + QUANTIZATION                                     */
/* ======================================================================== */

static void build_embeddings(const model_t *m, const int *ids,
                             double x[MAX_TOKENS][HID])
{
    tensor_t we = TEN(m, T43_WORD_EMB);
    tensor_t pe = TEN(m, T43_POS_EMB);
    tensor_t te = TEN(m, T43_TOK_TYPE);
    tensor_t gw = TEN(m, T43_EMB_LN_W);
    tensor_t gb = TEN(m, T43_EMB_LN_B);

    if (we.d1 != HID) die("%s has d1=%d, expected %d", we.name, we.d1, HID);
    if (pe.d0 < MAX_TOKENS)
        die("%s has only %d positions, need %d", pe.name, pe.d0, MAX_TOKENS);
    if (t_len1d(&gw) != HID || t_len1d(&gb) != HID)
        die("embedding LayerNorm params are not length %d", HID);

    for (int t = 0; t < MAX_TOKENS; t++) {
        int id = ids[t];
        if (id < 0 || id >= we.d0) die("token id %d out of range", id);

        for (int h = 0; h < HID; h++)
            x[t][h] = t_get(&we, id, h) + t_get(&pe, t, h) + t_get(&te, 0, h);

        /* LayerNorm over the hidden dimension, eps = 1e-12 (BERT default) */
        double mean = 0.0;
        for (int h = 0; h < HID; h++) mean += x[t][h];
        mean /= (double)HID;
        double var = 0.0;
        for (int h = 0; h < HID; h++) { double d = x[t][h] - mean; var += d * d; }
        var /= (double)HID;
        double inv = 1.0 / sqrt(var + 1e-12);
        for (int h = 0; h < HID; h++)
            x[t][h] = (x[t][h] - mean) * inv * t_get1(&gw, h) + t_get1(&gb, h);
    }
}

static double quantize_x(const double x[MAX_TOKENS][HID], int8_t xq[MAX_TOKENS][HID])
{
    double amax = 0.0;
    for (int t = 0; t < MAX_TOKENS; t++)
        for (int h = 0; h < HID; h++) {
            double a = fabs(x[t][h]);
            if (a > amax) amax = a;
        }
    double scale = (amax > 0.0) ? amax / 127.0 : 1.0;
    for (int t = 0; t < MAX_TOKENS; t++)
        for (int h = 0; h < HID; h++) {
            long q = lround(x[t][h] / scale);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            xq[t][h] = (int8_t)q;
        }
    return scale;
}

/* ======================================================================== */
/* ACCELERATOR DRIVER                                                        */
/* ======================================================================== */

typedef struct {
    int      fd;
    int      sim;              /* 1 = emulate the accelerator in software */
    uint32_t tuser_k, tuser_x, tuser_b;
    long     syscalls;
    long     invocations;
    long     us_cmd_to_done;   /* START write -> DONE observed */
    long     us_total;         /* whole per-tile sequence      */
    long     done_polls;
} accel_t;

static void reg_write(accel_t *a, off_t off, uint32_t val)
{
    ssize_t r = pwrite(a->fd, &val, 1, off);
    a->syscalls++;
    /* drv_test_128.c checks only for a negative return, so we do the same:
     * the organizer driver's success return value is not contractually the
     * word count. */
    if (r < 0) die("register write 0x%02lx failed: %s",
                   (unsigned long)off, strerror(errno));
}

static uint32_t reg_read(accel_t *a, off_t off)
{
    uint32_t val = 0;
    ssize_t r = pread(a->fd, &val, 1, off);
    a->syscalls++;
    if (r < 0) die("register read 0x%02lx failed: %s",
                   (unsigned long)off, strerror(errno));
    return val;
}

static void ddr_write(accel_t *a, unsigned long addr, const void *buf, int words)
{
    ssize_t r = pwrite(a->fd, buf, words, (off_t)addr);
    a->syscalls++;
    if (r < 0) die("DDR write to 0x%lx (%d words) failed: %s",
                   addr, words, strerror(errno));
}

static void ddr_read(accel_t *a, unsigned long addr, void *buf, int words)
{
    ssize_t r = pread(a->fd, buf, words, (off_t)addr);
    a->syscalls++;
    if (r < 0) die("DDR read from 0x%lx (%d words) failed: %s",
                   addr, words, strerror(errno));
}

/* The eleven registers that are CONSTANT for an entire run. In "hoist" mode
 * these are written once at start; in "legacy" mode once per tile. */
static void write_constant_regs(accel_t *a)
{
    reg_write(a, REG_K_ADDR,  (uint32_t)DDR_K);
    reg_write(a, REG_K_BYTES, BYTES_I8_TILE);
    reg_write(a, REG_K_TUSER, a->tuser_k);
    reg_write(a, REG_X_ADDR,  (uint32_t)DDR_X);
    reg_write(a, REG_X_BYTES, BYTES_I8_TILE);
    reg_write(a, REG_X_TUSER, a->tuser_x);
    reg_write(a, REG_B_ADDR,  (uint32_t)DDR_B);
    reg_write(a, REG_B_BYTES, BYTES_I32_TILE);
    reg_write(a, REG_B_TUSER, a->tuser_b);
    reg_write(a, REG_Y_ADDR,  (uint32_t)DDR_Y);
    reg_write(a, REG_Y_BYTES, BYTES_I32_TILE);
}

/* One physical 8x8x8 invocation.
 *   xt  : 64 INT8, column-major   index = kk*8 + rr
 *   wt  : 64 INT8, row-major      index = kk*8 + cc
 *   par : 64 INT32, column-major  index = cc*8 + rr   (in = bias, out = result)
 * The accelerator computes Y = X*K + bias, so feeding the running partial back
 * in as the bias accumulates across the K dimension. */
static void run_tile(accel_t *a, int legacy,
                     const int8_t *xt, const int8_t *wt, int32_t *par)
{
    long t0 = now_us();

    if (a->sim) {
        /* Software model of one 8x8x8 invocation, using exactly the packing
         * rules in handoff section 4. Lets the tile/accumulation logic be
         * checked without hardware: --sim must give the same numbers as the
         * real device, and both must match the plain software GEMM. */
        for (int cc = 0; cc < TILE; cc++)
            for (int rr = 0; rr < TILE; rr++) {
                int32_t acc = par[cc * TILE + rr];
                for (int kk = 0; kk < TILE; kk++)
                    acc += (int32_t)xt[kk * TILE + rr] * (int32_t)wt[kk * TILE + cc];
                par[cc * TILE + rr] = acc;
            }
        a->us_total += (now_us() - t0);
        a->invocations++;
        return;
    }

    /* Handshake order copied verbatim from drv_test_128.c:run_physical_tile():
     * clear START, write the three operands, (re)write the address registers,
     * assert START, poll Y_DONE, read Y back, clear START again. The two START
     * clears are part of the proven sequence -- do not drop them. */
    reg_write(a, REG_START, 0u);

    ddr_write(a, DDR_K, wt, WORDS_I8_TILE);
    ddr_write(a, DDR_X, xt, WORDS_I8_TILE);
    ddr_write(a, DDR_B, par, WORDS_I32_TILE);

    if (legacy) write_constant_regs(a);

    long tc0 = now_us();
    reg_write(a, REG_START, 1u);

    long waited;
    for (;;) {
        uint32_t d = reg_read(a, REG_DONE);
        a->done_polls++;
        if (d & 1u) break;
        waited = now_us() - tc0;
        if (waited > DONE_TIMEOUT_US) {
            reg_write(a, REG_START, 0u);
            die("Y_DONE never asserted within %ld us (invocation %ld). "
                "Check the register map and tuser values.",
                DONE_TIMEOUT_US, a->invocations);
        }
    }
    long tc1 = now_us();

    ddr_read(a, DDR_Y, par, WORDS_I32_TILE);
    reg_write(a, REG_START, 0u);

    long t1 = now_us();
    a->us_cmd_to_done += (tc1 - tc0);
    a->us_total       += (t1 - t0);
    a->invocations++;
}

/* ======================================================================== */
/* GEMM: X[8,K] * W[K,N] + bias, on the accelerator                          */
/* ======================================================================== */

static void gemm_accel(accel_t *a, int legacy,
                       const int8_t xq[MAX_TOKENS][HID],
                       const tensor_t *W, const int32_t *bias_i32,
                       int Kdim, int Ndim,
                       int32_t out[MAX_TOKENS][HID])
{
    int8_t  xt[64], wt[64];
    int32_t par[64];

    for (int n0 = 0; n0 < Ndim; n0 += TILE) {
        for (int cc = 0; cc < TILE; cc++)
            for (int rr = 0; rr < TILE; rr++)
                par[cc * TILE + rr] = bias_i32[n0 + cc];

        for (int k0 = 0; k0 < Kdim; k0 += TILE) {
            for (int kk = 0; kk < TILE; kk++) {
                for (int rr = 0; rr < TILE; rr++)
                    xt[kk * TILE + rr] =
                        (rr < MAX_TOKENS) ? xq[rr][k0 + kk] : 0;
                for (int cc = 0; cc < TILE; cc++)
                    wt[kk * TILE + cc] = w_i8(W, k0 + kk, n0 + cc);
            }
            run_tile(a, legacy, xt, wt, par);
        }

        for (int rr = 0; rr < MAX_TOKENS; rr++)
            for (int cc = 0; cc < TILE; cc++)
                out[rr][n0 + cc] = par[cc * TILE + rr];
    }
}

/* Exact software INT32 equivalent -- the reference the accelerator must match */
static void gemm_sw_i32(const int8_t xq[MAX_TOKENS][HID],
                        const tensor_t *W, const int32_t *bias_i32,
                        int Kdim, int Ndim,
                        int32_t out[MAX_TOKENS][HID])
{
    for (int r = 0; r < MAX_TOKENS; r++)
        for (int c = 0; c < Ndim; c++) {
            int32_t acc = bias_i32[c];
            for (int k = 0; k < Kdim; k++)
                acc += (int32_t)xq[r][k] * (int32_t)w_i8(W, k, c);
            out[r][c] = acc;
        }
}

/* FP32 reference from the UNQUANTIZED X and dequantized weights: this is the
 * number that tells us how much accuracy the INT8 activation quantization cost. */
static void gemm_fp_ref(const double x[MAX_TOKENS][HID],
                        const tensor_t *W, double wscale,
                        const tensor_t *B,
                        int Kdim, int Ndim,
                        double out[MAX_TOKENS][HID])
{
    for (int r = 0; r < MAX_TOKENS; r++)
        for (int c = 0; c < Ndim; c++) {
            double acc = t_get1(B, c);
            for (int k = 0; k < Kdim; k++)
                acc += x[r][k] * (double)w_i8(W, k, c) * wscale;
            out[r][c] = acc;
        }
}

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

static const int REF_IDS[MAX_TOKENS]  = {101, 4708, 1024, 3710, 4511, 102, 0, 0};
static const int REF_MASK[MAX_TOKENS] = {1, 1, 1, 1, 1, 1, 0, 0};

static void usage(const char *p)
{
    fprintf(stderr,
        "usage: %s --task \"<prompt>\" [options]\n"
        "  --model PATH   default /tmp/team43/team43_minilm_int8.bin\n"
        "  --vocab PATH   default /tmp/team43/vocab.txt\n"
        "  --dev   PATH   default /dev/accelerator_test\n"
        "  --stage 1|2    1 = tokenize+embed only (no device), 2 = full (default)\n"
        "  --sim          emulate the accelerator in software (no device opened)\n"
        "  --mode  legacy|hoist|both   register handling (default both)\n"
        "  --tuser K,X,B  override the three tuser register values\n"
        "  --show  N      how many output values to print (default 8)\n"
        "  --quiet\n", p);
    exit(1);
}

int main(int argc, char **argv)
{
    const char *task      = NULL;
    const char *model_path = "/tmp/team43/team43_minilm_int8.bin";
    const char *vocab_path = "/tmp/team43/vocab.txt";
    const char *dev_path   = "/dev/accelerator_test";
    int stage = 2, show = 8, sim = 0;
    int do_legacy = 1, do_hoist = 1;
    uint32_t tu_k = T43_TUSER_K, tu_x = T43_TUSER_X, tu_b = T43_TUSER_B;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--task")  && i + 1 < argc) task       = argv[++i];
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab_path = argv[++i];
        else if (!strcmp(argv[i], "--dev")   && i + 1 < argc) dev_path   = argv[++i];
        else if (!strcmp(argv[i], "--stage") && i + 1 < argc) stage = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--show")  && i + 1 < argc) show  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) g_quiet = 1;
        else if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--mode")  && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "legacy")) { do_legacy = 1; do_hoist = 0; }
            else if (!strcmp(m, "hoist")) { do_legacy = 0; do_hoist = 1; }
            else if (!strcmp(m, "both"))  { do_legacy = 1; do_hoist = 1; }
            else usage(argv[0]);
        }
        else if (!strcmp(argv[i], "--tuser") && i + 1 < argc) {
            unsigned a, b, c;
            if (sscanf(argv[++i], "%u,%u,%u", &a, &b, &c) != 3) usage(argv[0]);
            tu_k = a; tu_x = b; tu_b = c;
        }
        else usage(argv[0]);
    }
    if (!task) usage(argv[0]);

    printf("=== TEAM43 MILESTONE A ===\n");
    printf("task     : \"%s\"\n", task);
    printf("model    : %s\n", model_path);
    printf("stage    : %d   mode: %s%s\n", stage,
           do_legacy ? "legacy " : "", do_hoist ? "hoist" : "");

    /* ---- 1. map the model ------------------------------------------- */
    int mfd = open(model_path, O_RDONLY);
    if (mfd < 0) die("cannot open model '%s': %s", model_path, strerror(errno));
    struct stat st;
    if (fstat(mfd, &st) != 0) die("fstat on model failed: %s", strerror(errno));
    void *mp = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mfd, 0);
    if (mp == MAP_FAILED) die("mmap of model failed: %s", strerror(errno));
    model_t model;
    model.base = (const unsigned char *)mp;
    model.size = (size_t)st.st_size;
    printf("[model] mapped %lu bytes\n", (unsigned long)model.size);
    (void)COCO_NAMES;   /* the header defines it; Milestone B uses it */

    /* Cross-check the header against the package it claims to describe. */
    if (model.size != (size_t)MINILM_BLOB_BYTES)
        die("model is %lu bytes but the layout header says %ld -- "
            "header and .bin are from different exports",
            (unsigned long)model.size, (long)MINILM_BLOB_BYTES);
    if (MINILM_HIDDEN != HID || MINILM_MAX_TOKENS != MAX_TOKENS)
        die("header geometry (hidden=%d tokens=%d) disagrees with this build "
            "(hidden=%d tokens=%d)",
            MINILM_HIDDEN, MINILM_MAX_TOKENS, HID, MAX_TOKENS);

    /* ---- 2. tokenize the RUNTIME prompt ------------------------------ */
    vocab_t vocab;
    vocab_load(&vocab, vocab_path);

    int ids[MAX_TOKENS], mask[MAX_TOKENS], n_real = 0;
    tokenize_task(&vocab, task, ids, mask, &n_real);

    printf("[tok] ids  [");
    for (int i = 0; i < MAX_TOKENS; i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n[tok] mask [");
    for (int i = 0; i < MAX_TOKENS; i++) printf("%s%d", i ? ", " : "", mask[i]);
    printf("]\n[tok] toks ");
    for (int i = 0; i < MAX_TOKENS; i++) {
        const char *s = (ids[i] < vocab.n) ? vocab.tok[ids[i]] : "?";
        printf("%s%s", i ? " " : "", s);
    }
    printf("\n");

    if (!strcmp(task, "serve wine")) {
        int ok = 1;
        for (int i = 0; i < MAX_TOKENS; i++)
            if (ids[i] != REF_IDS[i] || mask[i] != REF_MASK[i]) ok = 0;
        printf("[tok] reference check (\"task: serve wine\"): %s\n",
               ok ? "MATCH" : "*** MISMATCH ***");
        if (!ok) die("tokenizer does not reproduce the handoff reference ids");
    }

    /* ---- 3. embeddings + LayerNorm + quantization -------------------- */
    static double x[MAX_TOKENS][HID];
    static int8_t xq[MAX_TOKENS][HID];
    build_embeddings(&model, ids, x);
    double sx = quantize_x(x, xq);
    printf("[emb] X[8,384] built. activation scale = %.9g\n", sx);
    printf("[emb] X[0][0..7] fp :");
    for (int i = 0; i < 8; i++) printf(" %+.6f", x[0][i]);
    printf("\n[emb] X[0][0..7] i8 :");
    for (int i = 0; i < 8; i++) printf(" %4d", (int)xq[0][i]);
    printf("\n");

    if (stage < 2) {
        printf("\nSTAGE 1 COMPLETE (no accelerator was touched).\n");
        return 0;
    }

    /* ---- 4. prepare the layer-0 Q projection ------------------------- */
    tensor_t Wq = TEN(&model, T43_L0_Q_W);
    tensor_t Bq = TEN(&model, T43_L0_Q_B);
    int Kdim = HID, Ndim = HID;
    if (Wq.d0 != HID || Wq.d1 != HID)
        die("%s is [%d,%d], expected [%d,%d]", Wq.name, Wq.d0, Wq.d1, HID, HID);
    if (t_len1d(&Bq) != HID)
        die("%s has length %d, expected %d", Bq.name, t_len1d(&Bq), HID);
    if (!Wq.is_i8) die("%s is not INT8", Wq.name);

    double sw = Wq.scale;
    double combined = sx * sw;
    int32_t *bias_i32 = (int32_t *)xcalloc(HID, sizeof(int32_t));
    for (int c = 0; c < HID; c++)
        bias_i32[c] = (int32_t)lround(t_get1(&Bq, c) / combined);

    printf("[gemm] X[8,%d] * Wq[%d,%d] + bq\n", Kdim, Kdim, Ndim);
    printf("[gemm] weight scale = %.9g   combined scale = %.9g\n", sw, combined);
    printf("[gemm] expected invocations = %d\n",
           (Kdim / TILE) * (Ndim / TILE));

    static int32_t y_sw[MAX_TOKENS][HID];
    long tsw0 = now_us();
    gemm_sw_i32(xq, &Wq, bias_i32, Kdim, Ndim, y_sw);
    long tsw = now_us() - tsw0;
    printf("[ref ] software INT32 GEMM done in %ld us (VEGA CPU)\n", tsw);

    /* ---- 5. run on the accelerator ----------------------------------- */
    int dfd = -1;
    if (sim) {
        printf("[sim ] SOFTWARE EMULATION -- the device is NOT opened. "
               "Timings below are meaningless; only the numbers matter.\n");
        do_legacy = 0; do_hoist = 1;
    } else {
        dfd = open(dev_path, O_RDWR);
        if (dfd < 0) die("cannot open '%s': %s", dev_path, strerror(errno));
    }

    static int32_t y_hw[2][MAX_TOKENS][HID];
    long tot_us[2] = {0, 0};
    accel_t acc[2];
    int ran[2] = {0, 0};
    const char *mode_name[2] = {"legacy", "hoist"};

    for (int m = 0; m < 2; m++) {
        int legacy = (m == 0);
        if (legacy && !do_legacy) continue;
        if (!legacy && !do_hoist) continue;

        memset(&acc[m], 0, sizeof(acc[m]));
        acc[m].fd = dfd;
        acc[m].sim = sim;
        acc[m].tuser_k = tu_k;
        acc[m].tuser_x = tu_x;
        acc[m].tuser_b = tu_b;

        printf("\n--- mode: %s ---\n", sim ? "sim" : mode_name[m]);
        if (!legacy && !sim) {
            write_constant_regs(&acc[m]);     /* the hoist: eleven writes, once */
            printf("[hoist] 11 constant registers written once (%ld syscalls)\n",
                   acc[m].syscalls);
        }

        long t0 = now_us();
        gemm_accel(&acc[m], legacy, xq, &Wq, bias_i32, Kdim, Ndim, y_hw[m]);
        tot_us[m] = now_us() - t0;
        ran[m] = 1;

        double inv = (double)acc[m].invocations;
        printf("invocations        : %ld\n", acc[m].invocations);
        printf("wall time          : %ld us (%.3f s)\n", tot_us[m], tot_us[m] / 1e6);
        printf("per invocation     : %.1f us\n", tot_us[m] / inv);
        if (!sim) {
            printf("command-to-DONE    : %.1f us avg\n", acc[m].us_cmd_to_done / inv);
            printf("software overhead  : %.1f %%\n",
                   100.0 * (1.0 - (double)acc[m].us_cmd_to_done / (double)tot_us[m]));
            printf("syscalls           : %ld (%.1f per invocation)\n",
                   acc[m].syscalls, acc[m].syscalls / inv);
            printf("DONE polls         : %ld (%.1f per invocation)\n",
                   acc[m].done_polls, acc[m].done_polls / inv);
        }

        /* correctness against the exact software integer GEMM */
        long match = 0, total = 0;
        int32_t worst = 0; int wr = -1, wc = -1;
        for (int r = 0; r < MAX_TOKENS; r++)
            for (int c = 0; c < Ndim; c++) {
                total++;
                int32_t d = y_hw[m][r][c] - y_sw[r][c];
                if (d == 0) match++;
                else if (labs((long)d) > labs((long)worst)) { worst = d; wr = r; wc = c; }
            }
        printf("INT32 match        : %ld/%ld %s\n", match, total,
               (match == total) ? "EXACT" : "*** MISMATCH ***");
        if (match != total)
            printf("worst element      : [%d][%d] hw=%d sw=%d diff=%d\n",
                   wr, wc, y_hw[m][wr][wc], y_sw[wr][wc], worst);
    }

    if (dfd >= 0) close(dfd);

    /* ---- 6. numeric report ------------------------------------------ */
    int mref = ran[1] ? 1 : 0;
    printf("\n--- layer-0 Q projection, first %d outputs of token 0 ---\n", show);
    for (int c = 0; c < show && c < Ndim; c++)
        printf("  q[0][%3d] = %+.6f   (int32 %d)\n",
               c, (double)y_hw[mref][0][c] * combined, y_hw[mref][0][c]);

    static double y_ref[MAX_TOKENS][HID];
    gemm_fp_ref(x, &Wq, sw, &Bq, Kdim, Ndim, y_ref);
    double maxerr = 0.0, sumerr = 0.0;
    for (int r = 0; r < MAX_TOKENS; r++)
        for (int c = 0; c < Ndim; c++) {
            double e = fabs((double)y_hw[mref][r][c] * combined - y_ref[r][c]);
            sumerr += e;
            if (e > maxerr) maxerr = e;
        }
    printf("quantized-vs-FP32  : max err %.9f   mean err %.9f\n",
           maxerr, sumerr / (MAX_TOKENS * (double)Ndim));
    printf("(handoff recorded max err 0.145594358 for this exact projection)\n");

    if (ran[0] && ran[1]) {
        int same = memcmp(y_hw[0], y_hw[1], sizeof(y_hw[0])) == 0;
        printf("\n--- REGISTER HOIST RESULT ---\n");
        printf("legacy : %ld us total, %.1f us/invocation, %.1f syscalls/inv\n",
               tot_us[0], tot_us[0] / (double)acc[0].invocations,
               acc[0].syscalls / (double)acc[0].invocations);
        printf("hoist  : %ld us total, %.1f us/invocation, %.1f syscalls/inv\n",
               tot_us[1], tot_us[1] / (double)acc[1].invocations,
               acc[1].syscalls / (double)acc[1].invocations);
        printf("speedup: %.2fx   results identical: %s\n",
               (double)tot_us[0] / (double)tot_us[1], same ? "yes" : "NO");
        if (!same) printf("*** hoist changed the numbers -- do NOT keep it ***\n");
    }

    printf("\nMILESTONE A COMPLETE.\n");
    return 0;
}
