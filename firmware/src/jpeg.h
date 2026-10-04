/*
 * jpeg.h -- baseline JPEG decoder, single header, no dependencies.
 *
 * Team 43 / DVCon India 2026. Written so the board can take an ordinary .jpg
 * directly, instead of a pre-converted PPM. Nothing about the image is then
 * prepared off-board: decode, letterbox, normalise, detect, rank and draw all
 * happen on the Genesys 2.
 *
 * Supports: baseline sequential DCT (SOF0), 8-bit, 1 or 3 components, any
 * chroma subsampling expressible as per-component H/V sampling factors,
 * restart intervals. NOT supported: progressive (SOF2), arithmetic coding,
 * 12-bit, CMYK. COCO images are baseline, which is the common case.
 *
 *   unsigned char *rgb = jpeg_decode(buf, len, &W, &H);   // caller frees
 *   returns NULL on any unsupported or malformed input, with *err set.
 */
#ifndef TEAM43_JPEG_H
#define TEAM43_JPEG_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    const unsigned char *d;
    size_t len, pos;
    unsigned int bitbuf;
    int bitcnt;
    int eof;
} jbits_t;

typedef struct {
    unsigned char bits[17];
    unsigned char vals[256];
    int mincode[17], maxcode[18], valptr[17];
} jhuff_t;

typedef struct {
    int id, h, v, tq, td, ta, dcpred;
    int bw, bh;              /* blocks across / down for this component */
    unsigned char *pix;      /* decoded samples, bw*8 x bh*8 */
} jcomp_t;

/* Identifies which decoder got compiled in. Printed at startup so a stale
 * header in the build directory can never be mistaken for a new one. */
#define JPEG_IMPL_NAME "integer islow IDCT + table colour"

#include <time.h>

static const unsigned char JZIGZAG[64] = {
     0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
};

static void jhuff_build(jhuff_t *h)
{
    int code = 0, k = 0;
    for (int i = 1; i <= 16; i++) {
        h->valptr[i] = k;
        h->mincode[i] = code;
        code += h->bits[i];
        k += h->bits[i];
        h->maxcode[i] = h->bits[i] ? code - 1 : -1;
        code <<= 1;
    }
    h->maxcode[17] = 0x7fffffff;
}

static int jbit(jbits_t *b)
{
    if (b->bitcnt == 0) {
        if (b->pos >= b->len) { b->eof = 1; return 0; }
        unsigned char c = b->d[b->pos++];
        if (c == 0xFF) {
            /* byte stuffing: FF 00 is a literal FF; any other marker ends data */
            if (b->pos < b->len && b->d[b->pos] == 0x00) b->pos++;
            else { b->eof = 1; return 0; }
        }
        b->bitbuf = c;
        b->bitcnt = 8;
    }
    b->bitcnt--;
    return (b->bitbuf >> b->bitcnt) & 1;
}

static int jhuff_decode(jbits_t *b, const jhuff_t *h)
{
    int code = jbit(b), l = 1;
    while (l <= 16 && (h->maxcode[l] < 0 || code > h->maxcode[l])) {
        code = (code << 1) | jbit(b);
        l++;
        if (b->eof) return -1;
    }
    if (l > 16) return -1;
    return h->vals[h->valptr[l] + code - h->mincode[l]];
}

static int jreceive_extend(jbits_t *b, int s)
{
    if (s == 0) return 0;
    int v = 0;
    for (int i = 0; i < s; i++) v = (v << 1) | jbit(b);
    if (v < (1 << (s - 1))) v -= (1 << s) - 1;
    return v;
}

/* ---- phase instrumentation (timers only, no behavioural change) --------
 * The on-board decode is far slower than any host estimate predicted, and we
 * do not yet know which phase is responsible. These split it three ways.   */
static long jt_scan_us = 0, jt_idct_us = 0, jt_color_us = 0;
static long jt_blocks = 0, jt_pixels = 0;
static long jt_now_us(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}
static void jpeg_reset_timers(void) __attribute__((unused));
static void jpeg_reset_timers(void)
{
    jt_scan_us = jt_idct_us = jt_color_us = 0; jt_blocks = jt_pixels = 0;
}

/* Integer IDCT (13-bit fixed point, the classic jpeg-6b "islow" dataflow).
 *
 * The float version this replaces cost 7,942 us per 8x8 block on the VEGA
 * core -- about 397,000 cycles at 50 MHz for at most 1024 multiply-adds,
 * i.e. roughly 390 cycles per floating-point operation. That is the cost of
 * soft-float on a core with no FPU. Every operation below is integer.
 *
 * The two shortcuts from the float version are kept because they are worth
 * more than the arithmetic: a DC-only block is filled directly, and a row of
 * zero coefficients is skipped in the first pass. In a typical photo most
 * chroma blocks and many luma blocks are DC-only.
 */
#define JCONST_BITS 13
#define JPASS_BITS   2
#define JDESCALE(x, n)  (((int32_t)(x) + ((int32_t)1 << ((n) - 1))) >> (n))

#define JFIX_0_298631336  2446
#define JFIX_0_390180644  3196
#define JFIX_0_541196100  4433
#define JFIX_0_765366865  6270
#define JFIX_0_899976223  7373
#define JFIX_1_175875602  9633
#define JFIX_1_501321110 12299
#define JFIX_1_847759065 15137
#define JFIX_1_961570560 16069
#define JFIX_2_053119869 16819
#define JFIX_2_562915447 20995
#define JFIX_3_072711026 25172

static inline unsigned char jclamp8(int32_t v)
{
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

static void jidct(const int *in, unsigned char *out, int stride)
{
    int32_t ws[64];

    int any = 0;
    for (int i = 1; i < 64; i++) if (in[i]) { any = 1; break; }
    if (!any) {                                  /* DC-only: flat block */
        int32_t d = in[0];
        int32_t v = ((d < 0 ? d - 4 : d + 4) / 8) + 128;   /* == lround(d/8) */
        unsigned char b = jclamp8(v);
        for (int y = 0; y < 8; y++) memset(out + y * stride, b, 8);
        return;
    }

    /* ---- pass 1: columns, results scaled up by JPASS_BITS ---- */
    for (int c = 0; c < 8; c++) {
        const int *ip = in + c;
        int32_t *wp = ws + c;

        if (ip[8] == 0 && ip[16] == 0 && ip[24] == 0 && ip[32] == 0 &&
            ip[40] == 0 && ip[48] == 0 && ip[56] == 0) {
            int32_t dc = (int32_t)ip[0] << JPASS_BITS;
            for (int y = 0; y < 8; y++) wp[y * 8] = dc;
            continue;
        }

        int32_t z2 = ip[16], z3 = ip[48];
        int32_t z1 = (z2 + z3) * JFIX_0_541196100;
        int32_t tmp2 = z1 + z3 * -JFIX_1_847759065;
        int32_t tmp3 = z1 + z2 *  JFIX_0_765366865;

        z2 = ip[0]; z3 = ip[32];
        int32_t tmp0 = (z2 + z3) << JCONST_BITS;
        int32_t tmp1 = (z2 - z3) << JCONST_BITS;

        int32_t t10 = tmp0 + tmp3, t13 = tmp0 - tmp3;
        int32_t t11 = tmp1 + tmp2, t12 = tmp1 - tmp2;

        tmp0 = ip[56]; tmp1 = ip[40]; tmp2 = ip[24]; tmp3 = ip[8];
        z1 = tmp0 + tmp3; z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2; int32_t z4 = tmp1 + tmp3;
        int32_t z5 = (z3 + z4) * JFIX_1_175875602;

        tmp0 *= JFIX_0_298631336;
        tmp1 *= JFIX_2_053119869;
        tmp2 *= JFIX_3_072711026;
        tmp3 *= JFIX_1_501321110;
        z1 *= -JFIX_0_899976223;
        z2 *= -JFIX_2_562915447;
        z3 *= -JFIX_1_961570560;
        z4 *= -JFIX_0_390180644;
        z3 += z5; z4 += z5;

        tmp0 += z1 + z3; tmp1 += z2 + z4;
        tmp2 += z2 + z3; tmp3 += z1 + z4;

        wp[0]  = JDESCALE(t10 + tmp3, JCONST_BITS - JPASS_BITS);
        wp[56] = JDESCALE(t10 - tmp3, JCONST_BITS - JPASS_BITS);
        wp[8]  = JDESCALE(t11 + tmp2, JCONST_BITS - JPASS_BITS);
        wp[48] = JDESCALE(t11 - tmp2, JCONST_BITS - JPASS_BITS);
        wp[16] = JDESCALE(t12 + tmp1, JCONST_BITS - JPASS_BITS);
        wp[40] = JDESCALE(t12 - tmp1, JCONST_BITS - JPASS_BITS);
        wp[24] = JDESCALE(t13 + tmp0, JCONST_BITS - JPASS_BITS);
        wp[32] = JDESCALE(t13 - tmp0, JCONST_BITS - JPASS_BITS);
    }

    /* ---- pass 2: rows ---- */
    for (int r = 0; r < 8; r++) {
        int32_t *wp = ws + r * 8;
        unsigned char *op = out + r * stride;

        int32_t z2 = wp[2], z3 = wp[6];
        int32_t z1 = (z2 + z3) * JFIX_0_541196100;
        int32_t tmp2 = z1 + z3 * -JFIX_1_847759065;
        int32_t tmp3 = z1 + z2 *  JFIX_0_765366865;

        int32_t tmp0 = (wp[0] + wp[4]) << JCONST_BITS;
        int32_t tmp1 = (wp[0] - wp[4]) << JCONST_BITS;

        int32_t t10 = tmp0 + tmp3, t13 = tmp0 - tmp3;
        int32_t t11 = tmp1 + tmp2, t12 = tmp1 - tmp2;

        tmp0 = wp[7]; tmp1 = wp[5]; tmp2 = wp[3]; tmp3 = wp[1];
        z1 = tmp0 + tmp3; z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2; int32_t z4 = tmp1 + tmp3;
        int32_t z5 = (z3 + z4) * JFIX_1_175875602;

        tmp0 *= JFIX_0_298631336;
        tmp1 *= JFIX_2_053119869;
        tmp2 *= JFIX_3_072711026;
        tmp3 *= JFIX_1_501321110;
        z1 *= -JFIX_0_899976223;
        z2 *= -JFIX_2_562915447;
        z3 *= -JFIX_1_961570560;
        z4 *= -JFIX_0_390180644;
        z3 += z5; z4 += z5;

        tmp0 += z1 + z3; tmp1 += z2 + z4;
        tmp2 += z2 + z3; tmp3 += z1 + z4;

        op[0] = jclamp8(JDESCALE(t10 + tmp3, JCONST_BITS + JPASS_BITS + 3) + 128);
        op[7] = jclamp8(JDESCALE(t10 - tmp3, JCONST_BITS + JPASS_BITS + 3) + 128);
        op[1] = jclamp8(JDESCALE(t11 + tmp2, JCONST_BITS + JPASS_BITS + 3) + 128);
        op[6] = jclamp8(JDESCALE(t11 - tmp2, JCONST_BITS + JPASS_BITS + 3) + 128);
        op[2] = jclamp8(JDESCALE(t12 + tmp1, JCONST_BITS + JPASS_BITS + 3) + 128);
        op[5] = jclamp8(JDESCALE(t12 - tmp1, JCONST_BITS + JPASS_BITS + 3) + 128);
        op[3] = jclamp8(JDESCALE(t13 + tmp0, JCONST_BITS + JPASS_BITS + 3) + 128);
        op[4] = jclamp8(JDESCALE(t13 - tmp0, JCONST_BITS + JPASS_BITS + 3) + 128);
    }
}

static unsigned char *jpeg_decode(const unsigned char *data, size_t len,
                                  int *W, int *H, const char **err)
{
    const char *dummy;
    if (!err) err = &dummy;
    *err = NULL;
    if (len < 4 || data[0] != 0xFF || data[1] != 0xD8) { *err = "not a JPEG"; return NULL; }

    unsigned short qt[4][64];
    jhuff_t hdc[4], hac[4];
    memset(qt, 0, sizeof(qt));
    memset(hdc, 0, sizeof(hdc));
    memset(hac, 0, sizeof(hac));
    jcomp_t comp[3];
    int ncomp = 0, width = 0, height = 0, restart = 0;
    size_t p = 2;
    unsigned char *rgb = NULL;

    while (p + 3 < len) {
        if (data[p] != 0xFF) { p++; continue; }
        int m = data[p + 1];
        p += 2;
        if (m == 0xD8 || m == 0x01 || (m >= 0xD0 && m <= 0xD7)) continue;
        if (m == 0xD9) break;
        if (p + 1 >= len) break;
        int seglen = (data[p] << 8) | data[p + 1];
        size_t seg = p + 2, segend = p + seglen;
        if (segend > len) { *err = "truncated segment"; goto fail; }

        if (m == 0xDB) {                                  /* DQT */
            while (seg < segend) {
                int pq = data[seg] >> 4, tq = data[seg] & 15;
                seg++;
                if (tq > 3) { *err = "bad DQT index"; goto fail; }
                for (int i = 0; i < 64; i++) {
                    qt[tq][i] = pq ? ((data[seg] << 8) | data[seg + 1]) : data[seg];
                    seg += pq ? 2 : 1;
                }
            }
        } else if (m == 0xC4) {                           /* DHT */
            while (seg < segend) {
                int tc = data[seg] >> 4, th = data[seg] & 15;
                seg++;
                if (th > 3) { *err = "bad DHT index"; goto fail; }
                jhuff_t *h = tc ? &hac[th] : &hdc[th];
                int total = 0;
                h->bits[0] = 0;
                for (int i = 1; i <= 16; i++) { h->bits[i] = data[seg++]; total += h->bits[i]; }
                if (total > 256) { *err = "bad DHT"; goto fail; }
                for (int i = 0; i < total; i++) h->vals[i] = data[seg++];
                jhuff_build(h);
            }
        } else if (m == 0xC0 || m == 0xC1) {              /* SOF0/SOF1 baseline */
            if (data[seg] != 8) { *err = "only 8-bit JPEG"; goto fail; }
            height = (data[seg + 1] << 8) | data[seg + 2];
            width  = (data[seg + 3] << 8) | data[seg + 4];
            ncomp  = data[seg + 5];
            if (ncomp != 1 && ncomp != 3) { *err = "only 1 or 3 components"; goto fail; }
            seg += 6;
            for (int i = 0; i < ncomp; i++) {
                comp[i].id = data[seg];
                comp[i].h  = data[seg + 1] >> 4;
                comp[i].v  = data[seg + 1] & 15;
                comp[i].tq = data[seg + 2];
                comp[i].dcpred = 0;
                if (comp[i].h < 1 || comp[i].h > 4 || comp[i].v < 1 || comp[i].v > 4) {
                    *err = "bad sampling factors"; goto fail;
                }
                seg += 3;
            }
        } else if (m == 0xC2) {
            *err = "progressive JPEG not supported"; goto fail;
        } else if (m == 0xDD) {                           /* DRI */
            restart = (data[seg] << 8) | data[seg + 1];
        } else if (m == 0xDA) {                           /* SOS -- scan data */
            if (!width || !height) { *err = "SOS before SOF"; goto fail; }
            int ns = data[seg];
            seg++;
            for (int i = 0; i < ns; i++) {
                int cid = data[seg], tt = data[seg + 1];
                for (int j = 0; j < ncomp; j++)
                    if (comp[j].id == cid) { comp[j].td = tt >> 4; comp[j].ta = tt & 15; }
                seg += 2;
            }
            seg += 3;                                     /* Ss, Se, Ah/Al */

            int hmax = 1, vmax = 1;
            for (int i = 0; i < ncomp; i++) {
                if (comp[i].h > hmax) hmax = comp[i].h;
                if (comp[i].v > vmax) vmax = comp[i].v;
            }
            int mcux = (width  + 8 * hmax - 1) / (8 * hmax);
            int mcuy = (height + 8 * vmax - 1) / (8 * vmax);
            for (int i = 0; i < ncomp; i++) {
                comp[i].bw = mcux * comp[i].h;
                comp[i].bh = mcuy * comp[i].v;
                comp[i].pix = calloc((size_t)comp[i].bw * 8 * comp[i].bh * 8, 1);
                if (!comp[i].pix) { *err = "out of memory"; goto fail; }
            }

            jbits_t b = { data, len, seg, 0, 0, 0 };
            int block[64], coef[64];
            long mcu = 0, until_restart = restart;
            long _tscan = jt_now_us();
            for (int my = 0; my < mcuy && !b.eof; my++) {
                for (int mx = 0; mx < mcux && !b.eof; mx++) {
                    if (restart && until_restart == 0) {
                        b.bitcnt = 0;
                        while (b.pos + 1 < b.len &&
                               !(b.d[b.pos] == 0xFF && b.d[b.pos+1] >= 0xD0 && b.d[b.pos+1] <= 0xD7))
                            b.pos++;
                        if (b.pos + 1 < b.len) b.pos += 2;
                        for (int i = 0; i < ncomp; i++) comp[i].dcpred = 0;
                        until_restart = restart;
                    }
                    for (int ci = 0; ci < ncomp; ci++) {
                        jcomp_t *c = &comp[ci];
                        for (int by = 0; by < c->v; by++)
                            for (int bx = 0; bx < c->h; bx++) {
                                memset(coef, 0, sizeof(coef));
                                int t = jhuff_decode(&b, &hdc[c->td]);
                                if (t < 0) { b.eof = 1; break; }
                                c->dcpred += jreceive_extend(&b, t);
                                coef[0] = c->dcpred * qt[c->tq][0];
                                for (int k = 1; k < 64; ) {
                                    int rs = jhuff_decode(&b, &hac[c->ta]);
                                    if (rs < 0) { b.eof = 1; break; }
                                    int r = rs >> 4, s = rs & 15;
                                    if (s == 0) { if (r != 15) break; k += 16; continue; }
                                    k += r;
                                    if (k > 63) break;
                                    coef[JZIGZAG[k]] = jreceive_extend(&b, s) * qt[c->tq][k];
                                    k++;
                                }
                                memcpy(block, coef, sizeof(block));
                                int px = (mx * c->h + bx) * 8;
                                int py = (my * c->v + by) * 8;
                                { long _ti = jt_now_us();
                                  jidct(block, c->pix + (size_t)py * (c->bw * 8) + px, c->bw * 8);
                                  jt_idct_us += jt_now_us() - _ti; jt_blocks++; }
                            }
                    }
                    mcu++;
                    if (restart) until_restart--;
                }
            }
            jt_scan_us += jt_now_us() - _tscan;

            rgb = malloc((size_t)width * height * 3);
            if (!rgb) { *err = "out of memory"; goto fail; }
            /* Integer YCbCr -> RGB.
             *
             * The double-precision version this replaces cost 121.3 us per
             * pixel on the VEGA core -- about 6,065 cycles -- for three
             * double multiplies, three lround() calls and six integer
             * divisions. Two changes remove essentially all of it:
             *
             *  1. the four colour terms become 16-bit fixed point, looked up
             *     in 256-entry tables built once, so there is no multiply in
             *     the pixel loop at all;
             *  2. the per-pixel subsampling divisions become per-row and
             *     per-column index tables, computed once.
             *
             * Output differs from the double version by at most 1 LSB. */
            long _tcol = jt_now_us();
            {
                static int32_t tCrR[256], tCbB[256], tCrG[256], tCbG[256];
                static int ctab_init = 0;
                if (!ctab_init) {
                    for (int i = 0; i < 256; i++) {
                        int d = i - 128;
                        tCrR[i] = (int32_t)( 91881L * d + 32768L);  /*  1.402000 */
                        tCbB[i] = (int32_t)(116130L * d + 32768L);  /*  1.772000 */
                        tCrG[i] = (int32_t)(-46802L * d);           /* -0.714136 */
                        tCbG[i] = (int32_t)(-22554L * d + 32768L);  /* -0.344136 */
                    }
                    ctab_init = 1;
                }
                int *mapx[3], *mapy[3];
                for (int ci = 0; ci < ncomp; ci++) {
                    jcomp_t *c = &comp[ci];
                    mapx[ci] = (int *)malloc(sizeof(int) * (size_t)width);
                    mapy[ci] = (int *)malloc(sizeof(int) * (size_t)height);
                    if (!mapx[ci] || !mapy[ci]) { *err = "out of memory"; goto fail; }
                    for (int x = 0; x < width;  x++) mapx[ci][x] = x * c->h / hmax;
                    for (int y = 0; y < height; y++)
                        mapy[ci][y] = (y * c->v / vmax) * (c->bw * 8);
                }
                unsigned char *o = rgb;
                for (int y = 0; y < height; y++) {
                    const unsigned char *pY = comp[0].pix + mapy[0][y];
                    const unsigned char *pB = (ncomp == 3) ? comp[1].pix + mapy[1][y] : NULL;
                    const unsigned char *pR = (ncomp == 3) ? comp[2].pix + mapy[2][y] : NULL;
                    const int *mx0 = mapx[0];
                    const int *mx1 = (ncomp == 3) ? mapx[1] : NULL;
                    const int *mx2 = (ncomp == 3) ? mapx[2] : NULL;
                    for (int x = 0; x < width; x++) {
                        int32_t Y = pY[mx0[x]];
                        if (ncomp == 3) {
                            int Cb = pB[mx1[x]], Cr = pR[mx2[x]];
                            o[0] = jclamp8(Y + (tCrR[Cr] >> 16));
                            o[1] = jclamp8(Y + ((tCbG[Cb] + tCrG[Cr]) >> 16));
                            o[2] = jclamp8(Y + (tCbB[Cb] >> 16));
                        } else {
                            o[0] = o[1] = o[2] = jclamp8(Y);
                        }
                        o += 3;
                    }
                }
                for (int ci = 0; ci < ncomp; ci++) { free(mapx[ci]); free(mapy[ci]); }
            }
            jt_color_us += jt_now_us() - _tcol;
            jt_pixels += (long)width * height;
            for (int i = 0; i < ncomp; i++) { free(comp[i].pix); comp[i].pix = NULL; }
            *W = width; *H = height;
            return rgb;
        }
        p = segend;
    }
    *err = "no scan found";
fail:
    for (int i = 0; i < ncomp; i++) if (comp[i].pix) free(comp[i].pix);
    free(rgb);
    return NULL;
}

#endif /* TEAM43_JPEG_H */
