/* team43_im2col -- how much does the im2col write pattern cost on THIS board?
 *
 * The pipeline spends most of its packing time in one loop: gathering input
 * patches into accelerator tile order. The loop sweeps every row-tile before
 * revisiting any of them, so its write working set is (row-tiles * 64) bytes.
 * When that exceeds the data cache, each 8 bytes stored costs a line fill.
 *
 * On a board whose DRAM came up healthy this pass costs ~0.66 s per call.
 * On one where it did not, the same call has been measured at 8.44 s -- 557 s
 * of an 816 s query -- while every sequential pass in the program was
 * unaffected. Blocking the row-tiles bounds the write set and should remove
 * most of that gap without changing a single output byte.
 *
 * This probe measures the effect directly, in seconds rather than the fourteen
 * minutes a full query takes, and proves the reordering is bit-identical.
 *
 * No device, no models, no network. Build:
 *   riscv64-buildroot-linux-gnu-gcc -std=gnu99 -O2 -static \
 *       -o team43_im2col team43_im2col.c -lrt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static long now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000L + ts.tv_nsec / 1000;
}

/* One im2col pass, exactly the pipeline's inner loop.
 * blk = row-tiles per block; a huge value reproduces the old single sweep. */
static void im2col(int8_t *packX, const int8_t *xq,
                   int C, int H, int W, int k, int s, int p,
                   int Wo, int rows, int m_base, int blk)
{
    int K  = C * k * k;
    int KT = (K + 7) / 8;
    int MT = (rows + 7) / 8;

    memset(packX, 0, (size_t)MT * KT * 64);
    for (int mt0 = 0; mt0 < MT; mt0 += blk) {
        int mtn = MT - mt0; if (mtn > blk) mtn = blk;
        int r_lo = mt0 * 8;
        int r_hi = r_lo + mtn * 8; if (r_hi > rows) r_hi = rows;
        for (int kk_g = 0; kk_g < K; kk_g++) {
            int c = kk_g / (k*k), rem = kk_g % (k*k);
            int di = rem / k, dj = rem % k;
            int k0 = kk_g / 8, kk = kk_g % 8;
            const int8_t *plane = xq + (size_t)c * H * W;
            int8_t *base = packX + (long)k0*64 + kk*8;
            for (int r = r_lo; r < r_hi; r++) {
                int gr = m_base + r;
                int oy = gr / Wo, ox = gr % Wo;
                int iy = oy*s - p + di, ix = ox*s - p + dj;
                if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W)
                    base[(long)(r/8) * KT * 64 + (r%8)] = plane[(size_t)iy*W + ix];
            }
        }
    }
}

static uint32_t fnv(const void *p, size_t n)
{
    const uint8_t *b = p; uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 16777619u; }
    return h;
}

struct shape { const char *name; int C, H, W, k, s, p; };

int main(void)
{
    /* Three shapes spanning the detector: a wide early layer with many rows,
     * a middle one, and a deep one with few rows but a long K. */
    struct shape shapes[] = {
        { "early  C=32  128x128 3x3", 32, 128, 128, 3, 1, 1 },
        { "middle C=128  32x32  3x3", 128, 32, 32, 3, 1, 1 },
        { "deep   C=256  16x16  3x3", 256, 16, 16, 3, 1, 1 },
    };
    int blocks[] = { 1 << 24, 256, 128, 64, 32, 16, 8 };
    int nb = (int)(sizeof(blocks)/sizeof(blocks[0]));

    printf("=== TEAM43 im2col blocking probe ===\n");
    printf("block = row-tiles filled before moving on; write set = block * 64 B\n\n");

    long best_total[16]; memset(best_total, 0, sizeof(best_total));

    for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); si++) {
        struct shape *sh = &shapes[si];
        int Wo   = (sh->W + 2*sh->p - sh->k) / sh->s + 1;
        int Ho   = (sh->H + 2*sh->p - sh->k) / sh->s + 1;
        int rows = Ho * Wo;
        if (rows > 4096) rows = 4096;
        int K  = sh->C * sh->k * sh->k;
        int KT = (K + 7) / 8;
        int MT = (rows + 7) / 8;

        size_t nx = (size_t)sh->C * sh->H * sh->W;
        size_t np = (size_t)MT * KT * 64;
        int8_t *xq    = malloc(nx);
        int8_t *packX = malloc(np);
        if (!xq || !packX) { printf("out of memory\n"); return 1; }
        for (size_t i = 0; i < nx; i++) xq[i] = (int8_t)(i * 31 + 7);

        printf("%s   rows=%d K=%d  packX=%lu KB  unblocked write set=%d KB\n",
               sh->name, rows, K, (unsigned long)(np >> 10), (MT * 64) >> 10);

        uint32_t ref = 0;
        long base_us = 0;
        for (int bi = 0; bi < nb; bi++) {
            int blk = blocks[bi] > MT ? MT : blocks[bi];
            long t = now_us();
            im2col(packX, xq, sh->C, sh->H, sh->W, sh->k, sh->s, sh->p,
                   Wo, rows, 0, blk);
            long d = now_us() - t;
            uint32_t h = fnv(packX, np);
            if (bi == 0) { ref = h; base_us = d; }
            char lbl[16];
            if (blocks[bi] >= (1<<24)) snprintf(lbl, sizeof lbl, "none");
            else                       snprintf(lbl, sizeof lbl, "%d", blk);
            printf("   block %7s : %8.1f ms   %5.2fx   %s\n",
                   lbl, d / 1000.0,
                   base_us ? (double)base_us / (double)d : 1.0,
                   h == ref ? "identical" : "*** DIFFERENT -- BUG ***");
            best_total[bi] += d;
        }
        printf("\n");
        free(xq); free(packX);
    }

    printf("=== totals across the three shapes ===\n");
    for (int bi = 0; bi < nb; bi++) {
        char lbl[16];
        if (blocks[bi] >= (1<<24)) snprintf(lbl, sizeof lbl, "none");
        else                       snprintf(lbl, sizeof lbl, "%d", blocks[bi]);
        printf("   block %-7s : %8.1f ms   %5.2fx\n",
               lbl, best_total[bi] / 1000.0,
               best_total[0] ? (double)best_total[0] / (double)best_total[bi] : 1.0);
    }

    printf("\nPick the fastest block size and pass it to the pipeline as\n");
    printf("  --im2col-block N        (the build already defaults to 64)\n");
    printf("Every row must read 'identical'; the reordering must not change a byte.\n");
    return 0;
}
