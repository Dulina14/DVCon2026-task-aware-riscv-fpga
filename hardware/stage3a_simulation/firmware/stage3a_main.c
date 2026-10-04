/* ===========================================================================
 * Stage 3A - 8x8 accelerator integration test (internal SRAM only, no DDR)
 * ---------------------------------------------------------------------------
 *   X[r][k] = ((3r + 2k) % 11) - 5      signed int8
 *   K[k][c] = ((5k + 3c) %  9) - 4      signed int8
 *   B[r][c] = 2r - c                    signed int32
 *   Y = X*K + B                         signed int32, 64 results
 *
 * Memory packing - derived from the VERIFIED 2x2 behaviour and generalised:
 *   K row-major    : K[k][c] at K_ADDR + k*8 + c     (beat k = row k of K)
 *   X column-major : X[r][k] at X_ADDR + k*8 + r     (beat k = column k of X)
 *   B column-major : B[r][c] at B_ADDR + c*32 + r*4  (beat c = column c)
 *   Y column-major : Y[r][c] at Y_ADDR + c*32 + r*4  (beat c = column c)
 *
 * UART markers: B=boot  C=configured  S=started  P=pass  F=fail  -> "BCSP"
 * No rx_uart(), no long prints, finite timeout.
 * =========================================================================== */
#include "uart.h"
#include "stdlib.h"

UL g_program_entry; volatile UL core_flag; UL g_dtb_address;  /* used by crt.S */

#define write_csr(reg,val) ({ asm volatile ("csrw " #reg ", %0" :: "rK"(val)); })
static void cache_maint(void){ write_csr(0x5c8,1); write_csr(0x5c0,1); }

#define ACC     0x20040000u
#define REG(a)  (*(volatile unsigned int  *)(unsigned long)(a))
#define BYTE(a) (*(volatile unsigned char *)(unsigned long)(a))

#define K_ADDR  0x00020000u   /* 64  B : 0x20000-0x2003F */
#define X_ADDR  0x00020100u   /* 64  B : 0x20100-0x2013F */
#define B_ADDR  0x00020200u   /* 256 B : 0x20200-0x202FF */
#define Y_ADDR  0x00020300u   /* 256 B : 0x20300-0x203FF */
#define SIG     0x00020400u
#define N       8

/* compile-time proof: buffers do not overlap and stay below .data/.bss(0x37000) */
_Static_assert(K_ADDR + 64  <= X_ADDR,   "K overlaps X");
_Static_assert(X_ADDR + 64  <= B_ADDR,   "X overlaps B");
_Static_assert(B_ADDR + 256 <= Y_ADDR,   "B overlaps Y");
_Static_assert(Y_ADDR + 256 <= SIG,      "Y overlaps SIG");
_Static_assert(SIG   + 4    <= 0x37000u, "buffers collide with .data/.bss/stack");

static int xval(int r,int k){ return ((3*r + 2*k) % 11) - 5; }
static int kval(int k,int c){ return ((5*k + 3*c) %  9) - 4; }
static int bval(int r,int c){ return 2*r - c; }

/* Precomputed operand tables.  Building these ONCE removes 512 integer
   modulo operations from the verification loop - at -O0 those cost ~1 ms of
   simulated time, which previously ran past the end of a 1 ms capture. */
static signed char Xv[N][N];   /* Xv[r][k] */
static signed char Kv[N][N];   /* Kv[k][c] */

int main(void)
{
    int r, k, c, ok, expected;
    unsigned int guard;

    init_uart(0x1b);
    tx_uart('B');                                   /* processor booted      */

    for (r = 0; r < N; r++)
        for (k = 0; k < N; k++) Xv[r][k] = (signed char)xval(r,k);
    for (k = 0; k < N; k++)
        for (c = 0; c < N; c++) Kv[k][c] = (signed char)kval(k,c);

    /* ---- K : row-major, signed int8 via two's complement ---------------- */
    for (k = 0; k < N; k++)
        for (c = 0; c < N; c++)
            BYTE(K_ADDR + k*N + c) = (unsigned char)Kv[k][c];

    /* ---- X : column-major, signed int8 ---------------------------------- */
    for (k = 0; k < N; k++)
        for (r = 0; r < N; r++)
            BYTE(X_ADDR + k*N + r) = (unsigned char)Xv[r][k];

    /* ---- Bias : column-major, signed int32 ------------------------------ */
    for (c = 0; c < N; c++)
        for (r = 0; r < N; r++)
            REG(B_ADDR + c*32 + r*4) = (unsigned int)bval(r,c);

    /* Y is deliberately NOT written or read here, so no stale Y cache line
       can exist before the DMA writes it (cache-coherency safety).          */
    REG(SIG) = 0u;
    cache_maint();                 /* push inputs to SRAM before DMA reads   */

    /* ---- configure DMA -------------------------------------------------- */
    REG(ACC+0x08) = K_ADDR;  REG(ACC+0x0C) = 64u;   REG(ACC+0x10) = 0u;
    REG(ACC+0x18) = X_ADDR;  REG(ACC+0x1C) = 64u;   REG(ACC+0x20) = 0u;
    REG(ACC+0x28) = B_ADDR;  REG(ACC+0x2C) = 256u;  REG(ACC+0x30) = 0u;
    REG(ACC+0x38) = Y_ADDR;  REG(ACC+0x3C) = 256u;
    tx_uart('C');                                   /* registers configured  */

    REG(ACC+0x00) = 1u;                             /* START                 */
    tx_uart('S');                                   /* accelerator started   */

    /* ---- poll S2MM_DONE with a finite timeout --------------------------- */
    guard = 0u;
    while (((REG(ACC+0x34) & 1u) != 1u) && (guard < 1000u)) guard++;

    ok = ((REG(ACC+0x34) & 1u) == 1u);

    /* ---- verify all 64 signed int32 outputs ----------------------------- */
    if (ok) {
        cache_maint();             /* make DMA-written Y visible to the CPU  */
        for (c = 0; c < N && ok; c++) {
            for (r = 0; r < N && ok; r++) {
                expected = bval(r,c);
                for (k = 0; k < N; k++) expected += (int)Xv[r][k] * (int)Kv[k][c];
                if ((int)REG(Y_ADDR + c*32 + r*4) != expected) ok = 0;
            }
        }
    }

    REG(SIG) = ok ? 0xC0DE0008u : 0xBADD0008u;
    cache_maint();
    tx_uart(ok ? 'P' : 'F');

    while (1);
    return 0;
}
