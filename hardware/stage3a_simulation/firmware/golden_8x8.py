#!/usr/bin/env python3
"""
Independent golden reference for the Stage 3A 8x8 test.
Run:  python golden_8x8.py
Produces: the row-major Y matrix, the column-major stream order, and the exact
packed memory images the firmware must write.  Nothing here reads RTL or
firmware output - it is a pure mathematical reference.

Definitions (given):
    X[r][k] = ((3r + 2k) % 11) - 5      signed int8
    K[k][c] = ((5k + 3c) %  9) - 4      signed int8
    B[r][c] = 2r - c                    signed int32
    Y = X @ K + B                       signed int32
"""
N = 8

X = [[((3*r + 2*k) % 11) - 5 for k in range(N)] for r in range(N)]
K = [[((5*k + 3*c) %  9) - 4 for c in range(N)] for k in range(N)]
B = [[2*r - c                for c in range(N)] for r in range(N)]
Y = [[sum(X[r][k]*K[k][c] for k in range(N)) + B[r][c]
      for c in range(N)] for r in range(N)]

def s8(v):  return v & 0xFF          # two's-complement byte
def s32(v): return v & 0xFFFFFFFF    # two's-complement word

print("X (row-major) ="); [print("   ", row) for row in X]
print("K (row-major) ="); [print("   ", row) for row in K]
print("B (row-major) ="); [print("   ", row) for row in B]
print("\nY = X*K + B   (row-major display order):")
for row in Y:
    print("   [" + ", ".join(f"{v:4d}" for v in row) + "]")

# ---- expected stream / memory order (column-major), matching the verified 2x2 --
col_major = [Y[r][c] for c in range(N) for r in range(N)]
print("\nY column-major stream order (64 values, the order written to memory):")
for i in range(0, 64, 8):
    print("   " + ", ".join(f"{v:4d}" for v in col_major[i:i+8]))

# ============================================================================
# Packed memory images.  Layout derived from the VERIFIED 2x2 behaviour:
#   K row-major    : K[k][c]  at K_ADDR + k*8 + c        (beat k = row k of K)
#   X column-major : X[r][k]  at X_ADDR + k*8 + r        (beat k = column k of X)
#   B column-major : B[r][c]  at B_ADDR + c*32 + r*4     (beat c = column c)
#   Y column-major : Y[r][c]  at Y_ADDR + c*32 + r*4     (beat c = column c)
# ============================================================================
print("\n--- K image: 8 x 64-bit beats @0x20000 (byte c within beat k) ---")
for k in range(N):
    beat = sum(s8(K[k][c]) << (8*c) for c in range(N))
    print(f"   0x{0x20000+k*8:05X} : 0x{beat:016X}   row {k} of K = {K[k]}")

print("\n--- X image: 8 x 64-bit beats @0x20100 (byte r within beat k) ---")
for k in range(N):
    beat = sum(s8(X[r][k]) << (8*r) for r in range(N))
    col = [X[r][k] for r in range(N)]
    print(f"   0x{0x20100+k*8:05X} : 0x{beat:016X}   column {k} of X = {col}")

print("\n--- Bias image: 64 x 32-bit words @0x20200 (column-major) ---")
for c in range(N):
    words = [s32(B[r][c]) for r in range(N)]
    print(f"   col {c} @0x{0x20200+c*32:05X} : " + " ".join(f"{w:08X}" for w in words))

print("\n--- Expected Y image: 64 x 32-bit words @0x20300 (column-major) ---")
for c in range(N):
    words = [s32(Y[r][c]) for r in range(N)]
    print(f"   col {c} @0x{0x20300+c*32:05X} : " + " ".join(f"{w:08X}" for w in words))

print("\n--- Expected 64-bit AXI write beats @0x20300 (32 beats, little-endian) ---")
for c in range(N):
    for p in range(0, N, 2):          # two 32-bit words per 64-bit beat
        lo, hi = s32(Y[p][c]), s32(Y[p+1][c])
        addr = 0x20300 + c*32 + p*4
        print(f"   0x{addr:05X} : 0x{hi:08X}_{lo:08X}   -> {Y[p][c]}, {Y[p+1][c]}")

# ---- self-check against the independently supplied reference -----------------
REF = [[ 27,-28, 16, 24,-31, 13, 21,-34],
       [ 18,-13,-17, 15,-16,-20, 12,-19],
       [ -2, 24,  5, -5, 21,  2, -8, 18],
       [-22, -5, -6,-25, -8, -9,-28,-11],
       [  2, 10, 27, -1,  7, 24, -4,  4],
       [ 26, -8, -6, 23,-11, -9, 20,-14],
       [ 39, 29,-17, 36, 26,-20, 33, 23],
       [ -3,-22, 49, -6,-25, 46, -9,-28]]
print("\nSELF-CHECK vs supplied reference matrix:",
      "MATCH" if Y == REF else "MISMATCH")
