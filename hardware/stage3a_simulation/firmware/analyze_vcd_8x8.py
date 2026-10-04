#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rigorous, repeatable Stage 3A 8x8 VCD analyser.  ASCII-only output.

Usage (native Windows Python, no PYTHONIOENCODING needed):
    python analyze_vcd_8x8.py <vcd> [out.md]

Exit code 0 only if EVERY acceptance check passes; nonzero otherwise.

Rules enforced
  * A transaction exists ONLY when VALID and READY are both high, sampled with
    the state that was stable immediately BEFORE a rising edge of s_axi_aclk.
  * AW and W are independent channels: they are queued and paired.
  * AR and R are paired using ARID/RID.
  * Write data is mapped to memory using the ACCEPTED AW ADDRESS, never the
    W-beat index.
  * The golden reference is computed here from the mathematical definition,
    never from firmware or RTL output.
"""
import sys, io, hashlib
from collections import defaultdict, deque

# ---------------------------------------------------------------- ASCII stdout
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

N = 8
K_ADDR, X_ADDR, B_ADDR, Y_ADDR = 0x20000, 0x20100, 0x20200, 0x20300

# ------------------------------------------------------------ golden reference
Xm  = [[((3*r + 2*k) % 11) - 5 for k in range(N)] for r in range(N)]
Km  = [[((5*k + 3*c) %  9) - 4 for c in range(N)] for k in range(N)]
Bm  = [[2*r - c                for c in range(N)] for r in range(N)]
RAW = [[sum(Xm[r][k]*Km[k][c] for k in range(N)) for c in range(N)] for r in range(N)]
Ym  = [[RAW[r][c] + Bm[r][c]                     for c in range(N)] for r in range(N)]

def u8(v):  return v & 0xFF
def u32(v): return v & 0xFFFFFFFF
def s32(v): return v - (1 << 32) if v & (1 << 31) else v

# golden packed memory images (independent of RTL)
GOLD_RD = {}                                        # addr -> expected 64-bit word
for k in range(N):
    GOLD_RD[K_ADDR + k*8] = sum(u8(Km[k][c]) << (8*c) for c in range(N))
    GOLD_RD[X_ADDR + k*8] = sum(u8(Xm[r][k]) << (8*r) for r in range(N))
for c in range(N):
    for p in range(0, N, 2):
        GOLD_RD[B_ADDR + c*32 + p*4] = (u32(Bm[p+1][c]) << 32) | u32(Bm[p][c])
GOLD_WR = {}                                        # addr -> expected 64-bit word
for c in range(N):
    for p in range(0, N, 2):
        GOLD_WR[Y_ADDR + c*32 + p*4] = (u32(Ym[p+1][c]) << 32) | u32(Ym[p][c])
# 256-bit stream beats, column c, r=0 in the LSB lane
GOLD_RAW256 = [sum(u32(RAW[r][c]) << (32*r) for r in range(N)) for c in range(N)]
GOLD_FIN256 = [sum(u32(Ym [r][c]) << (32*r) for r in range(N)) for c in range(N)]

# --------------------------------------------------------------- VCD front end
def parse(path):
    """Hierarchy-safe parse. Returns (full_paths{name->code}, changes[])."""
    full, changes, scope, t = {}, [], [], 0
    with io.open(path, 'r', encoding='utf-8', errors='replace') as f:
        indefs = True
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            if indefs:
                s = line.strip()
                if s.startswith('$scope'):
                    p = s.split()
                    if len(p) >= 3:
                        scope.append(p[2])
                elif s.startswith('$upscope'):
                    if scope:
                        scope.pop()
                elif s.startswith('$var'):
                    p = s.split()
                    if len(p) >= 5:
                        code, nm = p[3], p[4]
                        full['/'.join(scope + [nm])] = code
                elif s.startswith('$enddefinitions'):
                    indefs = False
                continue
            c0 = line[0]
            if c0 == '#' and line[1:].isdigit():
                t = int(line[1:]); continue
            if c0 in 'bB':
                sp = line.find(' ')
                if sp < 0: continue
                raw, code = line[1:sp], line[sp+1:].strip()
                try:    val = int(raw, 2)
                except ValueError: val = None
            elif c0 in '01xXzZ':
                code, val = line[1:].strip(), (int(c0) if c0 in '01' else None)
            else:
                continue
            changes.append((t, code, val))
    return full, changes

def build_resolver(full, prefer=('u_Accelerator_Top', 'u_accel', 'CONTROLLER')):
    """leaf name -> code, preferring paths inside the accelerator hierarchy."""
    by_leaf = defaultdict(list)
    for path, code in full.items():
        by_leaf[path.split('/')[-1]].append((path, code))
    resolved = {}
    for leaf, lst in by_leaf.items():
        if len(lst) == 1:
            resolved[leaf] = lst[0][1]
        else:
            best = None
            for path, code in lst:
                score = sum(1 for key in prefer if key in path)
                if best is None or score > best[0]:
                    best = (score, code)
            resolved[leaf] = best[1]
    return resolved

# ------------------------------------------------------------------------ main
def main():
    if len(sys.argv) < 2:
        print("usage: python analyze_vcd_8x8.py <vcd> [out.md]"); return 2
    vcd = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    with io.open(vcd, 'rb') as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    full, changes = parse(vcd)
    code_of = build_resolver(full)

    FAIL, NOTE = [], []
    def need(sig):
        if sig not in code_of:
            return False
        return True

    core = ['s_axi_aclk', 's_axi_aresetn', 'uart_out']
    missing_core = [s for s in core if not need(s)]
    stream_sigs = ['s_k_tvalid','s_x_tvalid','s_a_tvalid','m_y_tvalid',
                   'm_axis_s2mm_tvalid']
    missing_streams = [s for s in stream_sigs if not need(s)]

    clk_code = code_of.get('s_axi_aclk')
    st, st_prev = {}, {}
    def g(n):
        c = code_of.get(n)
        return st_prev.get(c) if c else None
    def hs(v, r):
        return g(v) == 1 and g(r) == 1

    cfg_w, cfg_r = [], []
    ar_q  = defaultdict(deque)          # rid -> pending AR
    rd    = []                          # (t, addr, data, rresp, rlast, rid)
    aw_q  = deque()                     # pending AW
    wr    = []                          # (t, addr, data, wstrb, wlast)
    bresp = []
    m_ar_raw, m_aw_raw = [], []
    kx, xx, ax, sa_in, raw_out, fin_out, s2mm_st = [], [], [], [], [], [], []
    desc, status = [], []
    uart_edges, reset_rel, done_time = [], None, None
    prev_clk = 0

    i, L = 0, len(changes)
    while i < L:
        t = changes[i][0]
        st_prev = dict(st)
        while i < L and changes[i][0] == t:
            _, code, val = changes[i]
            if code == code_of.get('uart_out') and val in (0, 1):
                uart_edges.append((t, val))
            if code == code_of.get('s_axi_aresetn') and val == 1 and reset_rel is None:
                reset_rel = t
            st[code] = val
            i += 1
        clk = st.get(clk_code)
        if prev_clk == 0 and clk == 1:
            # --- processor <-> accelerator config -------------------------
            if hs('s_axi_awvalid', 's_axi_awready'):
                cfg_w.append([t, g('s_axi_awaddr'), None])
            if hs('s_axi_wvalid', 's_axi_wready') and cfg_w:
                cfg_w[-1][2] = g('s_axi_wdata')
            if hs('s_axi_arvalid', 's_axi_arready'):
                cfg_r.append((t, g('s_axi_araddr')))
            # --- master read: queue AR by ID, pair R by RID ---------------
            if hs('m_axi_arvalid', 'm_axi_arready'):
                rec = (t, g('m_axi_araddr'), g('m_axi_arlen'),
                       g('m_axi_arsize'), g('m_axi_arburst'), g('m_axi_arid'))
                m_ar_raw.append(rec)
                ar_q[rec[5]].append(rec)
            if hs('m_axi_rvalid', 'm_axi_rready'):
                rid = g('m_axi_rid')
                src = ar_q[rid].popleft() if ar_q[rid] else None
                rd.append((t, (src[1] if src else None), g('m_axi_rdata'),
                           g('m_axi_rresp'), g('m_axi_rlast'), rid,
                           (src[5] if src else None)))
            # --- master write: queue AW, pair W beats ---------------------
            if hs('m_axi_awvalid', 'm_axi_awready'):
                rec = (t, g('m_axi_awaddr'), g('m_axi_awlen'),
                       g('m_axi_awsize'), g('m_axi_awburst'), g('m_axi_awid'))
                m_aw_raw.append(rec); aw_q.append(rec)
            if hs('m_axi_wvalid', 'm_axi_wready'):
                src = aw_q[0] if aw_q else None
                wr.append((t, (src[1] if src else None), g('m_axi_wdata'),
                           g('m_axi_wstrb'), g('m_axi_wlast')))
                if g('m_axi_wlast') == 1 and aw_q:
                    aw_q.popleft()
            if hs('m_axi_bvalid', 'm_axi_bready'):
                bresp.append((t, g('m_axi_bresp'), g('m_axi_bid')))
            # --- internal streams ----------------------------------------
            if hs('s_k_tvalid','s_k_tready'): kx.append((t,g('s_k_tdata'),g('s_k_tlast')))
            if hs('s_x_tvalid','s_x_tready'): xx.append((t,g('s_x_tdata'),g('s_x_tlast')))
            if hs('s_a_tvalid','s_a_tready'): ax.append((t,g('s_a_tdata'),g('s_a_tlast')))
            if hs('s_valid','s_ready'):       sa_in.append((t,g('s_last')))
            if hs('m_valid','m_ready'):       raw_out.append((t,g('m_data'),g('m_last')))
            if hs('m_y_tvalid','m_y_tready'): fin_out.append((t,g('m_y_tdata'),g('m_y_tlast')))
            if hs('m_axis_s2mm_tvalid','m_axis_s2mm_tready'):
                s2mm_st.append((t,g('m_axis_s2mm_tdata'),g('m_axis_s2mm_tlast'),
                                g('m_axis_s2mm_tkeep')))
            # --- descriptors + status ------------------------------------
            for nm in ('mm2s_0','mm2s_1','mm2s_2','s2mm'):
                if hs(nm+'_valid', nm+'_ready'):
                    desc.append((t, nm))
            for nm in ('mm2s_0','mm2s_1','mm2s_2','s2mm'):
                if g(nm+'_status_valid') == 1:
                    status.append((t, nm, g(nm+'_status_error')))
            if g('cfg_S2MM_DONE') == 1 and done_time is None:
                done_time = t
        prev_clk = clk

    # ------------------------------------------------------------- UART decode
    chars, bitper = [], None
    if len(uart_edges) >= 4:
        lv = uart_edges
        d = sorted(b[0]-a[0] for a, b in zip(lv, lv[1:]) if b[0]-a[0] > 0)
        bitper = d[0] if d else 8640000
        def at(ts):
            v = 1
            for te, ve in lv:
                if te <= ts: v = ve
                else: break
            return v
        falls = [lv[j][0] for j in range(1, len(lv)) if lv[j][1]==0 and lv[j-1][1]==1]
        last = -10**18
        for f in falls:
            if f - last < 9*bitper: continue
            bits = [at(f + int((1.5+kk)*bitper)) for kk in range(8)]
            byte = sum(b << kk for kk, b in enumerate(bits))
            chars.append((f, byte, at(f + int(9.5*bitper))))
            last = f
    msg = ''.join(chr(b) for _, b, _ in chars if 32 <= b < 127)

    us = lambda ps: ps / 1.0e6

    # -------------------------------------------------------------- read check
    rd_by_addr = {}
    rd_mism = []
    for (t, a, dv, rr, rl, rid, arid) in rd:
        if a is None:
            rd_mism.append(('unpaired R beat', t, None, None)); continue
        rd_by_addr[a] = dv
    for a, exp in sorted(GOLD_RD.items()):
        if a not in rd_by_addr:
            rd_mism.append(('missing read', a, None, exp))
        elif rd_by_addr[a] != exp:
            rd_mism.append(('bad read data', a, rd_by_addr[a], exp))
    extra_rd = sorted(set(rd_by_addr) - set(GOLD_RD))

    # ------------------------------------------------------------- write check
    wr_by_addr, dup_wr = {}, []
    for (t, a, dv, sb, wl) in wr:
        if a is None:
            continue
        if a in wr_by_addr: dup_wr.append(a)
        wr_by_addr[a] = dv
    got, wr_mism = {}, []
    for a, dv in wr_by_addr.items():
        if not (Y_ADDR <= a < Y_ADDR + 256): continue
        off = a - Y_ADDR
        c, p = off // 32, (off % 32) // 4
        if dv is None: continue
        got[(p,   c)] = s32(dv & 0xFFFFFFFF)
        got[(p+1, c)] = s32((dv >> 32) & 0xFFFFFFFF)
    for c in range(N):
        for r in range(N):
            if (r, c) not in got:
                wr_mism.append((r, c, None, Ym[r][c]))       # missing == mismatch
            elif got[(r, c)] != Ym[r][c]:
                wr_mism.append((r, c, got[(r, c)], Ym[r][c]))
    for a, exp in sorted(GOLD_WR.items()):
        if a not in wr_by_addr:
            wr_mism.append(('missing write', a, None, exp))
    extra_wr = sorted(set(wr_by_addr) - set(GOLD_WR))

    # ------------------------------------------------- stream content checks
    def cmp_stream(lst, gold, label, width_beats):
        bad = []
        if len(lst) != width_beats:
            bad.append('%s: %d beats, expected %d' % (label, len(lst), width_beats))
        for idx, item in enumerate(lst[:width_beats]):
            val = item[1]
            if val is None or val != gold[idx]:
                bad.append('%s beat %d: got %s expected 0x%X'
                           % (label, idx, ('None' if val is None else '0x%X' % val),
                              gold[idx]))
        return bad
    raw_bad = cmp_stream(raw_out, GOLD_RAW256, 'raw m_data', N) if raw_out else \
              ['raw m_data: no beats captured']
    fin_bad = cmp_stream(fin_out, GOLD_FIN256, 'm_y_tdata', N) if fin_out else \
              ['m_y_tdata: NOT PRESENT in VCD']
    s2mm_gold = [GOLD_WR[Y_ADDR + c*32 + p*4] for c in range(N) for p in range(0, N, 2)]
    s2mm_bad = cmp_stream(s2mm_st, s2mm_gold, 'm_axis_s2mm_tdata', 32) if s2mm_st else \
               ['m_axis_s2mm_tdata: NOT PRESENT in VCD']

    # --------------------------------------------------------- protocol checks
    bad_rresp = [t for (t,_,_,rr,_,_,_) in rd if rr not in (0, None)]
    bad_bresp = [t for (t, br, _) in bresp if br not in (0, None)]
    bad_wstrb = [t for (t,_,_,sb,_) in wr if sb not in (0xFF, None)]
    bad_wlast = [t for (t,_,_,_,wl) in wr if wl != 1]
    bad_rlast = [t for (t,_,_,_,rl,_,_) in rd if rl != 1]
    bad_arlen = [r for r in m_ar_raw if r[2] not in (0, None)]
    bad_awlen = [r for r in m_aw_raw if r[2] not in (0, None)]
    bad_arsz  = [r for r in m_ar_raw if r[3] not in (3, None)]
    bad_awsz  = [r for r in m_aw_raw if r[3] not in (3, None)]
    bad_arb   = [r for r in m_ar_raw if r[4] not in (1, None)]
    unpaired_r = [x for x in rd if x[1] is None]
    unpaired_w = [x for x in wr if x[1] is None]
    keep_bad  = [t for (t,_,_,kp) in s2mm_st if kp not in (0xFF, None)]
    tlast_s2mm = [idx for idx,(t,d,tl,kp) in enumerate(s2mm_st)
                  if (tl == 1) != (idx == len(s2mm_st)-1)] if s2mm_st else []
    all_rd = [r[1] for r in m_ar_raw if r[1] is not None]
    all_wr = [r[1] for r in m_aw_raw if r[1] is not None]
    ddr = [a for a in all_rd + all_wr if a >= 0x80000000]
    outside = [a for a in all_rd if not (0x20000 <= a < 0x20300)] + \
              [a for a in all_wr if not (Y_ADDR <= a < Y_ADDR+256)]
    dup_rd = len(all_rd) != len(set(all_rd))
    status_err = [(t, nm, e) for (t, nm, e) in status if e not in (0, None)]
    desc_kinds = sorted(set(nm for _, nm in desc))

    # ------------------------------------------------------ acceptance checks
    def chk(cond, label):
        if not cond: FAIL.append(label)
        return cond
    chk(not missing_core,          "core signals present")
    chk(not missing_streams,       "internal streams present in VCD (%s)" % (missing_streams or 'ok'))
    chk(reset_rel is not None,     "reset released")
    chk(msg == 'BCSP',             "UART sequence exactly BCSP (got '%s')" % msg)
    chk(len(cfg_w) == 12,          "12 configuration writes (got %d)" % len(cfg_w))
    chk(len(got) == 64,            "exactly 64 decoded results (got %d)" % len(got))
    chk(not wr_mism,               "all 64 results match golden (%d issues)" % len(wr_mism))
    chk(not rd_mism,               "all K/X/Bias read data match golden (%d issues)" % len(rd_mism))
    chk(not raw_bad,               "8 raw 256-bit m_data beats match X*K (%d issues)" % len(raw_bad))
    chk(not fin_bad,               "8 m_y_tdata beats match X*K+Bias (%d issues)" % len(fin_bad))
    chk(not s2mm_bad,              "32 S2MM stream beats match Y (%d issues)" % len(s2mm_bad))
    chk(len(sa_in) == 8,           "8 systolic input steps (got %d)" % len(sa_in))
    chk(len(desc_kinds) == 4,      "4 descriptor handshakes (got %s)" % desc_kinds)
    chk(not status_err,            "no DMA status errors (%d)" % len(status_err))
    chk(done_time is not None,     "S2MM_DONE asserted")
    chk(not bad_rresp,             "all RRESP OKAY")
    chk(not bad_bresp,             "all BRESP OKAY")
    chk(not bad_wstrb,             "all WSTRB 0xFF")
    chk(not bad_wlast,             "all WLAST correct")
    chk(not bad_rlast,             "all RLAST correct")
    chk(not (bad_arlen or bad_awlen), "ARLEN/AWLEN = 0")
    chk(not (bad_arsz or bad_awsz),   "ARSIZE/AWSIZE = 3 (8 bytes)")
    chk(not bad_arb,               "AR burst type INCR")
    chk(not unpaired_r,            "every R beat paired to an AR by ID")
    chk(not unpaired_w,            "every W beat paired to an AW")
    chk(not dup_rd,                "no duplicate read addresses")
    chk(not dup_wr,                "no duplicate write addresses")
    chk(not extra_rd,              "no unexpected read addresses")
    chk(not extra_wr,              "no unexpected write addresses")
    chk(not keep_bad,              "S2MM TKEEP all 0xFF")
    chk(not tlast_s2mm,            "S2MM TLAST only on final beat")
    chk(not ddr,                   "no DDR access")
    chk(not outside,               "no access outside intended buffers")

    # ------------------------------------------------------------------ report
    R = []; A = R.append
    A("# Stage 3A 8x8 - VCD Transaction Analysis\n")
    A("**VCD:** `%s`" % vcd)
    A("**SHA-256:** `%s`\n" % sha)
    A("Golden reference computed inside this script from the mathematical")
    A("definition - never from firmware or RTL output.\n")
    A("## Overall result\n")
    A("**%s** - %d acceptance check(s) failed.\n" %
      ("ACCEPTED" if not FAIL else "REJECTED", len(FAIL)))
    if FAIL:
        A("Failed checks:\n")
        for f in FAIL: A("  * %s" % f)
        A("")
    A("## Timeline\n")
    A("| Event | Time (us) |")
    A("|---|---|")
    A("| reset release | %.4f |" % us(reset_rel) if reset_rel else "| reset release | NOT SEEN |")
    for f, b, stop in chars:
        A("| UART `%s` (0x%02X, stop=%s) | %.4f |" %
          (chr(b) if 32 <= b < 127 else '?', b, stop, us(f)))
    if done_time: A("| cfg_S2MM_DONE=1 | %.4f |" % us(done_time))
    A("\nDecoded UART: `%s`  (bit period %.3f us)  -> required `BCSP`: %s\n" %
      (msg, (bitper or 0)/1e6, "YES" if msg == 'BCSP' else "NO"))

    A("## Configuration writes (posedge-sampled)\n")
    A("| # | time (us) | awaddr | off | wdata |")
    A("|---|---|---|---|---|")
    for j, (t, a, dv) in enumerate(cfg_w, 1):
        off = (a - 0x20040000) if a and a >= 0x20040000 else a
        A("| %d | %.4f | 0x%016X | 0x%02X | 0x%08X |" %
          (j, us(t), a or 0, off or 0, u32(dv or 0)))
    A("\nProcessor DONE reads (s_axi AR): %d\n" % len(cfg_r))

    A("## Master reads (AR queued, R paired by ID)\n")
    for nm, lo, hi, exp in (("K", K_ADDR, K_ADDR+64, 8),
                            ("X", X_ADDR, X_ADDR+64, 8),
                            ("Bias", B_ADDR, B_ADDR+256, 32)):
        sel = sorted(a for a in all_rd if lo <= a < hi)
        A("* %-4s beats=%d (exp %d) bytes=%d unique=%d" %
          (nm, len(sel), exp, len(sel)*8, len(set(sel))))
    A("\nRead-data vs golden operands: %s\n" %
      ("ALL MATCH" if not rd_mism else "%d ISSUE(S)" % len(rd_mism)))
    for m in rd_mism[:12]: A("  * %s" % (m,))

    A("\n## Master writes (AW queued, W paired, addr from AW)\n")
    A("AW=%d  W=%d  B=%d  unique write addrs=%d\n" %
      (len(m_aw_raw), len(wr), len(bresp), len(wr_by_addr)))

    A("## Internal streams\n")
    A("| stream | beats | expected |")
    A("|---|---|---|")
    # widths: K/X streams are 64-bit (C*WK, R*WX) -> 64 B / 8 = 8 beats.
    # Bias stream is 256-bit (R*32) after the 64->256 upsize -> 256 B / 32 = 8 beats.
    A("| K (s_k_t*, 64-bit) | %d | 8 |" % len(kx))
    A("| X (s_x_t*, 64-bit) | %d | 8 |" % len(xx))
    A("| Bias (s_a_t*, 256-bit after 64->256 upsize) | %d | 8 |" % len(ax))
    A("| systolic input (s_valid&s_ready) | %d | 8 |" % len(sa_in))
    A("| raw 256-bit (m_valid&m_ready) | %d | 8 |" % len(raw_out))
    A("| final 256-bit (m_y_t*) | %d | 8 |" % len(fin_out))
    A("| S2MM 64-bit (m_axis_s2mm_t*) | %d | 32 |" % len(s2mm_st))
    if missing_streams:
        A("\n**MISSING from VCD: %s**\n" % missing_streams)
    for lbl, bad in (("raw m_data", raw_bad), ("m_y_tdata", fin_bad),
                     ("m_axis_s2mm_tdata", s2mm_bad)):
        A("* %s: %s" % (lbl, "MATCH" if not bad else bad[0]))

    A("\n## Final results (%d/64 decoded)\n" % len(got))
    if len(got) == 64 and not wr_mism:
        A("All 64 decoded outputs are present and match the golden matrix.\n")
    else:
        A("NOT all 64 outputs verified: %d decoded, %d issue(s).\n"
          % (len(got), len(wr_mism)))
    A("| r\\c | " + " | ".join(str(c) for c in range(N)) + " |")
    A("|---|" + "---|"*N)
    for r in range(N):
        A("| **%d** | " % r + " | ".join(
            (str(got[(r, c)]) if (r, c) in got else "MISSING") for c in range(N)) + " |")

    A("\n## Protocol checks\n")
    A("| check | result |")
    A("|---|---|")
    for lbl, bad in (("RRESP OKAY", bad_rresp), ("BRESP OKAY", bad_bresp),
                     ("WSTRB 0xFF", bad_wstrb), ("WLAST", bad_wlast),
                     ("RLAST", bad_rlast), ("ARLEN=0", bad_arlen),
                     ("AWLEN=0", bad_awlen), ("ARSIZE=3", bad_arsz),
                     ("AWSIZE=3", bad_awsz), ("AR burst INCR", bad_arb),
                     ("unpaired R", unpaired_r), ("unpaired W", unpaired_w),
                     ("S2MM TKEEP", keep_bad), ("S2MM TLAST", tlast_s2mm),
                     ("DDR accesses", ddr), ("outside buffers", outside),
                     ("duplicate write addrs", dup_wr),
                     ("unexpected read addrs", extra_rd),
                     ("unexpected write addrs", extra_wr)):
        A("| %s | %s |" % (lbl, "OK" if not bad else "FAIL (%d)" % len(bad)))
    A("| duplicate read addrs | %s |" % ("FAIL" if dup_rd else "OK"))
    A("| descriptor handshakes | %s |" % (", ".join(desc_kinds) or "NONE"))
    A("| DMA status errors | %s |" % ("OK" if not status_err else status_err[:4]))

    txt = "\n".join(R)
    print(txt)
    if out:
        with io.open(out, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(txt + "\n")
        print("\n[written] %s" % out)
    print("\nRESULT: %s (%d failed checks)" %
          ("ACCEPTED" if not FAIL else "REJECTED", len(FAIL)))
    return 0 if not FAIL else 1

if __name__ == '__main__':
    sys.exit(main())
