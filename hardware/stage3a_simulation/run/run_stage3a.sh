#!/usr/bin/env bash
#=============================================================================
# run_stage3a.sh - DVCon Stage 3A 8x8 evaluation flow (Linux)
#
#   ./RUN/run_stage3a.sh                 interactive menu
#   ./RUN/run_stage3a.sh --simulation
#   ./RUN/run_stage3a.sh --synthesis
#   ./RUN/run_stage3a.sh --all
#   ./RUN/run_stage3a.sh --check
#   ./RUN/run_stage3a.sh --help
#
#   ./RUN/run_stage3a.sh --rebuild-firmware   (OPTIONAL developer mode, needs
#                                              Docker; NOT part of evaluation)
#
# DOCKER IS NOT REQUIRED for check / simulation / synthesis / all.
# The evaluation flow uses the submitted, verified 64-bit boot-ROM MIF and
# never rebuilds, copies or modifies firmware.
#
# Stage 3A scope only: no implementation, no bitstream, no FPGA programming,
# no QuestaSim submenu. The organizer's RUN/run.sh is left untouched.
#=============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

APP="$PROJECT_ROOT/Application/stage3a_test"
OUT="$PROJECT_ROOT/GEN_BIT_OUT"
TCL="$PROJECT_ROOT/TCL"
ROMDIR="$PROJECT_ROOT/DVCon_SoC_SRC/MEMORY_IP/rom_32KB_axi"
LOGDIR="$OUT/logs"
PART="xc7k325tffg900-2"
GUI="${GUI:-1}"          # 1 = launch Vivado GUI (default), 0 = -mode batch

CANON_MIF="$APP/build/accelarator_test.hex.mif"
ROM_MIF="$ROMDIR/rom_32KB_axi.mif"

DOCKER_IMAGE="dvcon_2:latest"
DOCKER_TAR="$PROJECT_ROOT/Application/.docker_scripts/docker.tar"

C_G=$'\033[0;32m'; C_R=$'\033[0;31m'; C_Y=$'\033[0;33m'; C_N=$'\033[0m'
ok(){   printf '%s[ OK ]%s %s\n' "$C_G" "$C_N" "$*"; }
bad(){  printf '%s[FAIL]%s %s\n' "$C_R" "$C_N" "$*"; }
warn(){ printf '%s[WARN]%s %s\n' "$C_Y" "$C_N" "$*"; }
die(){  bad "$*"; exit 1; }

sha(){ sha256sum "$1" | awk '{print $1}'; }

have_display(){
    [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]
}

# $1 = tcl script ; $2 = log
run_vivado(){
    local tcl="$1" log="$2"
    export STAGE3A_PYTHON="$(command -v python3 || command -v python)"
    if [[ "$GUI" == "1" ]]; then
        if ! have_display; then
            die "Vivado GUI is unavailable in this Linux session. Use --batch."
        fi
        unset STAGE3A_BATCH
        ( cd "$PROJECT_ROOT" && vivado -nojournal -log "$log" -source "$tcl" )
    else
        export STAGE3A_BATCH=1
        ( cd "$PROJECT_ROOT" && vivado -mode batch -nojournal -log "$log" -source "$tcl" )
    fi
}

# Does the log contain a marker as REAL OUTPUT, rather than as Vivado's echo of
# the script source?
#
# 'vivado -source' echoes each command into the log prefixed with '#', and it
# echoes an entire if/else block INCLUDING THE BRANCH THAT DID NOT RUN. So a
# failed simulation still contains, verbatim:
#       #     puts " STAGE 3A 8X8 SIMULATION PASS"
#       #     puts "STAGE3A_SIM: DONE"
# A plain substring search therefore reports PASS on a FAILING run. Real output
# lines never start with '#', so every marker test must be anchored.
log_marker(){   # log_marker <log> <extended-regex>
    grep -qE "$2" "$1" 2>/dev/null
}

# Decide whether a Vivado invocation succeeded.
#
# The exit code alone is NOT a reliable verdict in GUI mode. When the design is
# left open for review (the intended judge experience) the user closes the
# window when finished, and Vivado can return a non-zero exit code for that even
# though every step completed. The authority is therefore the completion marker
# the Tcl prints into the log; the exit code only sharpens the message when the
# marker is absent, and is enforced strictly in batch mode.
assert_vivado_outcome(){
    local rc="$1" log="$2" marker_re="$3" what="$4" marker="${5:-$3}"
    if ! log_marker "$log" "$marker_re"; then
        if [[ "$rc" -ne 0 ]]; then
            die "Vivado $what failed (exit code $rc) - see $log"
        fi
        die "$what did not reach its completion marker '$marker' - see $log"
    fi
    if [[ "$GUI" != "1" && "$rc" -ne 0 ]]; then
        die "Vivado $what printed '$marker' but exited with code $rc - see $log"
    fi
    if [[ "$GUI" == "1" && "$rc" -ne 0 ]]; then
        warn "Vivado exited with code $rc. The run had already completed ('$marker' is in the log), so this is just how Vivado reports the GUI window being closed. Not a failure."
    fi
}

viv_version(){
    if command -v vivado >/dev/null 2>&1; then
        vivado -version 2>/dev/null | head -1 | sed 's/^ *//' || echo "unknown"
    else
        echo "NOT FOUND"
    fi
}

# 64-bit MIF: every data line must be exactly 64 binary characters
mif_format_ok(){
    local first
    first="$(head -1 "$1" | tr -d '\r' | sed 's/[[:space:]]*$//')"
    [[ ${#first} -eq 64 && "$first" =~ ^[01]+$ ]]
}

#---------------------------------------------------------------- preflight ---
# $1 = "deep" to also query Vivado for target-part availability
preflight(){
    local deep="${1:-}"
    local rc=0
    echo "------------------------------------------------------------"
    echo " Preflight check  (Docker is NOT required)"
    echo "------------------------------------------------------------"

    # --- Vivado ---
    local v; v="$(viv_version)"
    if [[ "$v" == "NOT FOUND" ]]; then bad "Vivado not on PATH"; rc=1
    else ok "Vivado detected: $v"; fi

    # --- Python ---
    if command -v python3 >/dev/null 2>&1; then ok "Python detected: $(python3 --version 2>&1)"
    elif command -v python >/dev/null 2>&1; then ok "Python detected: $(python --version 2>&1)"
    else bad "Python not found (required by the strict VCD analyzer)"; rc=1; fi

    # --- required project files ---
    local missing=0
    for f in \
        "$PROJECT_ROOT/DVCon_SoC_SRC/AS1061_SYSTEM/AS1061_SYSTEM_TOP.v" \
        "$PROJECT_ROOT/DVCon_SoC_SRC/TOP/Top.vhd" \
        "$PROJECT_ROOT/DVCon_SoC_SRC/TB/test_bench.vhd" \
        "$PROJECT_ROOT/DVCon_SoC_SRC/ACCELERATOR_IP/Accelerator_Top.v" \
        "$PROJECT_ROOT/DVCon_SoC_SRC/ACCELERATOR_IP/rtl/sys/config.svh" \
        "$PROJECT_ROOT/DVCon_SoC_SRC/ACCELERATOR_IP/rtl/sys/top_axi_int.v" \
        "$PROJECT_ROOT/DVCon_SoC_SRC/ACCELERATOR_IP/rtl/sys/dma_controller.sv" \
        "$APP/stage3a_main.c" "$APP/Makefile" "$APP/mbl.lds" "$APP/crt.S" \
        "$APP/analyze_vcd_8x8.py" "$APP/golden_8x8.py" \
        "$TCL/DVCon_SIM_stage3a.tcl" "$TCL/DVCon_SYN_stage3a.tcl" ; do
        [[ -f "$f" ]] || { bad "MISSING ${f#"$PROJECT_ROOT"/}"; missing=1; rc=1; }
    done
    [[ $missing -eq 0 ]] && ok "required Stage 3A files found"

    [[ -f "$ROMDIR/rom_32KB_axi.xci" ]] && ok "ROM XCI found" \
        || { bad "MISSING ROM XCI"; rc=1; }

    # --- the submitted program image ---
    if [[ -s "$CANON_MIF" ]]; then ok "canonical firmware MIF found"
    else bad "canonical firmware MIF missing: ${CANON_MIF#"$PROJECT_ROOT"/}"; rc=1; fi
    if [[ -s "$ROM_MIF" ]]; then ok "installed ROM MIF found"
    else bad "installed ROM MIF missing: ${ROM_MIF#"$PROJECT_ROOT"/}"; rc=1; fi

    if [[ -s "$CANON_MIF" && -s "$ROM_MIF" ]]; then
        local ch ih; ch="$(sha "$CANON_MIF")"; ih="$(sha "$ROM_MIF")"
        if [[ "$ch" == "$ih" ]]; then
            ok "MIF hashes match  sha256=${ch:0:16}..."
        else
            bad "Installed ROM MIF does not match the submitted canonical Stage 3A firmware MIF."
            echo "      canonical : $ch"
            echo "      installed : $ih"
            rc=1
        fi
        if mif_format_ok "$ROM_MIF"; then ok "ROM MIF is 64-bit binary format"
        else bad "ROM MIF is not the expected 64-bit binary format"; rc=1; fi
    fi

    # --- writable outputs ---
    mkdir -p "$OUT" "$LOGDIR" "$PROJECT_ROOT/VIVADO_PROJECT" 2>/dev/null || true
    [[ -w "$OUT" ]] && ok "GEN_BIT_OUT writable" || { bad "GEN_BIT_OUT not writable"; rc=1; }
    [[ -w "$PROJECT_ROOT/VIVADO_PROJECT" ]] && ok "VIVADO_PROJECT writable" \
        || { bad "VIVADO_PROJECT not writable"; rc=1; }

    # --- no hard-coded paths in our automation ---
    if grep -RqE '(^|[^A-Za-z0-9_])[A-Za-z]:[/\\]|/home/[a-z]+/|/mnt/[a-z]/' \
         "$TCL/DVCon_SIM_stage3a.tcl" "$TCL/DVCon_SYN_stage3a.tcl" 2>/dev/null; then
        bad "hard-coded absolute path found in Stage 3A Tcl"; rc=1
    else ok "no hard-coded paths in Stage 3A Tcl"; fi

    # --- optional deep check: target part availability ---
    if [[ "$deep" == "deep" && "$v" != "NOT FOUND" ]]; then
        local tmp; tmp="$(mktemp -d)"
        printf 'if {[llength [get_parts -quiet %s]]>0} {puts "PART_OK"} else {puts "PART_MISSING"}\nexit 0\n' \
            "$PART" > "$tmp/p.tcl"
        if vivado -mode batch -nolog -nojournal -source "$tmp/p.tcl" 2>/dev/null | grep -q PART_OK; then
            ok "target part $PART available"
        else
            bad "target part $PART NOT available in this Vivado"; rc=1
        fi
        rm -rf "$tmp"
    else
        [[ "$v" != "NOT FOUND" ]] && ok "target part validated at run time by the Tcl scripts"
    fi

    echo "------------------------------------------------------------"
    [[ $rc -eq 0 ]] && ok "PREFLIGHT PASS" || bad "PREFLIGHT FAIL"
    return $rc
}

#---------------------------------------------------------------- operations ---
do_simulation(){
    preflight || die "preflight failed"
    mkdir -p "$LOGDIR"
    local log="$LOGDIR/stage3a_sim_$(date +%Y%m%d-%H%M%S).log"
    echo "------------------------------------------------------------"
    echo " Vivado XSim ($([[ "$GUI" == "1" ]] && echo GUI || echo batch))  ->  ${log#"$PROJECT_ROOT"/}"
    [[ "$GUI" == "1" ]] && \
        echo " The waveform stays open when the run finishes. Close Vivado to return here."
    echo "------------------------------------------------------------"
    local rc=0
    run_vivado "$TCL/DVCon_SIM_stage3a.tcl" "$log" || rc=$?
    assert_vivado_outcome "$rc" "$log" "^STAGE3A_SIM: DONE" "simulation" "STAGE3A_SIM: DONE"
    ok "simulation completed"

    # XSim working-copy MIF integrity
    local ch; ch="$(sha "$ROM_MIF")"
    local xm; xm="$(find "$PROJECT_ROOT/VIVADO_PROJECT" -name 'rom_32KB_axi.mif' \
                    -newermt '-2 hours' 2>/dev/null | head -1 || true)"
    if [[ -n "$xm" ]]; then
        [[ "$(sha "$xm")" == "$ch" ]] && ok "XSim working-directory MIF hash matches" \
                                       || die "XSim working-directory MIF hash MISMATCH"
    else
        warn "no XSim working copy of the MIF located (IP loaded it directly)"
    fi

    # The Tcl runs the strict analyzer itself (so it works while the GUI stays
    # open). In batch we re-verify here to gate the wrapper exit code.
    if [[ "$GUI" != "1" ]]; then
        local PY; PY="$(command -v python3 || command -v python)"
        echo "--- VCD analysis (strict) ---"
        set +e
        "$PY" "$APP/analyze_vcd_8x8.py" "$OUT/stage3a_8x8_accelerator.vcd" \
              "$OUT/stage3a_8x8_vcd_analysis.md" | tail -22
        local arc=${PIPESTATUS[0]}
        set -e
        (( arc == 0 )) || die "VCD analysis REJECTED (exit $arc)"
        ok "VCD analysis ACCEPTED (0 failed checks)"
    else
        # Anchored: see log_marker. An unanchored search matches Vivado's echo
        # of the untaken PASS branch and would green-light a failed run.
        log_marker "$log" "^[[:space:]]*STAGE 3A 8X8 SIMULATION PASS[[:space:]]*$" \
            || die "Tcl did not report SIMULATION PASS (see $log)"
        log_marker "$log" "^[[:space:]]*STAGE 3A 8X8 SIMULATION FAIL[[:space:]]*$" \
            && die "Tcl reported SIMULATION FAIL (see $log)"
        ok "VCD analysis ACCEPTED (reported by Vivado Tcl console)"
    fi

    { echo "stage3a_simulation"
      echo "date       : $(date -Is)"
      echo "vivado     : $(viv_version)"
      echo "rom_mif    : $ch"
      echo "vcd        : $(sha "$OUT/stage3a_8x8_accelerator.vcd")"
      echo "analysis   : $(sha "$OUT/stage3a_8x8_vcd_analysis.md")"
    } > "$OUT/stage3a_sim_manifest.txt"
    ok "manifest: GEN_BIT_OUT/stage3a_sim_manifest.txt"
    echo; ok "SIMULATION PASS"
}

do_synthesis(){
    preflight || die "preflight failed"
    mkdir -p "$LOGDIR"
    local log="$LOGDIR/stage3a_syn_$(date +%Y%m%d-%H%M%S).log"
    echo "------------------------------------------------------------"
    echo " Vivado synthesis ($([[ "$GUI" == "1" ]] && echo GUI || echo batch))  ->  ${log#"$PROJECT_ROOT"/}"
    [[ "$GUI" == "1" ]] && \
        echo " The synthesized design stays open when the run finishes. Close Vivado to return here."
    echo "------------------------------------------------------------"
    local rc=0
    run_vivado "$TCL/DVCon_SYN_stage3a.tcl" "$log" || rc=$?
    assert_vivado_outcome "$rc" "$log" "^STAGE3A_SYN: DONE" "synthesis" "STAGE3A_SYN: DONE"
    log_marker "$log" "^[[:space:]]*STAGE 3A 8X8 SYNTHESIS FAIL[[:space:]]*$" \
        && die "Tcl reported SYNTHESIS FAIL (see $log)"
    for r in total hier accelerator; do
        [[ -s "$OUT/stage3a_8x8_utilization_$r.rpt" ]] \
            && ok "report stage3a_8x8_utilization_$r.rpt" \
            || die "missing utilization report: $r"
    done
    echo "--- accelerator utilization ---"
    grep -E "Slice LUTs\*|Slice Registers|Block RAM Tile|DSPs " \
        "$OUT/stage3a_8x8_utilization_accelerator.rpt" | head -5 || true
    { echo "stage3a_synthesis"
      echo "date    : $(date -Is)"
      echo "vivado  : $(viv_version)"
      echo "rom_mif : $(sha "$ROM_MIF")"
    } > "$OUT/stage3a_syn_manifest.txt"
    echo; ok "SYNTHESIS PASS"
}

do_all(){ do_simulation; echo; do_synthesis; echo; ok "COMPLETE FLOW PASS"; }

#-------------------------------------- OPTIONAL developer-only firmware build --
do_rebuild_firmware(){
    echo "------------------------------------------------------------"
    echo " OPTIONAL developer firmware rebuild (requires Docker)"
    echo " Not part of the judges' evaluation flow."
    echo "------------------------------------------------------------"
    command -v docker >/dev/null 2>&1 || die "docker CLI not found"
    docker info >/dev/null 2>&1 || die "docker daemon not responding (start Docker and retry)"
    if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        [[ -f "$DOCKER_TAR" ]] || die "image $DOCKER_IMAGE absent and docker.tar not found"
        echo "loading $DOCKER_IMAGE from docker.tar ..."
        docker load -i "$DOCKER_TAR" >/dev/null || die "docker load failed"
    fi
    docker run --rm -v "$PROJECT_ROOT":/work -w /work/Application/stage3a_test \
        "$DOCKER_IMAGE" bash -lc \
        'export PATH=$PATH:/work/Application/toolchain-bare/bin:/work/Application/Util; \
         make clean >/dev/null 2>&1 || true; make' || die "firmware build failed"
    ok "firmware rebuilt"
    if [[ -s "$CANON_MIF" && -s "$ROM_MIF" ]]; then
        local a b; a="$(sha "$CANON_MIF")"; b="$(sha "$ROM_MIF")"
        echo "  rebuilt canonical MIF : $a"
        echo "  installed ROM MIF     : $b"
        [[ "$a" == "$b" ]] && ok "rebuilt MIF matches the installed ROM MIF" \
            || warn "rebuilt MIF DIFFERS from the installed ROM MIF (not auto-installing)"
    fi
    echo "NOTE: the rebuilt MIF is NOT installed automatically."
}

#---------------------------------------------------------------------- menu ---
show_menu(){
    cat <<EOF

================================================
       DVCon Stage 3A 8x8 Evaluation Flow
================================================
Project root : $PROJECT_ROOT
Vivado       : $(viv_version)
Target       : $PART
================================================
1) Functional Simulation - Vivado XSim GUI
2) RTL Synthesis - AMD Vivado GUI
3) Exit
================================================
EOF
}

usage(){
    cat <<EOF
DVCon Stage 3A 8x8 evaluation flow  (Docker NOT required)

  ./RUN/run_stage3a.sh                interactive menu
  ./RUN/run_stage3a.sh --simulation   XSim + strict VCD analysis
  ./RUN/run_stage3a.sh --synthesis    RTL synthesis + utilization reports
  ./RUN/run_stage3a.sh --all          simulation then synthesis
  ./RUN/run_stage3a.sh --check        preflight only (deep part check)
  ./RUN/run_stage3a.sh --help         this text

  ./RUN/run_stage3a.sh --rebuild-firmware
        OPTIONAL developer mode. Requires Docker. Rebuilds the bare-metal
        firmware from source and compares the result with the installed ROM
        MIF. Does NOT install it. Not needed for evaluation.

Stage 3A scope only: no implementation, no bitstream, no FPGA programming.
Exit code 0 only on genuine success.
EOF
}

main(){
    if [[ $# -gt 0 ]]; then
        local mode="$1"; shift || true
        for a in "$@"; do [[ "$a" == "--batch" ]] && GUI=0; done
        case "$mode" in
            --simulation|-s)     do_simulation ;;
            --synthesis|-y)      do_synthesis ;;
            --all|-a)            GUI=0; do_all ;;   # complete flow is batch-only
            --check|-c)          preflight deep ;;
            --rebuild-firmware)  do_rebuild_firmware ;;
            --help|-h)           usage ;;
            *) echo "unknown option: $mode"; usage; exit 2 ;;
        esac
        exit $?
    fi
    while true; do
        show_menu
        read -r -p "Select an option [1-3]: " choice
        case "${choice:-}" in
            1) GUI=1 do_simulation ;;
            2) GUI=1 do_synthesis ;;
            3) echo "Exiting."; exit 0 ;;
            *) echo "Invalid selection." ;;
        esac
        echo
    done
}
main "$@"
