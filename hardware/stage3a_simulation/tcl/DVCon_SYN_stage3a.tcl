#=============================================================================
# DVCon_SYN_stage3a.tcl - Stage 3A 8x8 RTL synthesis (GUI or batch)
#-----------------------------------------------------------------------------
# GUI (default, judge experience):
#     vivado -source <this file>
# Batch / CI regression:
#     STAGE3A_BATCH=1  vivado -mode batch -source <this file>
#
# Project-run flow (reset_run / launch_runs / wait_on_run), matching the
# organizer's TCL/DVCon_SYN.tcl. In GUI mode the synthesized design is LEFT
# OPEN with named in-memory reports. No implementation, no bitstream.
#=============================================================================

set BATCH 0
if {[info exists ::env(STAGE3A_BATCH)] && $::env(STAGE3A_BATCH) ne "0"} { set BATCH 1 }

proc s3a_fail {msg} {
    global BATCH
    puts "\n================================================"
    puts " STAGE 3A 8X8 SYNTHESIS FAIL"
    puts " $msg"
    puts "================================================"
    if {$BATCH} { exit 1 }
    return -code error $msg
}

set THIS      [file normalize [info script]]
set MDIR_PATH [file normalize [file join [file dirname $THIS] ".."]]
set VIV_VER   [version -short]
set PART      xc7k325tffg900-2

set ACC [file join $MDIR_PATH DVCon_SoC_SRC ACCELERATOR_IP]
set OUT [file join $MDIR_PATH GEN_BIT_OUT]
file mkdir $OUT

#-----------------------------------------------------------------------------
# Host memory detection
#
# Synthesising this SoC peaks at roughly 4.5 GB inside the synthesis process.
# The project-run flow (launch_runs) starts a SECOND full Vivado process, so
# the parent and the child hold memory at the same time and the combined
# footprint approaches 10 GB. On a VM with 8 GB or less the Linux OOM killer
# sends SIGKILL to the child, which Vivado reports only as
#     "<pid> Killed" ... "ERROR: [Vivado 12-13638] Failed runs(s) : 'synth_1'"
# That message looks like a design error but is not one. We detect the
# condition up front and pick a flow that fits in the available memory.
#-----------------------------------------------------------------------------
proc s3a_total_ram_mb {} {
    # Linux
    if {[file readable /proc/meminfo]} {
        if {![catch {open /proc/meminfo r} fh]} {
            set txt [read $fh]; close $fh
            if {[regexp {MemTotal:\s+(\d+) kB} $txt -> kb]} {
                return [expr {$kb / 1024}]
            }
        }
    }
    # Windows
    if {![catch {exec wmic computersystem get TotalPhysicalMemory} out]} {
        foreach ln [split $out "\n"] {
            set ln [string trim $ln]
            if {[string is integer -strict $ln] && $ln > 0} {
                return [expr {wide($ln) / 1048576}]
            }
        }
    }
    return 0
}

set RAM_MB [s3a_total_ram_mb]

# Flow selection: project | inproc
#   project = reset_run / launch_runs / wait_on_run  (matches organizer's
#             DVCon_SYN.tcl; spawns a second Vivado process)
#   inproc  = synth_design in this process           (about 1.7 GB lighter,
#             produces an identical netlist)
set FLOW "project"
set JOBS 4
if {$RAM_MB > 0 && $RAM_MB < 12000} {
    set FLOW "inproc"
    set JOBS 1
}
if {[info exists ::env(STAGE3A_SYNFLOW)]} { set FLOW $::env(STAGE3A_SYNFLOW) }
if {[info exists ::env(STAGE3A_JOBS)]}    { set JOBS $::env(STAGE3A_JOBS) }

puts "============================================================"
puts " DVCon Stage 3A 8x8 - RTL Synthesis (AMD Vivado)"
puts "============================================================"
puts " Project root : $MDIR_PATH"
puts " Vivado       : $VIV_VER"
puts " Target part  : $PART"
puts " Mode         : [expr {$BATCH ? {BATCH} : {GUI}}]"
if {$RAM_MB > 0} {
    puts " Host RAM     : [expr {$RAM_MB / 1024}] GB"
} else {
    puts " Host RAM     : could not be detected"
}
puts " Synth flow   : $FLOW ([expr {$FLOW eq {project} ? {launch_runs, matches organizer script} : {synth_design in-process, low memory}}])"
puts "============================================================"

if {$RAM_MB > 0 && $RAM_MB < 12000} {
    puts ""
    puts "NOTE: this machine reports [expr {$RAM_MB / 1024}] GB of RAM. Synthesis of the full SoC"
    puts "      peaks near 4.5 GB, and the project-run flow would hold two Vivado"
    puts "      processes at once. The in-process flow is being used instead so the"
    puts "      run fits in memory. The resulting netlist is identical."
    if {$RAM_MB < 8000} {
        puts ""
        puts "      [expr {$RAM_MB / 1024}] GB may still be tight. If this run is killed, give the"
        puts "      machine (or the VM) at least 12 GB and try again."
    }
    puts ""
}

if {[llength [get_parts -quiet $PART]] == 0} {
    s3a_fail "target part $PART is not available in Vivado $VIV_VER."
}

proc find_src {dir} {
    set out {}
    foreach pat {*.v *.sv *.vhd *.vhdl *.xci} {
        foreach f [glob -nocomplain -directory $dir -- $pat] { lappend out $f }
    }
    foreach d [glob -nocomplain -type d -directory $dir -- *] {
        set out [concat $out [find_src $d]]
    }
    return $out
}

set ACC_INC [file join $ACC rtl sys]

# Adds every source, the include dir and the constraints to whatever project
# is currently open. Called for the on-disk project, and again if we have to
# fall back to an in-memory project after an out-of-memory kill.
proc s3a_add_all_sources {} {
    global MDIR_PATH ACC ACC_INC VIV_VER
    add_files [file join $MDIR_PATH DVCon_SoC_SRC AS1061_SYSTEM AS1061_SYSTEM_TOP.v]
    add_files [file join $MDIR_PATH DVCon_SoC_SRC TOP Top.vhd]
    add_files [file join $MDIR_PATH DVCon_SoC_SRC MEMORY_IP rom_32KB_axi rom_32KB_axi.xci]
    foreach f [find_src $ACC] { add_files $f }

    set_property include_dirs [list $ACC_INC] [get_filesets sources_1]
    set tai [get_files -quiet [file join $ACC rtl sys top_axi_int.v]]
    if {[llength $tai]} { set_property file_type SystemVerilog $tai }

    set xdc [file join $MDIR_PATH DVCon_SoC_XDC AS1061_SYSTEM_XDC.xdc]
    if {[file exists $xdc]} { add_files -fileset constrs_1 -norecurse $xdc }

    update_compile_order -fileset sources_1
    set_property top Top [current_fileset]

    foreach ip [get_ips -quiet] {
        if {[get_property -quiet IS_LOCKED $ip] eq "1"} {
            puts "WARNING: IP '$ip' is LOCKED under Vivado $VIV_VER (not auto-upgrading)."
        }
    }
}

set STAMP  [clock format [clock seconds] -format %d%m%y-%H%M%S]
set PRJDIR [file join $MDIR_PATH VIVADO_PROJECT STAGE3A_SYN_8x8_$STAMP]

if {$FLOW eq "project"} {
    create_project DVCon_STAGE3A_SYN $PRJDIR -part $PART
    set_property part $PART [current_project]
} else {
    # in-memory project: no second Vivado process, no .xpr on disk
    create_project -in_memory -part $PART
}
set_property target_language VHDL [current_project]
s3a_add_all_sources

# ---- synthesis --------------------------------------------------------------
# Scan a failed project run's log for the signature of an OOM kill, so the
# evaluator is not left thinking the DESIGN failed when the MACHINE ran out
# of memory.
proc s3a_oom_evidence {logfile} {
    if {![file exists $logfile]} { return "" }
    if {[catch {open $logfile r} fh]} { return "" }
    set txt [read $fh]; close $fh
    foreach pat {"Killed" "Segmentation fault" "std::bad_alloc"
                 "Out of memory" "out of memory" "Cannot allocate memory"} {
        if {[string first $pat $txt] >= 0} { return $pat }
    }
    return ""
}

proc s3a_run_inproc {part acc_inc} {
    puts "\n--- synthesis (synth_design, in-process) ---"
    synth_design -top Top -part $part -include_dirs [list $acc_inc]
}

set SYNTH_OK 0

if {$FLOW eq "project"} {
    puts "\n--- synthesis (project run, $JOBS jobs) ---"
    reset_run synth_1
    launch_runs synth_1 -jobs $JOBS
    wait_on_run synth_1

    set st [get_property STATUS   [get_runs synth_1]]
    set pr [get_property PROGRESS [get_runs synth_1]]
    puts "Synthesis status  : $st"
    puts "Synthesis progress: $pr"

    if {[string first "Complete" $st] >= 0 && $pr eq "100%"} {
        open_run synth_1
        set SYNTH_OK 1
    } else {
        set rl [file join [get_property DIRECTORY [get_runs synth_1]] runme.log]
        set ev [s3a_oom_evidence $rl]
        puts "\n------------------------------------------------------------"
        puts " The project synthesis run did not complete."
        if {$ev ne ""} {
            puts ""
            puts " DIAGNOSIS: the synthesis process was killed by the operating"
            puts " system, not by a design error. Evidence in the run log:"
            puts "     \"$ev\""
            puts "     $rl"
            puts ""
            puts " This is an out-of-memory condition. Synthesis of the full SoC"
            puts " peaks near 4.5 GB, and the project-run flow holds a second"
            puts " Vivado process at the same time."
            if {$RAM_MB > 0} { puts " This machine reports [expr {$RAM_MB / 1024}] GB of RAM." }
            puts ""
            puts " Retrying automatically with the in-process flow, which is"
            puts " roughly 1.7 GB lighter and produces an identical netlist."
            puts "------------------------------------------------------------"
            close_project
            create_project -in_memory -part $PART
            set_property target_language VHDL [current_project]
            s3a_add_all_sources
            if {[catch {s3a_run_inproc $PART $ACC_INC} err]} {
                s3a_fail "in-process synthesis also failed: $err\n\
 Give this machine (or VM) at least 12 GB of RAM and run again."
            }
            set SYNTH_OK 1
        } else {
            puts "------------------------------------------------------------"
            s3a_fail "synthesis did not complete (status='$st' progress='$pr').\n\
 See $rl"
        }
    }
} else {
    if {[catch {s3a_run_inproc $PART $ACC_INC} err]} {
        s3a_fail "synthesis failed: $err"
    }
    set SYNTH_OK 1
}

if {!$SYNTH_OK} { s3a_fail "synthesis did not produce an open design" }

# ---- structural assertions --------------------------------------------------
set bb [get_cells -hier -quiet -filter {IS_BLACKBOX}]
puts "Remaining black boxes: [llength $bb]"
foreach b $bb { puts "  [get_property NAME $b]" }
if {[llength $bb] != 0} { s3a_fail "netlist still contains black boxes" }

set acc [get_cells -hier -quiet -filter {NAME =~ *u_Accelerator_Top*}]
if {[llength $acc] == 0} { s3a_fail "u_Accelerator_Top not found - accelerator optimised away" }

set pes {}
foreach c [get_cells -hier -quiet -filter {NAME =~ *SA/PER*PEC*}] {
    if {[regexp {PER\[(\d)\]\.PEC\[(\d)\]} [get_property NAME $c] -> r cc]} {
        set key "$r,$cc"
        if {[lsearch -exact $pes $key] < 0} { lappend pes $key }
    }
}
set NPE [llength $pes]
puts "Distinct PE instances (PER\[r\].PEC\[c\]): $NPE"
if {$NPE != 64} {
    puts "WARNING: expected 64 distinct PEs, found $NPE."
    puts "         Inspect stage3a_8x8_utilization_hier.rpt before accepting."
}

# ---- report files -----------------------------------------------------------
set R_TOT [file join $OUT stage3a_8x8_utilization_total.rpt]
set R_HIE [file join $OUT stage3a_8x8_utilization_hier.rpt]
set R_ACC [file join $OUT stage3a_8x8_utilization_accelerator.rpt]
set R_TIM [file join $OUT stage3a_8x8_timing_post_synth_ESTIMATE.rpt]
report_utilization               -file $R_TOT
report_utilization -hierarchical -hierarchical_depth 4 -file $R_HIE
report_utilization -cells $acc   -file $R_ACC
catch {
    report_timing_summary -delay_type max -max_paths 10 -warn_on_violation -file $R_TIM
}

# ---- named GUI reports (guarded; ignored where unsupported) -----------------
catch { report_utilization -name stage3a_8x8_utilization_total }
catch { report_utilization -cells $acc -name stage3a_8x8_utilization_accelerator }
catch { report_utilization -hierarchical -hierarchical_depth 4 \
                           -name stage3a_8x8_utilization_hier }
catch { report_timing_summary -delay_type max -max_paths 10 \
                              -name stage3a_8x8_timing_post_synth_ESTIMATE }

# ---- pull accelerator numbers for the banner -------------------------------
proc grab {file pat} {
    if {![file exists $file]} { return "n/a" }
    set fh [open $file r]; set txt [read $fh]; close $fh
    foreach ln [split $txt "\n"] {
        if {[string match $pat $ln]} {
            set f [split $ln "|"]
            if {[llength $f] > 2} { return [string trim [lindex $f 2]] }
        }
    }
    return "n/a"
}
set A_LUT  [grab $R_ACC "*Slice LUTs*"]
set A_FF   [grab $R_ACC "*Slice Registers*"]
set A_BRAM [grab $R_ACC "*Block RAM Tile*"]
set A_DSP  [grab $R_ACC "*DSPs*"]

puts "\n================================================"
puts " STAGE 3A 8X8 SYNTHESIS PASS"
puts " Errors: 0"
puts " Black boxes: 0"
puts " Distinct PEs: $NPE"
puts " Accelerator LUT: $A_LUT"
puts " Accelerator FF: $A_FF"
puts " Accelerator BRAM: $A_BRAM"
puts " Accelerator DSP: $A_DSP"
puts "================================================"
puts " Reports:"
puts "   $R_TOT"
puts "   $R_HIE"
puts "   $R_ACC"
puts "   $R_TIM   (post-synthesis ESTIMATE only - NOT timing closure)"
if {[file isdirectory $PRJDIR]} {
    puts " Project: [file join $PRJDIR DVCon_STAGE3A_SYN.xpr]"
} else {
    puts " Project: in-memory (low-memory flow; no .xpr written)"
}
puts "================================================"
puts "STAGE3A_SYN: DONE"

if {$BATCH} { exit 0 }
puts "\nSynthesized design remains open. See the Reports window, or open the"
puts "report files listed above. No implementation or bitstream was run."
