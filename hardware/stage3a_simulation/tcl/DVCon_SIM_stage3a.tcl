#=============================================================================
# DVCon_SIM_stage3a.tcl - Stage 3A 8x8 functional simulation (GUI or batch)
#-----------------------------------------------------------------------------
# GUI (default, judge experience):
#     vivado -source <this file>
# Batch / CI regression:
#     STAGE3A_BATCH=1  vivado -mode batch -source <this file>
#
# Both modes share identical project construction and verification.
# In GUI mode the simulation runs automatically, the waveform is displayed and
# Vivado is LEFT OPEN (no exit). In batch mode the script exits with a status.
#
# Optional environment:
#   STAGE3A_PYTHON  full path to python used for the strict VCD analyzer
#   STAGE3A_BATCH   set to 1 for headless/CI behaviour (exit at the end)
#   STAGE3A_RUN_US  simulation length in us (default 1500)
#
# Stage 3A scope: no implementation, no bitstream, no FPGA programming.
#=============================================================================

set BATCH 0
if {[info exists ::env(STAGE3A_BATCH)] && $::env(STAGE3A_BATCH) ne "0"} { set BATCH 1 }

proc s3a_fail {msg} {
    global BATCH
    puts "\n================================================"
    puts " STAGE 3A 8X8 SIMULATION FAIL"
    puts " $msg"
    puts "================================================"
    if {$BATCH} { exit 1 }
    return -code error $msg
}

# ---- root + environment ----------------------------------------------------
set THIS      [file normalize [info script]]
set MDIR_PATH [file normalize [file join [file dirname $THIS] ".."]]
set VIV_VER   [version -short]
set PART      xc7k325tffg900-2

set ACC       [file join $MDIR_PATH DVCon_SoC_SRC ACCELERATOR_IP]
set APP       [file join $MDIR_PATH Application stage3a_test]
set OUT       [file join $MDIR_PATH GEN_BIT_OUT]
set VCD       [file join $OUT stage3a_8x8_accelerator.vcd]
set WCFG      [file join $OUT stage3a_8x8_accelerator.wcfg]
set MD        [file join $OUT stage3a_8x8_vcd_analysis.md]
set ANALYZER  [file join $APP analyze_vcd_8x8.py]
set CANON_MIF [file join $APP build accelarator_test.hex.mif]
set ROM_MIF   [file join $MDIR_PATH DVCon_SoC_SRC MEMORY_IP rom_32KB_axi rom_32KB_axi.mif]

set RUNUS 1500
if {[info exists ::env(STAGE3A_RUN_US)]} { set RUNUS $::env(STAGE3A_RUN_US) }

puts "============================================================"
puts " DVCon Stage 3A 8x8 - Functional Simulation (XSim)"
puts "============================================================"
puts " Project root : $MDIR_PATH"
puts " Vivado       : $VIV_VER"
puts " Target part  : $PART"
puts " Mode         : [expr {$BATCH ? {BATCH} : {GUI}}]"
puts "============================================================"

if {[llength [get_parts -quiet $PART]] == 0} {
    s3a_fail "target part $PART is not available in Vivado $VIV_VER."
}

# ---- submitted program image integrity -------------------------------------
proc sha256_of {f} {
    if {![file exists $f]} { return "" }
    if {[catch {exec sha256sum $f} r]} {
        if {[catch {exec certutil -hashfile $f SHA256} r2]} { return "UNAVAILABLE" }
        foreach ln [split $r2 "\n"] {
            set ln [string trim $ln]
            if {[regexp {^[0-9a-fA-F]{64}$} $ln]} { return [string tolower $ln] }
        }
        return "UNAVAILABLE"
    }
    return [string tolower [lindex [split [string trim $r]] 0]]
}
foreach f [list $CANON_MIF $ROM_MIF] {
    if {![file exists $f]} { s3a_fail "required MIF missing: $f" }
}
set ch [sha256_of $CANON_MIF]
set ih [sha256_of $ROM_MIF]
if {$ch eq "UNAVAILABLE" || $ih eq "UNAVAILABLE"} {
    puts "NOTE: no hashing utility found; skipping MIF hash comparison."
} elseif {$ch ne $ih} {
    s3a_fail "Installed ROM MIF does not match the submitted canonical Stage 3A firmware MIF."
} else {
    puts " ROM MIF sha256 : $ch  (canonical == installed)"
}

# ---- pure-Tcl recursive source discovery -----------------------------------
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

# ---- fresh timestamped project ---------------------------------------------
set STAMP [clock format [clock seconds] -format %d%m%y-%H%M%S]
set PRJDIR [file join $MDIR_PATH VIVADO_PROJECT STAGE3A_SIM_8x8_$STAMP]
create_project DVCon_STAGE3A_SIM $PRJDIR -part $PART
set_property part $PART           [current_project]
set_property target_language VHDL [current_project]

add_files [file join $MDIR_PATH DVCon_SoC_SRC AS1061_SYSTEM AS1061_SYSTEM_TOP.v]
add_files [file join $MDIR_PATH DVCon_SoC_SRC TOP Top.vhd]
add_files [file join $MDIR_PATH DVCon_SoC_SRC TB test_bench.vhd]
add_files [file join $MDIR_PATH DVCon_SoC_SRC MEMORY_IP rom_32KB_axi rom_32KB_axi.xci]
foreach f [find_src $ACC] { add_files $f }

set ACC_INC [file join $ACC rtl sys]
set_property include_dirs [list $ACC_INC] [get_filesets sources_1]
set_property include_dirs [list $ACC_INC] [get_filesets sim_1]
set tai [get_files -quiet [file join $ACC rtl sys top_axi_int.v]]
if {[llength $tai]} { set_property file_type SystemVerilog $tai }

set xdc [file join $MDIR_PATH DVCon_SoC_XDC AS1061_SYSTEM_XDC.xdc]
if {[file exists $xdc]} { add_files -fileset constrs_1 -norecurse $xdc }

update_compile_order -fileset sources_1
set_property top test_bench [get_filesets sim_1]
update_compile_order -fileset sim_1

foreach ip [get_ips -quiet] {
    if {[get_property -quiet IS_LOCKED $ip] eq "1"} {
        puts "WARNING: IP '$ip' is LOCKED under Vivado $VIV_VER (not auto-upgrading)."
    }
}

# ---- elaborate + start XSim -------------------------------------------------
puts "\n--- launching simulation ---"
if {[catch {launch_simulation -simset sim_1} emsg]} { s3a_fail "launch_simulation failed: $emsg" }

set A ""
foreach p {test_bench/u_Top/u_Accelerator_Top /test_bench/u_Top/u_Accelerator_Top} {
    if {[llength [get_scopes -quiet $p]] > 0} { set A $p; break }
}
if {$A eq ""} { s3a_fail "accelerator scope not found after elaboration" }
set tb ""
foreach p {test_bench /test_bench} { if {[llength [get_scopes -quiet $p]]>0} { set tb $p; break } }
puts "scope: $A"

# ---- wave window: clean config + 11 divider groups (GUI) --------------------
proc W  {objs {radix hex}} {
    foreach o $objs {
        set h [get_objects -quiet $o]
        if {[llength $h] > 0} { catch { add_wave -radix $radix $h } }
    }
}
proc W1 {objs} { W $objs bin }

set WAVE_OK 0
if {![catch { create_wave_config -name stage3a_8x8 } ]} { set WAVE_OK 1 }

if {$WAVE_OK} {
    catch {
    add_wave_divider "CLOCK / RESET"
    W1 [list $A/s_axi_aclk $A/s_axi_aresetn $tb/proc_beat]

    add_wave_divider "UART STATUS"
    W1 [list $tb/uart_out $tb/sout0 $tb/sin0]

    add_wave_divider "PROCESSOR -> ACCELERATOR AXI SLAVE : WRITE"
    W1 [list $A/s_axi_awvalid $A/s_axi_awready]
    W  [list $A/s_axi_awaddr $A/s_axi_awid $A/s_axi_awlen $A/s_axi_awsize]
    W1 [list $A/s_axi_wvalid $A/s_axi_wready $A/s_axi_wlast]
    W  [list $A/s_axi_wdata $A/s_axi_wstrb]
    W1 [list $A/s_axi_bvalid $A/s_axi_bready]
    W  [list $A/s_axi_bresp]

    add_wave_divider "PROCESSOR -> ACCELERATOR AXI SLAVE : READ"
    W1 [list $A/s_axi_arvalid $A/s_axi_arready]
    W  [list $A/s_axi_araddr]
    W1 [list $A/s_axi_rvalid $A/s_axi_rready $A/s_axi_rlast]
    W  [list $A/s_axi_rdata $A/s_axi_rresp]

    add_wave_divider "AXI4 -> AXI-LITE CONFIG BRIDGE"
    W1 [list $A/u_accel/s_axil_awvalid $A/u_accel/s_axil_awready]
    W  [list $A/u_accel/s_axil_awaddr $A/u_accel/s_axil_wdata $A/u_accel/s_axil_wstrb]
    W1 [list $A/u_accel/s_axil_wvalid $A/u_accel/s_axil_wready $A/u_accel/s_axil_bvalid]
    W1 [list $A/u_accel/s_axil_arvalid $A/u_accel/s_axil_arready $A/u_accel/s_axil_rvalid]
    W  [list $A/u_accel/s_axil_araddr $A/u_accel/s_axil_rdata]

    add_wave_divider "ACCELERATOR CONTROL / DMA STATUS"
    W1 [list $A/u_accel/reg_wr_en]
    W  [list $A/u_accel/reg_wr_addr $A/u_accel/reg_wr_data]
    W1 [list $A/u_accel/reg_rd_en]
    W  [list $A/u_accel/reg_rd_addr $A/u_accel/reg_rd_data]
    W  [list $A/u_accel/CONTROLLER/cfg_START]
    W  [list $A/u_accel/CONTROLLER/cfg_MM2S_0_ADDR $A/u_accel/CONTROLLER/cfg_MM2S_0_BYTES \
             $A/u_accel/CONTROLLER/cfg_MM2S_0_TUSER $A/u_accel/CONTROLLER/cfg_MM2S_0_DONE]
    W  [list $A/u_accel/CONTROLLER/cfg_MM2S_1_ADDR $A/u_accel/CONTROLLER/cfg_MM2S_1_BYTES \
             $A/u_accel/CONTROLLER/cfg_MM2S_1_TUSER $A/u_accel/CONTROLLER/cfg_MM2S_1_DONE]
    W  [list $A/u_accel/CONTROLLER/cfg_MM2S_2_ADDR $A/u_accel/CONTROLLER/cfg_MM2S_2_BYTES \
             $A/u_accel/CONTROLLER/cfg_MM2S_2_TUSER $A/u_accel/CONTROLLER/cfg_MM2S_2_DONE]
    W  [list $A/u_accel/CONTROLLER/cfg_S2MM_ADDR $A/u_accel/CONTROLLER/cfg_S2MM_BYTES \
             $A/u_accel/CONTROLLER/cfg_S2MM_DONE]
    W1 [list $A/u_accel/CONTROLLER/mm2s_0_valid $A/u_accel/CONTROLLER/mm2s_0_ready]
    W1 [list $A/u_accel/CONTROLLER/mm2s_1_valid $A/u_accel/CONTROLLER/mm2s_1_ready]
    W1 [list $A/u_accel/CONTROLLER/mm2s_2_valid $A/u_accel/CONTROLLER/mm2s_2_ready]
    W1 [list $A/u_accel/CONTROLLER/s2mm_valid   $A/u_accel/CONTROLLER/s2mm_ready]
    W1 [list $A/u_accel/CONTROLLER/mm2s_0_status_valid $A/u_accel/CONTROLLER/mm2s_1_status_valid \
             $A/u_accel/CONTROLLER/mm2s_2_status_valid $A/u_accel/CONTROLLER/s2mm_status_valid]
    W  [list $A/u_accel/CONTROLLER/mm2s_0_status_error $A/u_accel/CONTROLLER/mm2s_1_status_error \
             $A/u_accel/CONTROLLER/mm2s_2_status_error $A/u_accel/CONTROLLER/s2mm_status_error]

    add_wave_divider "ACCELERATOR -> SRAM AXI MASTER : READ"
    W1 [list $A/m_axi_arvalid $A/m_axi_arready]
    W  [list $A/m_axi_araddr $A/m_axi_arid $A/m_axi_arlen $A/m_axi_arsize $A/m_axi_arburst]
    W1 [list $A/m_axi_rvalid $A/m_axi_rready $A/m_axi_rlast]
    W  [list $A/m_axi_rdata $A/m_axi_rid $A/m_axi_rresp]

    add_wave_divider "INTERNAL K / X / BIAS STREAMS"
    W1 [list $A/u_accel/s_k_tvalid $A/u_accel/s_k_tready $A/u_accel/s_k_tlast]
    W  [list $A/u_accel/s_k_tdata]
    W1 [list $A/u_accel/s_x_tvalid $A/u_accel/s_x_tready $A/u_accel/s_x_tlast]
    W  [list $A/u_accel/s_x_tdata]
    W1 [list $A/u_accel/s_a_tvalid $A/u_accel/s_a_tready $A/u_accel/s_a_tlast]
    W  [list $A/u_accel/s_a_tdata]

    add_wave_divider "8x8 SYSTOLIC ARRAY / FINAL RESULT STREAM"
    W1 [list $A/u_accel/s_valid $A/u_accel/s_ready $A/u_accel/s_last]
    W1 [list $A/u_accel/m_valid $A/u_accel/m_ready $A/u_accel/m_last]
    W  [list $A/u_accel/m_data]
    W1 [list $A/u_accel/m_y_tvalid $A/u_accel/m_y_tready $A/u_accel/m_y_tlast]
    W  [list $A/u_accel/m_y_tdata]
    W1 [list $A/u_accel/m_axis_s2mm_tvalid $A/u_accel/m_axis_s2mm_tready $A/u_accel/m_axis_s2mm_tlast]
    W  [list $A/u_accel/m_axis_s2mm_tdata $A/u_accel/m_axis_s2mm_tkeep]

    add_wave_divider "ACCELERATOR -> SRAM AXI MASTER : WRITE"
    W1 [list $A/m_axi_awvalid $A/m_axi_awready]
    W  [list $A/m_axi_awaddr $A/m_axi_awid $A/m_axi_awlen $A/m_axi_awsize]
    W1 [list $A/m_axi_wvalid $A/m_axi_wready $A/m_axi_wlast]
    W  [list $A/m_axi_wdata $A/m_axi_wstrb]
    W1 [list $A/m_axi_bvalid $A/m_axi_bready]
    W  [list $A/m_axi_bresp $A/m_axi_bid]

    add_wave_divider "RESULT / COMPLETION"
    W  [list $A/m_axi_awaddr $A/m_axi_wdata $A/m_axi_wstrb]
    W1 [list $A/m_axi_wlast]
    W  [list $A/u_accel/CONTROLLER/cfg_S2MM_DONE]
    }
    puts "wave window: 11 divider groups added"
} else {
    puts "NOTE: no wave window (batch). VCD remains the authoritative evidence;"
    puts "      the committed stage3a_8x8_accelerator.wcfg is used for GUI review."
}

# ---- targeted VCD logging (before reset release) ----------------------------
file mkdir $OUT
open_vcd $VCD
set stream_objs {}
foreach sig {
    s_k_tvalid s_k_tready s_k_tlast s_k_tdata
    s_x_tvalid s_x_tready s_x_tlast s_x_tdata
    s_a_tvalid s_a_tready s_a_tlast s_a_tdata
    s_valid s_ready s_last  m_valid m_ready m_last m_data
    m_y_tvalid m_y_tready m_y_tlast m_y_tdata
    m_axis_s2mm_tvalid m_axis_s2mm_tready m_axis_s2mm_tlast
    m_axis_s2mm_tdata  m_axis_s2mm_tkeep
    reg_wr_en reg_wr_addr reg_wr_data reg_rd_en reg_rd_addr reg_rd_data
} {
    set h [get_objects -quiet $A/u_accel/$sig]
    if {[llength $h] > 0} { lappend stream_objs {*}$h } else { puts "  WARN: missing $sig" }
}
set objs [concat \
    [get_objects -quiet $A/*] \
    [get_objects -quiet $A/u_accel/s_axil_*] \
    $stream_objs \
    [get_objects -quiet $A/u_accel/CONTROLLER/*] \
    [get_objects -quiet $tb/uart_out] [get_objects -quiet $tb/sout0] \
    [get_objects -quiet $tb/sin0]     [get_objects -quiet $tb/proc_beat] ]
puts "logging [llength $objs] objects to VCD"
log_vcd $objs

# ---- run --------------------------------------------------------------------
puts "\n--- running $RUNUS us (automatic; no user action required) ---"
run ${RUNUS} us
flush_vcd
close_vcd
puts "VCD written: $VCD ([file size $VCD] bytes)"

# ---- save + zoom the wave configuration ------------------------------------
set ZOOM_START "460 us"
set ZOOM_END   "500 us"
if {$WAVE_OK} { catch { save_wave_config $WCFG } }
if {[file exists $WCFG]} {
    if {![catch {
        set fh [open $WCFG r]; set xml [read $fh]; close $fh
        regsub {<ZoomStartTime time="[^"]*">} $xml "<ZoomStartTime time=\"$ZOOM_START\">" xml
        regsub {<ZoomEndTime time="[^"]*">}   $xml "<ZoomEndTime time=\"$ZOOM_END\">"     xml
        set fh [open $WCFG w]; puts -nonewline $fh $xml; close $fh
    }]} { puts "wcfg zoom set to $ZOOM_START .. $ZOOM_END" }
}
# reload so the GUI shows the intended zoom
catch { open_wave_config $WCFG }
catch { wave zoom range 460us 500us }

# ---- strict VCD analysis (runs while the GUI stays open) --------------------
set PY "python3"
if {[info exists ::env(STAGE3A_PYTHON)] && $::env(STAGE3A_PYTHON) ne ""} {
    set PY $::env(STAGE3A_PYTHON)
}
puts "\n--- strict VCD analysis ($PY) ---"
set arc 0
set aout ""
if {[catch {set aout [exec $PY $ANALYZER $VCD $MD 2>@1]} aout]} { set arc 1 }
foreach ln [split $aout "\n"] {
    if {[string match "*RESULT:*" $ln] || [string match "*Decoded UART*" $ln] ||
        [string match "*Overall result*" $ln] || [string match "*acceptance check*" $ln] ||
        [string match "*decoded*" $ln]} { puts "  $ln" }
}

set uart "?"
if {[regexp {Decoded UART: `([A-Z]+)`} $aout -> u]} { set uart $u }
set n64 "?"
if {[regexp {Final results \((\d+)/64 decoded\)} $aout -> n]} { set n64 "$n/64" }

if {$arc == 0} {
    puts "\n================================================"
    puts " STAGE 3A 8X8 SIMULATION PASS"
    puts " UART: $uart"
    puts " Outputs: $n64 correct"
    puts " VCD analysis: ACCEPTED"
    puts "================================================"
    puts " VCD    : $VCD"
    puts " WCFG   : $WCFG"
    puts " Report : $MD"
    puts " Project: [file join $PRJDIR DVCon_STAGE3A_SIM.xpr]"
    puts "================================================"
    puts "STAGE3A_SIM: DONE"
    if {$BATCH} { exit 0 }
    puts "\nGUI remains open. Waveform zoom ~460-500 us; widen to 0-1200 us for UART BCSP."
} else {
    puts "\n================================================"
    puts " STAGE 3A 8X8 SIMULATION FAIL"
    puts " VCD analysis REJECTED - see $MD"
    puts "================================================"
    if {$BATCH} { exit 1 }
}
