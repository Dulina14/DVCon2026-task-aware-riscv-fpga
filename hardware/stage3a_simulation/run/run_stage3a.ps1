#Requires -Version 5.1
<#
=============================================================================
 run_stage3a.ps1 - DVCon Stage 3A 8x8 evaluation flow (Windows controller)

   .\RUN\run_stage3a.ps1                        interactive menu
   .\RUN\run_stage3a.ps1 -Mode Simulation
   .\RUN\run_stage3a.ps1 -Mode Synthesis
   .\RUN\run_stage3a.ps1 -Mode All
   .\RUN\run_stage3a.ps1 -Mode Check
   .\RUN\run_stage3a.ps1 -Mode RebuildFirmware   (OPTIONAL developer mode,
                                                  requires Docker)

 DOCKER IS NOT REQUIRED for Check / Simulation / Synthesis / All.
 The evaluation flow uses the submitted, verified 64-bit boot-ROM MIF and
 never rebuilds, copies or modifies firmware.

 Stage 3A scope only: no implementation, no bitstream, no FPGA programming,
 no QuestaSim submenu. No WSL/Bash required. PowerShell 5.1 and 7 supported.
=============================================================================
#>
[CmdletBinding()]
param(
    [ValidateSet('Simulation','Synthesis','All','Check','RebuildFirmware','Menu')]
    [string]$Mode = 'Menu',

    # Headless/CI. Omit for the default Vivado GUI experience.
    [switch]$Batch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$App    = Join-Path $ProjectRoot 'Application\stage3a_test'
$Out    = Join-Path $ProjectRoot 'GEN_BIT_OUT'
$TclDir = Join-Path $ProjectRoot 'TCL'
$RomDir = Join-Path $ProjectRoot 'DVCon_SoC_SRC\MEMORY_IP\rom_32KB_axi'
$LogDir = Join-Path $Out 'logs'
$Part   = 'xc7k325tffg900-2'

$CanonMif = Join-Path $App    'build\accelarator_test.hex.mif'
$RomMif   = Join-Path $RomDir 'rom_32KB_axi.mif'

$DockerImage = 'dvcon_2:latest'
$DockerTar   = Join-Path $ProjectRoot 'Application\.docker_scripts\docker.tar'

function Write-Ok   ($m){ Write-Host "[ OK ] $m" -ForegroundColor Green }
function Write-Bad  ($m){ Write-Host "[FAIL] $m" -ForegroundColor Red }
function Write-Warn2($m){ Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Die ($m){ Write-Bad $m; exit 1 }

function Get-Sha256([string]$Path){ (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLower() }

function Get-VivadoVersion {
    if (-not (Get-Command vivado -ErrorAction SilentlyContinue)) { return 'NOT FOUND' }
    try {
        $v = & vivado -version 2>$null | Select-Object -First 1
        if ($v) { return $v.ToString().Trim() } else { return 'unknown' }
    } catch { return 'unknown' }
}

function Get-PythonExe {
    foreach ($c in @('python','python3','py')) {
        $e = Get-Command $c -ErrorAction SilentlyContinue
        if ($e) { return $e.Source }
    }
    return $null
}

function Invoke-Vivado {
    param([string]$Tcl, [string]$Log, [bool]$UseBatch)
    $py = Get-PythonExe
    if ($py) { $env:STAGE3A_PYTHON = $py }
    $code = 0
    Push-Location $ProjectRoot
    try {
        # NOTE: the '| Out-Host' is essential and must not be removed.
        #
        # A PowerShell function returns EVERYTHING left on its success stream,
        # not just what 'return' names. Without Out-Host, every line Vivado
        # prints becomes part of this function's return value, so the caller
        # receives an object[] of console text with the exit code appended
        # instead of a single integer. '$rc -ne 0' on an array is a filter, not
        # a comparison - it yields the non-zero elements, which is a non-empty
        # array, which is truthy. The result was that a perfectly successful
        # synthesis always reported failure, and Vivado's progress output was
        # swallowed instead of shown.
        #
        # Out-Host writes Vivado's output straight to the console (so the user
        # sees progress) while keeping the success stream clean.
        if ($UseBatch) {
            $env:STAGE3A_BATCH = '1'
            & vivado -mode batch -nojournal -log $Log -source $Tcl | Out-Host
        } else {
            Remove-Item Env:\STAGE3A_BATCH -ErrorAction SilentlyContinue
            & vivado -nojournal -log $Log -source $Tcl | Out-Host
        }
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } finally { Pop-Location }
    return [int]$code
}

# Decide whether a Vivado invocation succeeded.
#
# The exit code alone is NOT a reliable verdict in GUI mode. When the design is
# left open for review (which is the intended judge experience) the user closes
# the window when finished, and Vivado can return a non-zero exit code for that
# even though every step completed. The authority is therefore the completion
# marker the Tcl prints into the log; the exit code is only used to produce a
# better message when the marker is absent, and is enforced strictly in batch.
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
function Test-LogMarker {
    param([string]$Log, [string]$Regex)
    if (-not (Test-Path -LiteralPath $Log)) { return $false }
    return [bool](Select-String -LiteralPath $Log -Pattern $Regex -Quiet)
}

function Assert-VivadoOutcome {
    param($Rc, [string]$Log, [string]$MarkerRegex, [string]$MarkerName,
          [string]$What, [bool]$UseBatch)

    # Defensive: if anything ever leaks onto the caller's success stream again,
    # the real exit code is the last element. Never let a stray array be
    # compared against 0 - that is a filter in PowerShell, not a test.
    if ($Rc -is [array]) { $Rc = $Rc[-1] }
    $code = 0
    if ($null -ne $Rc) { [void][int]::TryParse([string]$Rc, [ref]$code) }
    $Rc = $code

    $Marker = $MarkerName
    $done = Test-LogMarker -Log $Log -Regex $MarkerRegex

    if (-not $done) {
        if ($Rc -ne 0) { Die "Vivado $What failed (exit code $Rc) - see $Log" }
        Die "$What did not reach its completion marker '$Marker' - see $Log"
    }

    if ($UseBatch -and $Rc -ne 0) {
        Die "Vivado $What printed '$Marker' but exited with code $Rc - see $Log"
    }

    if ((-not $UseBatch) -and $Rc -ne 0) {
        Write-Warn2 "Vivado exited with code $Rc. The run had already completed ('$Marker' is in the log), so this is just how Vivado reports the GUI window being closed. Not a failure."
    }
}

function Test-MifFormat([string]$Path){
    $first = (Get-Content -LiteralPath $Path -TotalCount 1)
    if ($null -eq $first) { return $false }
    $first = $first.TrimEnd()
    return ($first.Length -eq 64 -and $first -match '^[01]+$')
}

#---------------------------------------------------------------- preflight ---
function Invoke-Preflight {
    param([switch]$Deep)
    $rc = $true
    Write-Host '------------------------------------------------------------'
    Write-Host ' Preflight check  (Docker is NOT required)'
    Write-Host '------------------------------------------------------------'

    $vv = Get-VivadoVersion
    if ($vv -eq 'NOT FOUND') { Write-Bad 'Vivado not on PATH'; $rc = $false }
    else { Write-Ok "Vivado detected: $vv" }

    $py = Get-PythonExe
    if ($py) {
        $pv = (& $py --version 2>&1 | Select-Object -First 1)
        Write-Ok "Python detected: $pv"
    } else { Write-Bad 'Python not found (required by the strict VCD analyzer)'; $rc = $false }

    $required = @(
        'DVCon_SoC_SRC\AS1061_SYSTEM\AS1061_SYSTEM_TOP.v',
        'DVCon_SoC_SRC\TOP\Top.vhd',
        'DVCon_SoC_SRC\TB\test_bench.vhd',
        'DVCon_SoC_SRC\ACCELERATOR_IP\Accelerator_Top.v',
        'DVCon_SoC_SRC\ACCELERATOR_IP\rtl\sys\config.svh',
        'DVCon_SoC_SRC\ACCELERATOR_IP\rtl\sys\top_axi_int.v',
        'DVCon_SoC_SRC\ACCELERATOR_IP\rtl\sys\dma_controller.sv',
        'Application\stage3a_test\stage3a_main.c',
        'Application\stage3a_test\Makefile',
        'Application\stage3a_test\mbl.lds',
        'Application\stage3a_test\crt.S',
        'Application\stage3a_test\analyze_vcd_8x8.py',
        'Application\stage3a_test\golden_8x8.py',
        'TCL\DVCon_SIM_stage3a.tcl',
        'TCL\DVCon_SYN_stage3a.tcl'
    )
    $missing = $false
    foreach ($r in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $r))) {
            Write-Bad "MISSING $r"; $missing = $true; $rc = $false
        }
    }
    if (-not $missing) { Write-Ok 'required Stage 3A files found' }

    if (Test-Path -LiteralPath (Join-Path $RomDir 'rom_32KB_axi.xci')) { Write-Ok 'ROM XCI found' }
    else { Write-Bad 'MISSING ROM XCI'; $rc = $false }

    $haveCanon = (Test-Path -LiteralPath $CanonMif) -and ((Get-Item -LiteralPath $CanonMif).Length -gt 0)
    $haveRom   = (Test-Path -LiteralPath $RomMif)   -and ((Get-Item -LiteralPath $RomMif).Length   -gt 0)
    if ($haveCanon) { Write-Ok 'canonical firmware MIF found' } else { Write-Bad 'canonical firmware MIF missing'; $rc = $false }
    if ($haveRom)   { Write-Ok 'installed ROM MIF found' }      else { Write-Bad 'installed ROM MIF missing';      $rc = $false }

    if ($haveCanon -and $haveRom) {
        $ch = Get-Sha256 $CanonMif
        $ih = Get-Sha256 $RomMif
        if ($ch -eq $ih) {
            Write-Ok ("MIF hashes match  sha256={0}..." -f $ch.Substring(0,16))
        } else {
            Write-Bad 'Installed ROM MIF does not match the submitted canonical Stage 3A firmware MIF.'
            Write-Host "      canonical : $ch"
            Write-Host "      installed : $ih"
            $rc = $false
        }
        if (Test-MifFormat $RomMif) { Write-Ok 'ROM MIF is 64-bit binary format' }
        else { Write-Bad 'ROM MIF is not the expected 64-bit binary format'; $rc = $false }
    }

    foreach ($d in @($Out, $LogDir, (Join-Path $ProjectRoot 'VIVADO_PROJECT'))) {
        if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    }
    Write-Ok 'GEN_BIT_OUT writable'
    Write-Ok 'VIVADO_PROJECT writable'

    $hard = $false
    foreach ($f in @((Join-Path $TclDir 'DVCon_SIM_stage3a.tcl'), (Join-Path $TclDir 'DVCon_SYN_stage3a.tcl'))) {
        if (Test-Path -LiteralPath $f) {
            if (Select-String -LiteralPath $f -Pattern '[A-Za-z]:\\HDL|[A-Za-z]:/HDL|/home/|/mnt/[a-z]/' -Quiet) { $hard = $true }
        }
    }
    if ($hard) { Write-Bad 'hard-coded absolute path found in Stage 3A Tcl'; $rc = $false }
    else { Write-Ok 'no hard-coded paths in Stage 3A Tcl' }

    if ($Deep -and $vv -ne 'NOT FOUND') {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("part_{0}.tcl" -f ([guid]::NewGuid().ToString('N')))
        "if {[llength [get_parts -quiet $Part]]>0} {puts `"PART_OK`"} else {puts `"PART_MISSING`"}`nexit 0" |
            Set-Content -LiteralPath $tmp -Encoding ascii
        $res = & vivado -mode batch -nolog -nojournal -source $tmp 2>$null | Out-String
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
        if ($res -match 'PART_OK') { Write-Ok "target part $Part available" }
        else { Write-Bad "target part $Part NOT available in this Vivado"; $rc = $false }
    } elseif ($vv -ne 'NOT FOUND') {
        Write-Ok 'target part validated at run time by the Tcl scripts'
    }

    Write-Host '------------------------------------------------------------'
    if ($rc) { Write-Ok 'PREFLIGHT PASS' } else { Write-Bad 'PREFLIGHT FAIL' }
    return $rc
}

#--------------------------------------------------------------- operations ---
function Invoke-Simulation {
    param([bool]$UseBatch = $false)
    if (-not (Invoke-Preflight)) { Die 'preflight failed' }
    $log = Join-Path $LogDir ("stage3a_sim_{0}.log" -f (Get-Date -Format yyyyMMdd-HHmmss))
    Write-Host '------------------------------------------------------------'
    Write-Host (" Vivado XSim ({0})  ->  {1}" -f $(if ($UseBatch) {'batch'} else {'GUI'}), $log)
    if (-not $UseBatch) {
        Write-Host ' The waveform stays open when the run finishes. Close Vivado to return here.'
    }
    Write-Host '------------------------------------------------------------'
    $rc = Invoke-Vivado -Tcl (Join-Path $TclDir 'DVCon_SIM_stage3a.tcl') -Log $log -UseBatch:$UseBatch
    Assert-VivadoOutcome -Rc $rc -Log $log `
                         -MarkerRegex '^STAGE3A_SIM: DONE' -MarkerName 'STAGE3A_SIM: DONE' `
                         -What 'simulation' -UseBatch $UseBatch
    Write-Ok 'simulation completed'

    $romHash = Get-Sha256 $RomMif
    $xm = Get-ChildItem -Path (Join-Path $ProjectRoot 'VIVADO_PROJECT') -Recurse -Filter 'rom_32KB_axi.mif' `
            -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($xm) {
        if ((Get-Sha256 $xm.FullName) -eq $romHash) { Write-Ok 'XSim working-directory MIF hash matches' }
        else { Die 'XSim working-directory MIF hash MISMATCH' }
    } else { Write-Warn2 'no XSim working copy of the MIF located (IP loaded it directly)' }

    # The Tcl runs the strict analyzer itself so it works while the GUI stays
    # open. In batch we re-verify here to gate the wrapper exit code.
    if ($UseBatch) {
        $py = Get-PythonExe
        if (-not $py) { Die 'python not available for VCD analysis' }
        Write-Host '--- VCD analysis (strict) ---'
        # '| Out-Host' for the same reason as in Invoke-Vivado: keep the
        # analyzer's report on the console and off the function's success stream.
        & $py (Join-Path $App 'analyze_vcd_8x8.py') `
              (Join-Path $Out 'stage3a_8x8_accelerator.vcd') `
              (Join-Path $Out 'stage3a_8x8_vcd_analysis.md') | Out-Host
        if ($LASTEXITCODE -ne 0) { Die "VCD analysis REJECTED (exit $LASTEXITCODE)" }
        Write-Ok 'VCD analysis ACCEPTED (0 failed checks)'
    } else {
        # Anchored: see Test-LogMarker. An unanchored search matches Vivado's
        # echo of the untaken PASS branch and would green-light a failed run.
        if (-not (Test-LogMarker -Log $log -Regex '^\s*STAGE 3A 8X8 SIMULATION PASS\s*$')) {
            Die "Tcl did not report SIMULATION PASS (see $log)"
        }
        if (Test-LogMarker -Log $log -Regex '^\s*STAGE 3A 8X8 SIMULATION FAIL\s*$') {
            Die "Tcl reported SIMULATION FAIL (see $log)"
        }
        Write-Ok 'VCD analysis ACCEPTED (reported by Vivado Tcl console)'
    }

    @(
        'stage3a_simulation',
        ("date     : {0}" -f (Get-Date -Format o)),
        ("vivado   : {0}" -f (Get-VivadoVersion)),
        ("rom_mif  : {0}" -f $romHash),
        ("vcd      : {0}" -f (Get-Sha256 (Join-Path $Out 'stage3a_8x8_accelerator.vcd'))),
        ("analysis : {0}" -f (Get-Sha256 (Join-Path $Out 'stage3a_8x8_vcd_analysis.md')))
    ) | Set-Content -LiteralPath (Join-Path $Out 'stage3a_sim_manifest.txt') -Encoding ascii
    Write-Ok 'manifest: GEN_BIT_OUT\stage3a_sim_manifest.txt'
    Write-Ok 'SIMULATION PASS'
}

function Invoke-Synthesis {
    param([bool]$UseBatch = $false)
    if (-not (Invoke-Preflight)) { Die 'preflight failed' }
    $log = Join-Path $LogDir ("stage3a_syn_{0}.log" -f (Get-Date -Format yyyyMMdd-HHmmss))
    Write-Host '------------------------------------------------------------'
    Write-Host (" Vivado synthesis ({0})  ->  {1}" -f $(if ($UseBatch) {'batch'} else {'GUI'}), $log)
    if (-not $UseBatch) {
        Write-Host ' The synthesized design stays open when the run finishes. Close Vivado to return here.'
    }
    Write-Host '------------------------------------------------------------'
    $rc = Invoke-Vivado -Tcl (Join-Path $TclDir 'DVCon_SYN_stage3a.tcl') -Log $log -UseBatch:$UseBatch
    Assert-VivadoOutcome -Rc $rc -Log $log `
                         -MarkerRegex '^STAGE3A_SYN: DONE' -MarkerName 'STAGE3A_SYN: DONE' `
                         -What 'synthesis' -UseBatch $UseBatch
    if (Test-LogMarker -Log $log -Regex '^\s*STAGE 3A 8X8 SYNTHESIS FAIL\s*$') {
        Die "Tcl reported SYNTHESIS FAIL (see $log)"
    }
    foreach ($r in @('total','hier','accelerator')) {
        $p = Join-Path $Out "stage3a_8x8_utilization_$r.rpt"
        if (Test-Path -LiteralPath $p) { Write-Ok "report stage3a_8x8_utilization_$r.rpt" }
        else { Die "missing utilization report: $r" }
    }
    Write-Host '--- accelerator utilization ---'
    Select-String -LiteralPath (Join-Path $Out 'stage3a_8x8_utilization_accelerator.rpt') `
        -Pattern 'Slice LUTs\*|Slice Registers|Block RAM Tile|DSPs ' |
        Select-Object -First 5 | ForEach-Object { $_.Line }
    @(
        'stage3a_synthesis',
        ("date    : {0}" -f (Get-Date -Format o)),
        ("vivado  : {0}" -f (Get-VivadoVersion)),
        ("rom_mif : {0}" -f (Get-Sha256 $RomMif))
    ) | Set-Content -LiteralPath (Join-Path $Out 'stage3a_syn_manifest.txt') -Encoding ascii
    Write-Ok 'SYNTHESIS PASS'
}

function Invoke-All { Invoke-Simulation -UseBatch $true; Write-Host ''; Invoke-Synthesis -UseBatch $true; Write-Host ''; Write-Ok 'COMPLETE FLOW PASS' }

#------------------------------- OPTIONAL developer-only firmware rebuild -----
function Invoke-RebuildFirmware {
    Write-Host '------------------------------------------------------------'
    Write-Host ' OPTIONAL developer firmware rebuild (requires Docker)'
    Write-Host " Not part of the judges' evaluation flow."
    Write-Host '------------------------------------------------------------'
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Die 'docker CLI not found' }
    & docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Die 'docker daemon not responding (start Docker Desktop and retry)' }
    & docker image inspect $DockerImage 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        if (-not (Test-Path -LiteralPath $DockerTar)) { Die "image $DockerImage absent and docker.tar not found" }
        Write-Host "loading $DockerImage from docker.tar ..."
        & docker load -i $DockerTar 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { Die 'docker load failed' }
    }
    $mount = "$($ProjectRoot):/work"
    & docker run --rm -v $mount -w /work/Application/stage3a_test $DockerImage `
        bash -lc 'export PATH=$PATH:/work/Application/toolchain-bare/bin:/work/Application/Util; make clean >/dev/null 2>&1 || true; make'
    if ($LASTEXITCODE -ne 0) { Die 'firmware build failed' }
    Write-Ok 'firmware rebuilt'
    if ((Test-Path -LiteralPath $CanonMif) -and (Test-Path -LiteralPath $RomMif)) {
        $a = Get-Sha256 $CanonMif; $b = Get-Sha256 $RomMif
        Write-Host "  rebuilt canonical MIF : $a"
        Write-Host "  installed ROM MIF     : $b"
        if ($a -eq $b) { Write-Ok 'rebuilt MIF matches the installed ROM MIF' }
        else { Write-Warn2 'rebuilt MIF DIFFERS from the installed ROM MIF (not auto-installing)' }
    }
    Write-Host 'NOTE: the rebuilt MIF is NOT installed automatically.'
}

function Show-Menu {
    Write-Host ''
    Write-Host '================================================'
    Write-Host '       DVCon Stage 3A 8x8 Evaluation Flow'
    Write-Host '================================================'
    Write-Host "Project root : $ProjectRoot"
    Write-Host ("Vivado       : {0}" -f (Get-VivadoVersion))
    Write-Host "Target       : $Part"
    Write-Host '================================================'
    Write-Host '1) Functional Simulation - Vivado XSim GUI'
    Write-Host '2) RTL Synthesis - AMD Vivado GUI'
    Write-Host '3) Exit'
    Write-Host '================================================'
}

switch ($Mode) {
    'Simulation'      { Invoke-Simulation -UseBatch:$Batch.IsPresent; exit 0 }
    'Synthesis'       { Invoke-Synthesis  -UseBatch:$Batch.IsPresent; exit 0 }
    'All'             { Invoke-All;        exit 0 }
    'Check'           { if (Invoke-Preflight -Deep) { exit 0 } else { exit 1 } }
    'RebuildFirmware' { Invoke-RebuildFirmware; exit 0 }
    default {
        while ($true) {
            Show-Menu
            $c = Read-Host 'Select an option [1-3]'
            switch ($c) {
                '1' { Invoke-Simulation -UseBatch $false }
                '2' { Invoke-Synthesis  -UseBatch $false }
                '3' { Write-Host 'Exiting.'; exit 0 }
                default { Write-Host 'Invalid selection.' }
            }
            Write-Host ''
        }
    }
}
