@echo off
REM ==========================================================================
REM  run_stage3a.cmd - CMD launcher for the DVCon Stage 3A 8x8 flow
REM
REM    RUN\run_stage3a.cmd                     interactive menu (Vivado GUI)
REM    RUN\run_stage3a.cmd simulation          GUI simulation
REM    RUN\run_stage3a.cmd simulation batch    headless simulation
REM    RUN\run_stage3a.cmd synthesis           GUI synthesis
REM    RUN\run_stage3a.cmd synthesis batch     headless synthesis
REM    RUN\run_stage3a.cmd all                 headless sim + synth
REM    RUN\run_stage3a.cmd check               preflight only
REM
REM  This file ONLY translates the argument and invokes run_stage3a.ps1.
REM  run_stage3a.ps1 is the real Windows automation controller.
REM ==========================================================================
setlocal
set "PS1=%~dp0run_stage3a.ps1"

if not exist "%PS1%" (
  echo [FAIL] Cannot find "%PS1%"
  exit /b 1
)

set "MODE=Menu"
if /i "%~1"=="simulation" set "MODE=Simulation"
if /i "%~1"=="sim"        set "MODE=Simulation"
if /i "%~1"=="synthesis"  set "MODE=Synthesis"
if /i "%~1"=="syn"        set "MODE=Synthesis"
if /i "%~1"=="all"        set "MODE=All"
if /i "%~1"=="check"      set "MODE=Check"
if /i "%~1"=="rebuild-firmware" set "MODE=RebuildFirmware"
if /i "%~1"=="rebuild"          set "MODE=RebuildFirmware"
if /i "%~1"=="-h"         goto :usage
if /i "%~1"=="--help"     goto :usage
if /i "%~1"=="help"       goto :usage

set "BATCHSW="
if /i "%~2"=="batch"   set "BATCHSW=-Batch"
if /i "%~2"=="--batch" set "BATCHSW=-Batch"
if /i "%MODE%"=="All"  set "BATCHSW=-Batch"

if not "%~1"=="" if /i not "%MODE%"=="Menu" goto :run
if not "%~1"=="" if /i "%MODE%"=="Menu" (
  echo Unknown option: %~1
  goto :usage
)

:run
where pwsh >nul 2>&1
if %ERRORLEVEL%==0 (
  pwsh -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Mode %MODE% %BATCHSW%
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Mode %MODE% %BATCHSW%
)
exit /b %ERRORLEVEL%

:usage
echo.
echo DVCon Stage 3A 8x8 evaluation flow
echo.
echo   RUN\run_stage3a.cmd              interactive menu
echo   RUN\run_stage3a.cmd simulation   firmware build + XSim + VCD analysis
echo   RUN\run_stage3a.cmd synthesis    RTL synthesis + utilization reports
echo   RUN\run_stage3a.cmd all          simulation then synthesis
echo   RUN\run_stage3a.cmd check        preflight only
echo.
echo Stage 3A scope only: no implementation, no bitstream, no FPGA programming.
exit /b 2
