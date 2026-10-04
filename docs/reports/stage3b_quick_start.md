# QUICK START — program, boot, deploy, run, retrieve

This guide follows the same sequence used in the Stage 3B hardware-demonstration
video. Background, architecture and validation details are in
`Team43_SiliconSyndicate_Stage3B_Implementation_Report.docx`.

## 0. Required environment

- Development PC: `hdg_vivado-lab2`, CentOS 7
- Project root: `/home/Team43/Desktop/DVCon_2026`
- Board: Digilent Genesys 2 running VEGA Linux
- Development-PC Ethernet address: `192.168.1.1`
- Board Ethernet address used in this demonstration: `192.168.1.50`
- Organizer-provided Vivado, programming, Linux-transfer and Buildroot files

Do not modify the organizer environment unless specifically authorized. Passwords
are intentionally omitted from this document.

## 1. Program the FPGA

On the development PC, use the organizer launcher:

```bash
cd /home/Team43/Desktop/DVCon_2026/RUN
./run.sh
```

The menu is:

```text
1) Simulation
2) FPGA Implementation
3) FPGA Programming
4) Exit
```

Select **3** to program the existing `GEN_BIT_OUT/DVCon_2026.bit`. Confirm that
Vivado detects `xc7k325t_0`, executes `program_hw_devices`, and exits without a
programming error.

Select menu option **2** only when a new bitstream must be generated. Full
implementation can take several hours and is not required for an ordinary
application rerun.

## 2. Open the board UART console and boot VEGA Linux

Keep the organizer launcher terminal available. Open a second development-PC
terminal and start Minicom:

```bash
sudo minicom dvcon
```

After FPGA programming, the UART console should reach `Waiting for data...`.
Leave Minicom open. In another development-PC terminal, transfer the
organizer-provided Linux image and device tree:

```bash
cd /home/Team43/Desktop/DVCon_2026/FILE_TRANSFER
./send.sh
```

Watch Minicom while VEGA Linux boots. At the `C-DAC Linux` prompt, log in using
the board credentials supplied privately by the organizers. Do not place those
credentials in scripts, reports or recordings.

Optional board identity checks:

```sh
uname -a
head -6 /proc/cpuinfo
head -3 /proc/meminfo
ls -l /dev/accelerator_test
```

The output must identify a `riscv64` system and show the accelerator character
device `/dev/accelerator_test`.

## 3. Configure the board network after every boot

Run these commands in the **board Minicom console**:

```sh
ifconfig eth0 192.168.1.50 netmask 255.255.255.0 up
sleep 2
ifconfig eth0
ping -c 3 192.168.1.1
```

Continue only after `eth0` is up and the ping succeeds.

## 4. Build the application on the development PC (only when needed)

The submitted RISC-V binary may be used directly. Rebuild only after changing
the C sources or headers.

Place `team43_pipeline.c`, `team43_yolo_graph.h`,
`team43_minilm_layout.h`, `jpeg.h` and `font8x8.h` together in:

```text
/home/Team43/Desktop/DVCon_2026/Application/
```

Then run on the development PC:

```bash
cd /home/Team43/Desktop/DVCon_2026/Application

docker run --rm -t \
  -v "$PWD":/workspace \
  -w /workspace \
  --user $(id -u):$(id -g) \
  buildroot-2004_new \
  bash -c 'buildroot_dvcon/buildroot/output/host/bin/riscv64-buildroot-linux-gnu-gcc \
      -std=gnu99 -O2 -static -Wall -Wextra \
      -o team43_pipeline team43_pipeline.c -lm -lrt'

file team43_pipeline
```

The result must be a statically linked 64-bit RISC-V ELF executable.

## 5. Deploy to the board

Two equivalent deployment methods are provided. The board is the SCP client
and pulls files from the development PC.

### Method A — runtime archive, as used in the video

#### A1. Create the archive on the development PC

```bash
cd /home/Team43/Desktop/DVCon_2026/Application

tar -czf ../FILE_TRANSFER/runtime_bundle.tar.gz \
    team43_pipeline \
    team43_minilm_int8.bin \
    team43_yolo_int8.bin \
    vocab.txt \
    demo_serve_wine_glasses.ppm \
    demo_sit_comfortably_couch.ppm

tar -tzf ../FILE_TRANSFER/runtime_bundle.tar.gz
sha256sum ../FILE_TRANSFER/runtime_bundle.tar.gz
```

Check that all six files are listed before continuing.

#### A2. Pull and extract the archive on the board

Run in the board Minicom console:

```sh
mkdir -p /tmp/team43
cd /tmp/team43

scp Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/FILE_TRANSFER/runtime_bundle.tar.gz .

gzip -dc runtime_bundle.tar.gz | tar -xvf -
chmod +x team43_pipeline
ls -lh
```

The organizer board uses BusyBox `tar`; `tar -xzf` may fail because its `tar`
does not accept the `-z` option. The `gzip -dc ... | tar -xvf -` command above
is the verified extraction method.

### Method B — send files one by one

Use this method if archive creation or extraction is unavailable. Run in the
board Minicom console:

```sh
mkdir -p /tmp/team43
cd /tmp/team43

scp Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/Application/team43_pipeline .
scp Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/Application/team43_minilm_int8.bin .
scp Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/Application/team43_yolo_int8.bin .
scp Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/Application/vocab.txt .
scp Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/Application/demo_serve_wine_glasses.ppm .
scp Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/Application/demo_sit_comfortably_couch.ppm .

chmod +x team43_pipeline
ls -lh
```

For either method, the final board directory must contain:

```text
/tmp/team43/team43_pipeline
/tmp/team43/team43_minilm_int8.bin
/tmp/team43/team43_yolo_int8.bin
/tmp/team43/vocab.txt
/tmp/team43/demo_serve_wine_glasses.ppm
/tmp/team43/demo_sit_comfortably_couch.ppm
```

`/tmp` is a RAM disk. A board reboot removes all deployed runtime files, so
repeat sections 3 and 5 after every reboot.

## 6. Run the known-good hardware demonstration

Run in the board Minicom console:

```sh
cd /tmp/team43

./team43_pipeline \
    --image /tmp/team43/demo_serve_wine_glasses.ppm \
    --task "serve wine" \
    --imgsz 128 \
    --rank-by minilm \
    --out /tmp/team43/result_serve_wine.ppm
```

Confirm that the banner reports:

```text
mode: accelerator
```

Do not use `--sim` for the hardware demonstration. At 128 pixels, the verified
run takes approximately nine minutes and prints progress while running.

Known-good result:

```text
--- RESULT ---
  SELECTED : wine glass   (rank 0.97006)
  marked image written to /tmp/team43/result_serve_wine.ppm
--- WORK ---
  total       : 517888 invocations, 538.8 s
```

The full captured output is in `evidence/board_log_milestone_D.txt`.

### Optional second runtime task

This uses the same FPGA bitstream, executable and model packages without
recompilation:

```sh
./team43_pipeline \
    --image /tmp/team43/demo_sit_comfortably_couch.ppm \
    --task "sit comfortably" \
    --imgsz 128 \
    --rank-by minilm \
    --out /tmp/team43/result_sit_comfortably.ppm
```

Report the result actually printed by the hardware run; do not assume the
winner before the detector and task scorer finish.

## 7. Command-line options

Exactly two arguments are required: `--image` and `--task`.

| Option | Default | Meaning |
|---|---|---|
| `--image PATH` | **required** | Input JPEG or binary P6 PPM. The verified timed hardware demonstration used P6 PPM. |
| `--task "TEXT"` | **required** | Runtime task prompt. Always quote prompts containing spaces. |
| `--out PATH` | `/tmp/team43/result.ppm` | Marked output image. Output is always binary P6 PPM. |
| `--imgsz N` | `128` | Detector input square; must be a multiple of 32. Larger values take substantially longer. |
| `--conf F` | `0.25` | YOLO detection-confidence threshold. |
| `--iou F` | `0.45` | NMS IoU threshold. |
| `--rank-by minilm\|product` | `minilm` | Rank using MiniLM alone or MiniLM score multiplied by YOLO confidence. |
| `--yolo PATH` | `/tmp/team43/team43_yolo_int8.bin` | YOLOv8n model package. |
| `--minilm PATH` | `/tmp/team43/team43_minilm_int8.bin` | MiniLM model package. |
| `--vocab PATH` | `/tmp/team43/vocab.txt` | WordPiece vocabulary. |
| `--dev PATH` | `/dev/accelerator_test` | Accelerator character device. |
| `--sim` | off | Software emulation. Do not use for hardware proof. |
| `--quiet` | off | Suppress per-operation progress. |

### Fine-tuned task prompts

The scorer accepts free text but was fine-tuned on these fourteen tasks:

```text
step on something to reach top of a shelf
sit comfortably
place flowers
get potatoes out of fire
water plant
get lemon out of tea
dig hole
open bottle of beer
open parcel
serve wine
pour sugar
smear butter
extinguish fire
pound carpet
```

Prompts outside this list are out of distribution and should not be presented
as validated behavior.

## 8. Retrieve the marked image

### Method A — board pushes the result to the development PC

First create the destination directory on the development PC:

```bash
mkdir -p /home/Team43/Desktop/DVCon_2026/TEAM43_RESULTS
```

Then run on the board:

```sh
scp /tmp/team43/result_serve_wine.ppm \
    Team43@192.168.1.1:/home/Team43/Desktop/DVCon_2026/TEAM43_RESULTS/
```

### Method B — development PC pulls the result from the board

Run on the development PC:

```bash
mkdir -p /home/Team43/Desktop/DVCon_2026/TEAM43_RESULTS
cd /home/Team43/Desktop/DVCon_2026/TEAM43_RESULTS
scp root@192.168.1.50:/tmp/team43/result_serve_wine.ppm .
```

Use whichever direction is permitted by the current SSH configuration. Verify
the received file:

```bash
ls -lh /home/Team43/Desktop/DVCon_2026/TEAM43_RESULTS/result_serve_wine.ppm
sha256sum /home/Team43/Desktop/DVCon_2026/TEAM43_RESULTS/result_serve_wine.ppm
```

Convert the PPM on a host with Python and Pillow:

```bash
python scripts/ppm_to_png.py result_serve_wine.ppm result_serve_wine.png
```

## 9. Optional host-only accuracy evaluation

This requires the COCO-Tasks dataset and is not part of the board quick run:

```bash
python scripts/eval_coco_tasks.py --root D:\DVCon\datasets --boxes gt
python scripts/eval_coco_tasks.py --root D:\DVCon\datasets --boxes faster
python scripts/eval_coco_tasks.py --root D:\DVCon\datasets --boxes gt --method pickbest
```
