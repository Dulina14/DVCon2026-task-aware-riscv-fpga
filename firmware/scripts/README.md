# firmware/scripts/

Scripts we used to build, deploy and run on the board. Paths such as `/home/Team43/Desktop/DVCon_2026` and the addresses `192.168.1.1` (lab PC) and `192.168.1.50` (board) are from the C-DAC lab set-up — change them for yours. No passwords are stored in any script.

| Script | Runs on | What it does |
|---|---|---|
| `build.sh` | lab PC | Builds the board binary in the `buildroot-2004_new` container and the x86 `--sim` binary, both with `-ffp-contract=off` |
| `build_pipeline_n.sh`, `build_memcheck.sh` | lab PC | Build into a new file, never overwriting the validated binary |
| `run_pipeline_n.sh IMAGE TASK [IMGSZ] [OUT]` | board | Runs the YOLOv8n pipeline with `--rect --packed` and the package paths filled in |
| `verify.sh` | board | Checks the deployed payload with `sha256sum` (falls back to `md5sum`, then sizes). Prints `PAYLOAD OK` or the files to re-send |
| `rnet` | board | `ifconfig eth0 192.168.1.50` + ping the lab PC |
| `rcold` | board | Pulls the files from the lab PC with `scp` after a power cycle |
| `rcold_s256` | board | **Timed cold start** for the final YOLOv8s 256 px build: network → pull bundle → unpack → check → first answer. Expected: `wine glass, rank 0.81284` for "water plant" |
| `rdemo` | board | The paper's Figure-1 contrast at 128 px: wine glasses → wine glass (0.97006); cup only → cup (0.93395) |
| `r128`, `r192`, `r320` | board | "serve wine" at 128 / 192 / 320 px with the new defaults, logged with `tee` |
| `rrank` | board | Product ranking vs MiniLM-only ranking on "water plant" |
| `revidence` | board | All 14 tasks at 320 px with YOLOv8n (~47 min) |
| `revidence_s256` | board | All 14 tasks at 256 px with YOLOv8s (~65 min) — produced [`results/board_logs/final_14_tasks/`](../../results/board_logs/final_14_tasks/) |

> `revidence_s256` here uses image `000000158956` for task 8, while the logged run used `000000006177`. Use the copy in this folder as a template and set the image list to the ones you have.
