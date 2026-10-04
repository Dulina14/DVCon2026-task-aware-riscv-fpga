# test_data/

| Path | What it is |
|---|---|
| `input/demo_image.jpg` | The test photo (480 × 640, portrait) used in our board demos and in the PC quick-start; "serve wine" selects the **cup** in it |
| `expected/team43_minilm_golden.json` | Host reference for MiniLM scoring (used to validate Milestone B) |
| `expected/minilm_task_class_scores.json` | 14 × 80 suitability scores from the deployed INT8 MiniLM |
| `expected/MILESTONE_C_GOLDEN.txt` | Host `--sim` detection goldens for two test images at 128/192/256 px (authoritative for the board) |

The 14 photos used in the board evidence run are COCO val2014 images; their ids are listed in [`results/README.md`](../results/README.md). Download them from [cocodataset.org](https://cocodataset.org/#download).

COCO images are distributed under their original Flickr licences; see the [COCO terms of use](https://cocodataset.org/#termsofuse).
