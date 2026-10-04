# DVCon India 2026 Design Contest - Stage 2 Report

Team Silicon Syndicate, Team 43  
Team 43 — Silicon Syndicate, University of Moratuwa  
University of Moratuwa, Sri Lanka

---

## Page 1 - Approach Description

### Task-Aware Object Selection Pipeline

The objective of this stage was to convert the Stage 1 task-aware object detection proposal into a working software pipeline. The system receives an RGB image and a natural language task prompt, detects the candidate objects in the scene, and selects the object that is most semantically suitable for the given task. Unlike a conventional object detector, the final output is not simply the highest-confidence detected object. Instead, the pipeline reasons over the relationship between the detected object classes and the task description, then returns the selected object, its bounding box, and a task-relevance score.

The implemented software pipeline contains three main stages:

1. **Object detection:** The input image is processed using YOLOv8x to detect visible COCO objects and produce bounding boxes, class labels, and confidence scores.
2. **Text-task understanding:** The input task prompt is encoded using a fine-tuned MiniLM-based text model. The model evaluates the semantic compatibility between the task prompt and each detected object class.
3. **Task-aware ranking:** Each candidate object is ranked using the MiniLM relevance score. If the highest score is above the rejection threshold, the corresponding object is selected; otherwise, the system reports that no suitable object is available.

The software implementation follows the same functional goal proposed in Stage 1: identify the object that best satisfies a task such as "serve wine", "open parcel", "water plant", or "sit comfortably". The main difference is that the Stage 2 implementation focuses first on validating the complete AI pipeline in software before hardware acceleration. The current program produces JSON output for automated evaluation and annotated images for visual inspection.

### Modifications From the Stage 1 Proposal

In Stage 1, the proposed pipeline used YOLOv8n for object detection and MobileCLIP for vision-language reasoning. During implementation, we modified both model choices to improve the practical accuracy and resource behavior of the software pipeline.

**YOLOv8n was replaced with YOLOv8x.** YOLOv8n was selected initially because it is lightweight and suitable for edge deployment. However, in the implemented task-aware pipeline, object detection recall became a critical requirement. If the correct object is missed during detection, the semantic ranking stage cannot recover it. Therefore, we used YOLOv8x to improve candidate-object detection quality, especially for small or low-confidence task-relevant objects. To preserve recall, the detection confidence threshold was kept low, allowing MiniLM to perform the final task-aware filtering.

**MobileCLIP was replaced with MiniLM.** MobileCLIP was originally considered because it jointly reasons over image and text embeddings. In the implemented pipeline, YOLO already provides object localization and object class names, so the semantic decision can be reduced to matching the task prompt against the detected object candidates. MiniLM is well suited for this text-level ranking problem because it is compact, fast, and easier to fine-tune on COCO-Tasks style prompt-object pairs. This gave better resource utilization than running a full image-text model for every object crop, while still preserving the semantic reasoning required for task-aware object selection.

This revised design keeps the original hardware-software co-design direction. YOLO remains the compute-heavy vision component that can later benefit from convolution acceleration, while MiniLM provides a lightweight decision module that can run efficiently as software on the processor side or be optimized separately if required.

---

## Page 2 - Results Snapshot

### Software Implementation Summary

The Stage 2 software prototype was implemented as an end-to-end inference program. For each test image, the program performs object detection, ranks all detected candidates against the task prompt, and writes both machine-readable and visual outputs.

**Implemented outputs:**

- Selected object class
- Bounding box coordinates
- YOLO confidence score
- MiniLM semantic relevance score
- Final status: selected or below threshold
- Annotated result image for visual verification

### Evaluation Snapshot

The pipeline was evaluated on a balanced 1000-sample subset derived from the COCO-Tasks test data. The evaluation included both positive cases, where a suitable task object is expected to exist, and negative cases, where the system should reject unsuitable detections.

| Metric | Result |
|---|---:|
| Total evaluated samples | 1000 |
| Overall accuracy | 72.3% |
| Positive samples | 501 |
| Positive top-1 accuracy | 85.43% |
| Positive top-1 accuracy when expected object was detected | 87.53% |
| Expected object in top-5 when detected | 97.96% |
| Negative samples | 499 |
| Negative rejection accuracy | 59.12% |
| Samples selected by pipeline | 699 |
| Samples rejected below threshold | 301 |

These results show that the implemented pipeline performs strongly when the correct candidate object is present in the YOLO detections. The high top-5 availability indicates that the detector and candidate generation stage usually includes the correct object, while the MiniLM ranker is able to select the correct object in most positive cases. The lower negative rejection accuracy shows that threshold calibration remains an important improvement area, since some objects receive a meaningful semantic score even when the dataset label expects no suitable object.

### Example Result Snapshot

**Example prompt:** `serve wine`  
**Detected candidates:** motorcycle, dog, cup, chair, person, bicycle  
**Selected object:** `cup`  
**Selected bounding box:** `[312.66, 75.31, 338.38, 98.59]`  
**YOLO confidence:** `0.1165`  
**MiniLM relevance score:** `0.9340`  
**Final status:** `selected`

Insert annotated result image here:

`result.jpg` or selected images from `results/`

Insert evaluation visuals here:

- YOLO training result plot: `runs/detect/yolo_coco_tasks/results.png`
- Confusion matrix: `runs/detect/yolo_coco_tasks/confusion_matrix.png`
- Sample annotated outputs: `results/*.jpg`

### Key Observations

The change from YOLOv8n to YOLOv8x improved the reliability of candidate detection, which is essential because missing the correct object at the detection stage prevents correct task-aware selection. The change from MobileCLIP to MiniLM made the semantic stage more efficient for the implemented software pipeline by using the detected object class names directly instead of repeatedly processing image crops through a vision-language model.

The current implementation successfully demonstrates the complete task-aware object selection flow in software. The next improvement steps are to tune the rejection threshold, compare YOLOv8x against smaller fine-tuned YOLO variants for deployment trade-offs, and map the compute-heavy vision stage toward the FPGA acceleration architecture proposed in Stage 1.
