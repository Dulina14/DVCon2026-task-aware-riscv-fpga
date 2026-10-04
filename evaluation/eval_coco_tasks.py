#!/usr/bin/env python3
"""
Team 43 / DVCon India 2026 -- COCO-Tasks evaluation.

Measures our lightweight task-driven object detection pipeline with the same
metric the paper uses: AP@0.5 per task against the preferred-object
annotations, averaged over the 14 tasks (mAP@0.5).

Reference: Sawatzky, Souri, Grund, Gall, "What Object Should I Use? - Task
Driven Object Detection", CVPR 2019, Tables 2 and 3.

Three box sources, so each of our numbers lands in the right column of their
table:

  --boxes gt        the ground-truth COCO objects        -> their "gt bbox" column
  --boxes faster    the paper's own detections_faster    -> their "Faster-RCNN detections"
  --boxes yolov8n   our YOLOv8n at --imgsz               -> the full lightweight pipeline

The first two need no images and no torch: they read JSON only. The third needs
ultralytics and the COCO val2014 images.

Scoring follows the paper: "the final confidence for mAP evaluation is obtained
by multiplying the detection score with the estimated probability". For gt boxes
there is no detection score, so the probability is used directly.

Our probability is MiniLM(task, class_name), read from a precomputed
14x80 table produced by the DEPLOYED INT8 package -- so this evaluates exactly
the numbers the board computes, not a separate float model.

Usage:
    python eval_coco_tasks.py --boxes gt
    python eval_coco_tasks.py --boxes faster
    python eval_coco_tasks.py --boxes yolov8n --imgsz 128
    python eval_coco_tasks.py --boxes yolov8n --imgsz 192 --limit 100
"""
import argparse, json, os, sys
from collections import defaultdict

# 80 contiguous class indices -> original COCO category ids
COCO80_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
              27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,
              51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,
              76,77,78,79,80,81,82,84,85,86,87,88,89,90]
COCO80_NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
 "boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
 "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
 "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",
 "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",
 "knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
 "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv",
 "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",
 "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
CATID2NAME = {c: n for c, n in zip(COCO80_IDS, COCO80_NAMES)}
TASK_IDS = list(range(1, 15))


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    w = x2 - x1; h = y2 - y1
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ap50(preds, gts, n_gt):
    """COCO-style AP@0.5, 101-point interpolation.
    preds: list of (score, image_id, box_xyxy) ; gts: {image_id: [box_xyxy,...]}"""
    if n_gt == 0:
        return float('nan')
    preds = sorted(preds, key=lambda p: -p[0])
    matched = defaultdict(set)
    tp = []; fp = []
    for score, img, box in preds:
        best_i, best_iou = -1, 0.0
        for i, g in enumerate(gts.get(img, [])):
            if i in matched[img]:
                continue
            v = iou_xyxy(box, g)
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= 0.5 and best_i >= 0:
            matched[img].add(best_i); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    ctp = 0; cfp = 0
    rec = []; prec = []
    for t, f in zip(tp, fp):
        ctp += t; cfp += f
        rec.append(ctp / n_gt)
        prec.append(ctp / (ctp + cfp))
    # make precision monotonically decreasing, then sample at 101 recall points
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])
    ap = 0.0
    j = 0
    for k in range(101):
        r = k / 100.0
        while j < len(rec) and rec[j] < r:
            j += 1
        ap += (prec[j] if j < len(prec) else 0.0)
    return ap / 101.0


def load_gt(ann_dir):
    """{task_id: {image_id: [preferred boxes xyxy]}} plus the image id list per task,
    plus {task_id: {image_id: set(class names of the preferred objects)}}."""
    gt = {}; imgs = {}; gtc = {}
    for t in TASK_IDS:
        p = os.path.join(ann_dir, f'task_{t}_test.json')
        d = json.load(open(p))
        g = defaultdict(list); c = defaultdict(set)
        for a in d['annotations']:
            if a.get('category_id') == 1:      # 1 = preferred by the majority
                x, y, w, h = a['bbox']
                g[a['image_id']].append([x, y, x + w, y + h])
                n = CATID2NAME.get(a.get('COCO_category_id'))
                if n:
                    c[a['image_id']].add(n)
        gt[t] = g; gtc[t] = c
        imgs[t] = [im['id'] for im in d['images']]
    return gt, imgs, gtc


def top1_selection(ids, src, gt_t, gtc_t, rank_fn):
    """What the demo actually shows: one box per image, the highest-ranked one.

    Returns (n_answerable, n_box_hit, n_class_hit, n_no_candidate).

      box hit    the marked box overlaps a preferred object at IoU >= 0.5
                 -- the user sees the right thing circled
      class hit  the marked box's class is one of the preferred classes in that
                 image, regardless of which instance was boxed. Separates a
                 ranker that reasoned correctly but was handed a bad box from a
                 ranker that chose the wrong kind of object entirely.

    Denominator is images that actually contain a preferred object; where there
    is none the question has no right answer and scoring it is meaningless.
    """
    n = hit = chit = nocand = orac = 0
    for iid in ids:
        gts = gt_t.get(iid, [])
        if not gts:
            continue
        n += 1
        cands = src.get(iid, [])
        if not cands:
            nocand += 1
            continue
        name, conf, box = max(cands, key=lambda c: rank_fn(c[0], c[1]))
        if max((iou_xyxy(box, g) for g in gts), default=0.0) >= 0.5:
            hit += 1
        if name in gtc_t.get(iid, ()):
            chit += 1
        # oracle: was the right answer even on the list? An upper bound on what
        # any ranker could score given these boxes. The gap between orac and hit
        # is the ranker's loss; the gap between orac and 1.0 is the detector's.
        if any(max((iou_xyxy(b, g) for g in gts), default=0.0) >= 0.5
               for _, _, b in cands):
            orac += 1
    return n, hit, chit, nocand, orac


def boxes_from_gt(ann_dir, t):
    """Every annotated COCO object in the task's test images, with its class."""
    d = json.load(open(os.path.join(ann_dir, f'task_{t}_test.json')))
    out = defaultdict(list)
    for a in d['annotations']:
        cid = a.get('COCO_category_id')
        name = CATID2NAME.get(cid)
        if name is None:
            continue
        x, y, w, h = a['bbox']
        out[a['image_id']].append((name, 1.0, [x, y, x + w, y + h]))
    return out


def load_dets(path):
    """Accept either a JSON array or newline-delimited JSON records."""
    txt = open(path).read().lstrip()
    if txt.startswith('['):
        return json.loads(txt)
    out, bad = [], 0
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            bad += 1          # a run killed mid-write leaves one torn record
    if bad:
        print(f'  note: skipped {bad} torn line(s) in {os.path.basename(path)}')
    return out


def boxes_from_detfile(path, keep_ids, conf=0.0):
    out = defaultdict(list)
    for d in load_dets(path):
        if d.get('category_id') == 0:      # placeholder: image had no detections
            continue
        if d['image_id'] not in keep_ids:
            continue
        name = CATID2NAME.get(d['category_id'])
        if name is None or d['score'] < conf:
            continue
        x, y, w, h = d['bbox']
        out[d['image_id']].append((name, float(d['score']), [x, y, x + w, y + h]))
    return out


def boxes_from_yolo(img_dir, image_ids, imgsz, conf, weights):
    from ultralytics import YOLO
    model = YOLO(weights)
    out = defaultdict(list)
    todo = [(i, os.path.join(img_dir, 'COCO_val2014_%012d.jpg' % i)) for i in image_ids]
    todo = [(i, p) for i, p in todo if os.path.exists(p)]
    B = 16
    for s in range(0, len(todo), B):
        chunk = todo[s:s+B]
        res = model.predict([p for _, p in chunk], imgsz=imgsz, conf=conf,
                            max_det=300, verbose=False)
        for (iid, _), r in zip(chunk, res):
            for b in r.boxes:
                out[iid].append((model.model.names[int(b.cls)], float(b.conf),
                                 [float(v) for v in b.xyxy[0].tolist()]))
        if s % (B*20) == 0:
            print(f'    {s}/{len(todo)} images', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\DVCon\datasets')
    ap.add_argument('--boxes', choices=['gt', 'faster', 'yolo_paper', 'yolov8n', 'detfile'],
                    default='gt',
                    help="detfile scores any COCO detection file, including the INT8 "
                         "detections the deployed pipeline writes with --detect-json")
    ap.add_argument('--detfile', default=None, help='detection file for --boxes detfile')
    ap.add_argument('--scores', default='minilm_task_class_scores.json')
    ap.add_argument('--imgsz', type=int, default=128)
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--weights', default='yolov8n.pt')
    ap.add_argument('--limit', type=int, default=0, help='images per task, 0 = all')
    ap.add_argument('--method', choices=['minilm','pickbest'], default='minilm',
                    help="pickbest reproduces the paper's pick-best-class baseline "
                         "(published 0.386 on gt boxes) as a check on the AP code")
    ap.add_argument('--minsup', type=int, default=50)
    ap.add_argument('--rank', choices=['auto', 'product', 'minilm', 'det'], default='auto',
                    help="how candidates are ordered. auto = the paper's rule "
                         "(MiniLM x detection score, or MiniLM alone for gt boxes). "
                         "minilm = MiniLM alone, which is what the deployed pipeline "
                         "uses by default. det = detection score alone, a control that "
                         "shows how much of the score is the language model at all")
    ap.add_argument('--images', default=None,
                    help='restrict every task to the image ids listed in this file '
                         '(one "<id> <path>" or "<id>" per line), so two detection '
                         'runs over the same subsample are compared on equal ground')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    ct = os.path.join(a.root, 'mscoco', 'coco-tasks')
    if not os.path.isdir(ct):
        ct = os.path.join(a.root, 'coco-tasks')
    ann_dir = os.path.join(ct, 'annotations')
    img_dir = os.path.join(a.root, 'mscoco', 'val2014')

    only = None
    if a.images:
        only = set()
        for l in open(a.images):
            l = l.strip()
            if l:
                only.add(int(l.split()[0]))
        print(f'restricted to {len(only)} image ids from {os.path.basename(a.images)}')

    tbl = json.load(open(a.scores))
    gt, task_imgs, gtc = load_gt(ann_dir)

    label = a.boxes if a.boxes != 'detfile' else 'detfile ' + os.path.basename(a.detfile)
    print(f'box source: {label}' + (f'  imgsz={a.imgsz}' if a.boxes == 'yolov8n' else '')
          + (f'  conf>={a.conf}' if a.boxes in ('detfile','faster','yolo_paper') else '')
          + f'  rank={a.rank}')
    results = {}
    sel = {}
    T_n = T_hit = T_chit = T_no = T_or = 0
    for t in TASK_IDS:
        ids = task_imgs[t]
        if a.limit:
            ids = ids[:a.limit]
        if only is not None:
            ids = [i for i in ids if i in only]
        keep = set(ids)

        if a.boxes == 'gt':
            src = boxes_from_gt(ann_dir, t)
        elif a.boxes == 'faster':
            src = boxes_from_detfile(os.path.join(ct, 'detections_faster.json'), keep)
        elif a.boxes == 'yolo_paper':
            src = boxes_from_detfile(os.path.join(ct, 'detections_yolo.json'), keep)
        elif a.boxes == 'detfile':
            if not a.detfile: raise SystemExit('--boxes detfile needs --detfile PATH')
            src = boxes_from_detfile(a.detfile, keep, a.conf)
        else:
            src = boxes_from_yolo(img_dir, ids, a.imgsz, a.conf, a.weights)

        if a.method == 'pickbest':
            # class preference rate on the train split, then per image keep only
            # the highest-rated class that is actually present (the paper's rule)
            tr = json.load(open(os.path.join(ann_dir, f'task_{t}_train.json')))
            num = defaultdict(int); den = defaultdict(int)
            for an in tr['annotations']:
                n = CATID2NAME.get(an.get('COCO_category_id'))
                if n is None: continue
                den[n] += 1; num[n] += (an.get('category_id') == 1)
            rate = {n: num[n]/den[n] for n in den if den[n] >= a.minsup}
            preds = []
            for iid in ids:
                bs = src.get(iid, [])
                present = [n for n, _, _ in bs if n in rate]
                if not present: continue
                bestc = max(present, key=lambda n: rate[n])
                for name, det_conf, box in bs:
                    if name == bestc:
                        preds.append((1.0 if a.boxes == 'gt' else det_conf, iid, box))
        else:
            scores = tbl[str(t)]['scores']
            mode = a.rank
            if mode == 'auto':
                mode = 'minilm' if a.boxes == 'gt' else 'product'
            if mode == 'minilm':
                rank_fn = lambda name, c: scores.get(name, 0.0)
            elif mode == 'det':
                rank_fn = lambda name, c: c
            else:
                rank_fn = lambda name, c: scores.get(name, 0.0) * c
            preds = []
            for iid in ids:
                for name, det_conf, box in src.get(iid, []):
                    preds.append((rank_fn(name, det_conf), iid, box))

        g = {i: gt[t].get(i, []) for i in ids}
        n_gt = sum(len(v) for v in g.values())
        v = ap50(preds, g, n_gt)
        results[t] = v

        if a.method != 'pickbest':
            n, hit, chit, nc, orc = top1_selection(ids, src, gt[t], gtc[t], rank_fn)
            sel[t] = {'images': n, 'box_hit': hit, 'class_hit': chit,
                      'no_candidate': nc, 'oracle': orc}
            T_n += n; T_hit += hit; T_chit += chit; T_no += nc; T_or += orc
            s1 = f'  top1 {hit/n:.3f} box / {chit/n:.3f} class' if n else '  top1 n/a'
        else:
            s1 = ''
        print(f'  task {t:2d}  {tbl[str(t)]["prompt"]:<42s} AP@0.5 = {v:.4f}'
              f'   ({n_gt} preferred objects, {len(preds)} candidates){s1}')

    vals = [v for v in results.values() if v == v]
    mAP = sum(vals) / len(vals)
    print(f'\n  mAP@0.5 = {mAP:.4f}   over {len(vals)} tasks')

    if T_n:
        print(f'\n  top-1 selection -- the single box the demo marks, over {T_n} images '
              f'that contain a preferred object')
        print(f'    box  correct  {T_hit}/{T_n} = {T_hit/T_n:.4f}   '
              f'(marked box overlaps a preferred object at IoU >= 0.5)')
        print(f'    class correct {T_chit}/{T_n} = {T_chit/T_n:.4f}   '
              f'(marked box is of a preferred class, any instance)')
        print(f'    nothing to mark {T_no}/{T_n} = {T_no/T_n:.4f}   '
              f'(detector returned no candidate at all)')
        print(f'    ORACLE ceiling {T_or}/{T_n} = {T_or/T_n:.4f}   '
              f'(a preferred object was somewhere on the candidate list)')
        if T_or:
            print(f'    -> of the {T_or} images where the right answer was available, '
                  f'the ranker picked it {T_hit}/{T_or} = {T_hit/T_or:.4f} of the time')
            print(f'    -> attribution: detector loses {1-T_or/T_n:.3f}, '
                  f'ranker loses {(T_or-T_hit)/T_n:.3f}, of the {1-T_hit/T_n:.3f} total miss')

    print('\n  paper Table 2 for reference:')
    print('    gt bbox      : pick-best-class 0.386 | classification 0.616 | proposed 0.742')
    print('    detections   : pick-best-class 0.141 | classification 0.288 | proposed 0.326')
    print('    Table 3 (g) no visual input : 0.589 gt / 0.237 detections')

    if a.out:
        json.dump({'boxes': a.boxes, 'imgsz': a.imgsz, 'conf': a.conf, 'rank': a.rank,
                   'per_task': results, 'mAP50': mAP,
                   'select_per_task': sel,
                   'select': {'images': T_n, 'box_hit': T_hit, 'class_hit': T_chit,
                              'no_candidate': T_no,
                              'oracle': T_or,
                              'box_acc': (T_hit/T_n if T_n else None),
                              'class_acc': (T_chit/T_n if T_n else None),
                              'oracle_acc': (T_or/T_n if T_n else None)}},
                  open(a.out, 'w'), indent=1)
        print(f'  wrote {a.out}')


if __name__ == '__main__':
    main()
