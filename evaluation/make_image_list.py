#!/usr/bin/env python3
"""Write '<image_id> <path>' lines for the COCO-Tasks test images.

The union of the fourteen task test splits is 9,505 distinct val2014 images
(900 per task, mostly disjoint). Running the detector once over the union and
caching the result lets all fourteen tasks be scored from one pass, and lets
the confidence threshold be swept afterwards for free.

  python make_image_list.py --root D:\\DVCon\\datasets --out images_all.txt
  python make_image_list.py --root ... --per-task 200 --out images_200.txt
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=r'D:\DVCon\datasets')
ap.add_argument('--out', default='images_all.txt')
ap.add_argument('--per-task', type=int, default=0, help='cap per task, 0 = all 900')
ap.add_argument('--img-dir', default=None)
a = ap.parse_args()

ct = os.path.join(a.root, 'mscoco', 'coco-tasks')
if not os.path.isdir(ct):
    ct = os.path.join(a.root, 'coco-tasks')
ann = os.path.join(ct, 'annotations')
img_dir = a.img_dir or os.path.join(a.root, 'mscoco', 'val2014')

seen, rows, missing = set(), [], 0
for t in range(1, 15):
    d = json.load(open(os.path.join(ann, 'task_%d_test.json' % t)))
    ids = [im['id'] for im in d['images']]
    if a.per_task:
        ids = ids[:a.per_task]
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        p = os.path.join(img_dir, 'COCO_val2014_%012d.jpg' % i)
        if not os.path.exists(p):
            missing += 1
            continue
        rows.append('%d %s' % (i, p))

with open(a.out, 'w') as f:
    f.write('\n'.join(rows) + '\n')

print('%d distinct images written to %s' % (len(rows), a.out))
if missing:
    print('%d image ids had no file under %s' % (missing, img_dir))
