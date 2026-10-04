"""INT8 simulation of the flattened YOLOv8n, compared against FP32 on real images."""
import numpy as np, torch, warnings, sys
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from yolo_flatten import flatten, run_numpy, decode, silu, maxpool, Cout_shape

IMGSZ = int(sys.argv[1]) if len(sys.argv) > 1 else 128
PERCH = True          # per-output-channel weight scales (free: dequant is per column)

def letterbox(path, sz):
    from PIL import Image
    im = Image.open(path).convert('RGB')
    w, h = im.size
    r = min(sz / w, sz / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (sz, sz), (114, 114, 114))
    canvas.paste(im, ((sz - nw) // 2, (sz - nh) // 2))
    a = np.asarray(canvas, dtype=np.float64).transpose(2, 0, 1) / 255.0
    return a, r, ((sz - nw) // 2, (sz - nh) // 2)

def im2col(x, k, s, p):
    C, H, W = x.shape
    Ho = (H + 2*p - k)//s + 1; Wo = (W + 2*p - k)//s + 1
    if p:
        xp = np.zeros((C, H+2*p, W+2*p), x.dtype); xp[:, p:p+H, p:p+W] = x
    else: xp = x
    cols = np.empty((Ho*Wo, C*k*k), x.dtype); idx = 0
    for c in range(C):
        for i in range(k):
            for j in range(k):
                cols[:, idx] = xp[c, i:i+Ho*s:s, j:j+Wo*s:s].reshape(-1); idx += 1
    return cols, Ho, Wo

def q_act(x):
    s = np.abs(x).max() / 127.0
    if s == 0: s = 1.0
    return np.clip(np.rint(x / s), -127, 127), s

def q_w(w):
    if PERCH:
        s = np.abs(w).max(axis=1) / 127.0
        s[s == 0] = 1.0
        return np.clip(np.rint(w / s[:, None]), -127, 127), s
    s = np.abs(w).max() / 127.0
    if s == 0: s = 1.0
    return np.clip(np.rint(w / s), -127, 127), np.full(w.shape[0], s)

def run_int8(g, img):
    T = {0: img}
    for op in g.ops:
        t = op['type']
        if t == 'CONV':
            w, b = g.weights[op['w']]
            cols, Ho, Wo = im2col(T[op['i'][0]], op['k'], op['s'], op['p'])
            xq, sx = q_act(cols)
            wq, sw = q_w(w)
            acc = xq.astype(np.int64) @ wq.astype(np.int64).T          # [M, Cout]
            bi = np.rint(b / (sx * sw))
            y = (acc + bi[None, :]) * (sx * sw)[None, :]
            y = y.T.reshape(Cout_shape(w), Ho, Wo)
            if op['act'] == 'silu': y = silu(y)
            T[op['o']] = y
        elif t == 'MAXPOOL':  T[op['o']] = maxpool(T[op['i'][0]], op['k'], op['s'], op['p'])
        elif t == 'UPSAMPLE': T[op['o']] = np.repeat(np.repeat(T[op['i'][0]], 2, 1), 2, 2)
        elif t == 'CONCAT':   T[op['o']] = np.concatenate([T[j] for j in op['i']], axis=0)
        elif t == 'SLICE':    T[op['o']] = T[op['i'][0]][op['c0']:op['c1']]
        elif t == 'ADD':      T[op['o']] = T[op['i'][0]] + T[op['i'][1]]
    return T

def nms(boxes, scores, iou_thr=0.45):
    order = np.argsort(-scores); keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = np.clip(xx2-xx1,0,None) * np.clip(yy2-yy1,0,None)
        a1 = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        a2 = (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1])
        iou = inter/(a1+a2-inter+1e-9)
        order = order[1:][iou <= iou_thr]
    return keep

def detections(g, T, names, conf=0.25):
    boxes, cls = decode(g, T, IMGSZ)
    best_c = cls.argmax(0); best_s = cls.max(0)
    m = best_s >= conf
    if not m.any(): return []
    b, s, c = boxes[m], best_s[m], best_c[m]
    out = []
    for ci in np.unique(c):
        sel = c == ci
        keep = nms(b[sel], s[sel])
        for k in keep: out.append((names[int(ci)], float(s[sel][k]), b[sel][k]))
    return sorted(out, key=lambda r: -r[1])

if __name__ == '__main__':
    y = YOLO('/mnt/user-data/uploads/minilm-new-pipeline/yolov8n.pt')
    names = y.model.names
    net = y.model.eval().float().fuse()
    g = flatten(net)
    import os
    ASSETS = os.path.join(os.path.dirname(__import__('ultralytics').__file__), 'assets')
    for img_name in ('bus.jpg', 'zidane.jpg'):
        path = os.path.join(ASSETS, img_name)
        img, r, pad = letterbox(path, IMGSZ)
        print(f'\n=== {img_name} @ {IMGSZ} ===')
        for tag, runner in (('FP32', run_numpy), ('INT8', run_int8)):
            T = runner(g, img)
            det = detections(g, T, names)
            print(f'  {tag}: ' + (', '.join(f'{n} {s:.3f}' for n, s, _ in det[:6]) or 'nothing'))
