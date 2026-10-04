"""
Flatten YOLOv8n into a linear op list the board can replay, and re-implement the
forward pass in NumPy from the fused weights only.

If the NumPy output matches torch's, the op list is a faithful description of the
network and the C interpreter can just replay it.

Ops emitted (all tensors are NCHW float32, N=1):
  CONV     in, w_idx, k, s, pad, act(silu|none), out
  MAXPOOL  in, k, s, pad, out          (SPPF, k=5 s=1 pad=2)
  UPSAMPLE in, scale, out              (nearest)
  CONCAT   [ins], out
  SLICE    in, c0, c1, out             (C2f split)
  ADD      a, b, out                   (Bottleneck shortcut)
"""
import numpy as np, torch, warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Bottleneck, Concat, Detect

# --------------------------------------------------------------- op list ---
class Graph:
    def __init__(self):
        self.ops = []
        self.weights = []          # list of (Cout, K) float32 arrays  + bias
        self.n_tensors = 0
    def new_tensor(self):
        self.n_tensors += 1
        return self.n_tensors - 1
    def add_weight(self, w, b):
        self.weights.append((w, b))
        return len(self.weights) - 1

def emit_conv(g, x, convmod, act):
    """convmod is a torch Conv2d (already BN-fused, has bias)."""
    W = convmod.weight.detach().numpy().astype(np.float64)   # [Cout,Cin,kh,kw]
    B = (convmod.bias.detach().numpy().astype(np.float64)
         if convmod.bias is not None else np.zeros(W.shape[0]))
    Cout, Cin, kh, kw = W.shape
    wi = g.add_weight(W.reshape(Cout, Cin * kh * kw), B)     # im2col order
    out = g.new_tensor()
    g.ops.append(dict(type='CONV', i=[x], o=out, w=wi, k=kh, s=convmod.stride[0],
                      p=convmod.padding[0], act=act, cin=Cin, cout=Cout))
    return out

def emit_convblock(g, x, mod):
    """ultralytics Conv = conv + (fused bn) + act"""
    act = 'silu' if isinstance(mod.act, nn.SiLU) else 'none'
    return emit_conv(g, x, mod.conv, act)

def emit_bottleneck(g, x, mod):
    y = emit_convblock(g, x, mod.cv1)
    y = emit_convblock(g, y, mod.cv2)
    if mod.add:
        out = g.new_tensor()
        g.ops.append(dict(type='ADD', i=[x, y], o=out))
        return out
    return y

def emit_c2f(g, x, mod):
    y = emit_convblock(g, x, mod.cv1)                 # -> 2c channels
    c = mod.c
    a = g.new_tensor(); g.ops.append(dict(type='SLICE', i=[y], o=a, c0=0,  c1=c))
    b = g.new_tensor(); g.ops.append(dict(type='SLICE', i=[y], o=b, c0=c, c1=2*c))
    outs = [a, b]
    cur = b
    for m in mod.m:
        cur = emit_bottleneck(g, cur, m)
        outs.append(cur)
    cc = g.new_tensor(); g.ops.append(dict(type='CONCAT', i=outs, o=cc))
    return emit_convblock(g, cc, mod.cv2)

def emit_sppf(g, x, mod):
    x1 = emit_convblock(g, x, mod.cv1)
    k = mod.m.kernel_size
    ys = [x1]
    cur = x1
    for _ in range(3):
        o = g.new_tensor()
        g.ops.append(dict(type='MAXPOOL', i=[cur], o=o, k=k, s=1, p=k // 2))
        ys.append(o); cur = o
    cc = g.new_tensor(); g.ops.append(dict(type='CONCAT', i=ys, o=cc))
    return emit_convblock(g, cc, mod.cv2)

def flatten(net):
    g = Graph()
    seq = net.model
    outputs = {}
    x = g.new_tensor()          # input image tensor id 0
    prev = x
    detect_inputs = None
    for i, mod in enumerate(seq):
        f = mod.f
        if isinstance(f, int):
            inp = prev if f == -1 else outputs[f]
        else:
            inp = [prev if j == -1 else outputs[j] for j in f]

        if isinstance(mod, Conv):        y = emit_convblock(g, inp, mod)
        elif isinstance(mod, C2f):       y = emit_c2f(g, inp, mod)
        elif isinstance(mod, SPPF):      y = emit_sppf(g, inp, mod)
        elif isinstance(mod, nn.Upsample):
            y = g.new_tensor(); g.ops.append(dict(type='UPSAMPLE', i=[inp], o=y, scale=2))
        elif isinstance(mod, Concat):
            y = g.new_tensor(); g.ops.append(dict(type='CONCAT', i=list(inp), o=y))
        elif isinstance(mod, Detect):
            detect_inputs = list(inp)
            heads = []
            for lvl in range(mod.nl):
                src = detect_inputs[lvl]
                b = src
                for sub in mod.cv2[lvl][:-1]: b = emit_convblock(g, b, sub)
                b = emit_conv(g, b, mod.cv2[lvl][-1], 'none')      # 4*reg_max
                c = src
                for sub in mod.cv3[lvl][:-1]: c = emit_convblock(g, c, sub)
                c = emit_conv(g, c, mod.cv3[lvl][-1], 'none')      # nc
                heads.append((b, c))
            g.detect = dict(heads=heads, nc=mod.nc, reg_max=mod.reg_max,
                            strides=[float(s) for s in mod.stride],
                            dfl_w=mod.dfl.conv.weight.detach().numpy().reshape(-1).astype(np.float64))
            y = None
        else:
            raise RuntimeError('unhandled module ' + type(mod).__name__)
        outputs[i] = y
        prev = y
    return g

# ------------------------------------------------------------ numpy exec ---
def silu(x): return x / (1.0 + np.exp(-x))

def conv2d(x, w, b, k, s, p):
    """x [C,H,W] float64, w [Cout, C*k*k], returns [Cout,Ho,Wo]"""
    C, H, W = x.shape
    Ho = (H + 2*p - k)//s + 1
    Wo = (W + 2*p - k)//s + 1
    if p:
        xp = np.zeros((C, H+2*p, W+2*p), x.dtype); xp[:, p:p+H, p:p+W] = x
    else:
        xp = x
    cols = np.empty((Ho*Wo, C*k*k), x.dtype)
    idx = 0
    for c in range(C):
        for i in range(k):
            for j in range(k):
                patch = xp[c, i:i+Ho*s:s, j:j+Wo*s:s]
                cols[:, idx] = patch.reshape(-1)
                idx += 1
    return (cols @ w.T + b).T.reshape(Cout_shape(w), Ho, Wo)

def Cout_shape(w): return w.shape[0]

def maxpool(x, k, s, p):
    C, H, W = x.shape
    xp = np.full((C, H+2*p, W+2*p), -np.inf, x.dtype); xp[:, p:p+H, p:p+W] = x
    Ho = (H + 2*p - k)//s + 1; Wo = (W + 2*p - k)//s + 1
    out = np.full((C, Ho, Wo), -np.inf, x.dtype)
    for i in range(k):
        for j in range(k):
            out = np.maximum(out, xp[:, i:i+Ho*s:s, j:j+Wo*s:s])
    return out

def run_numpy(g, img):
    T = {}
    T[0] = img
    for op in g.ops:
        t = op['type']
        if t == 'CONV':
            w, b = g.weights[op['w']]
            y = conv2d(T[op['i'][0]], w, b, op['k'], op['s'], op['p'])
            if op['act'] == 'silu': y = silu(y)
            T[op['o']] = y
        elif t == 'MAXPOOL':
            T[op['o']] = maxpool(T[op['i'][0]], op['k'], op['s'], op['p'])
        elif t == 'UPSAMPLE':
            T[op['o']] = np.repeat(np.repeat(T[op['i'][0]], op['scale'], 1), op['scale'], 2)
        elif t == 'CONCAT':
            T[op['o']] = np.concatenate([T[j] for j in op['i']], axis=0)
        elif t == 'SLICE':
            T[op['o']] = T[op['i'][0]][op['c0']:op['c1']]
        elif t == 'ADD':
            T[op['o']] = T[op['i'][0]] + T[op['i'][1]]
        else:
            raise RuntimeError(t)
    return T

def decode(g, T, imgsz):
    """Replicate Detect.forward inference path -> (boxes xyxy, scores[nc,A])"""
    d = g.detect
    nc, rm = d['nc'], d['reg_max']
    boxes_all, cls_all, anchors, strides = [], [], [], []
    for lvl, (bi, ci) in enumerate(d['heads']):
        bt, ct = T[bi], T[ci]
        C, H, W = bt.shape
        st = d['strides'][lvl]
        # anchor points: grid + 0.5
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        anchors.append(np.stack([gx.reshape(-1) + 0.5, gy.reshape(-1) + 0.5], 1))
        strides.append(np.full(H*W, st))
        b = bt.reshape(4, rm, H*W)
        e = np.exp(b - b.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        proj = np.arange(rm, dtype=np.float64)
        boxes_all.append((prob * proj[None, :, None]).sum(1))     # [4, HW]
        cls_all.append(1.0/(1.0+np.exp(-ct.reshape(nc, H*W))))
    dist = np.concatenate(boxes_all, 1)          # [4, A]
    cls  = np.concatenate(cls_all, 1)            # [nc, A]
    a = np.concatenate(anchors, 0).T             # [2, A]
    s = np.concatenate(strides, 0)               # [A]
    x1 = (a[0] - dist[0]) * s; y1 = (a[1] - dist[1]) * s
    x2 = (a[0] + dist[2]) * s; y2 = (a[1] + dist[3]) * s
    return np.stack([x1, y1, x2, y2], 1), cls

if __name__ == '__main__':
    IMGSZ = 128
    net = YOLO('/mnt/user-data/uploads/minilm-new-pipeline/yolov8n.pt').model.eval().float()
    net = net.fuse()
    g = flatten(net)
    print(f'ops={len(g.ops)}  convs={sum(1 for o in g.ops if o["type"]=="CONV")}  '
          f'tensors={g.n_tensors}  weights={len(g.weights)}')

    rng = np.random.default_rng(43)
    img = rng.random((3, IMGSZ, IMGSZ))
    with torch.no_grad():
        ref = net(torch.tensor(img[None], dtype=torch.float32))
    T = run_numpy(g, img)
    boxes, cls = decode(g, T, IMGSZ)

    r = ref[0][0].numpy()                       # [4+nc, A]
    ours = np.concatenate([np.stack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2,
                                     boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1]], 0), cls], 0)
    print('torch out', r.shape, 'ours', ours.shape)
    print('max abs diff boxes :', np.abs(r[:4] - ours[:4]).max())
    print('max abs diff scores:', np.abs(r[4:] - ours[4:]).max())
