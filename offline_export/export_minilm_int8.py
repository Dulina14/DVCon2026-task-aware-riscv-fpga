#!/usr/bin/env python3
"""
Team 43 / DVCon India 2026 Stage 3B
One-time offline export of the fine-tuned MiniLM task-object scorer to an
INT8 package for the VEGA + 8x8 accelerator board runtime.

RUN THIS ON THE WINDOWS PC (torch + transformers are installed there).

    python export_minilm_int8.py

Outputs (next to this script):
    team43_minilm_int8.bin      quantized weight package for the board
    team43_minilm_manifest.json human-readable manifest
    team43_minilm_golden.json   host golden reference for validation

WHAT IS LEGAL HERE (per the standalone requirement):
  * quantizing constant trained weights                       -> offline OK
  * embedding the 80 fixed COCO label names                   -> offline OK
    (they do not depend on the runtime image or prompt, and the result is
     mathematically identical to running them at runtime)
  * producing a golden reference for testing                  -> offline OK
WHAT IS NOT DONE HERE:
  * the runtime task prompt is NOT embedded or scored here.
    task_emb and the final selection are computed on the board.
"""

import json, struct, sys
from pathlib import Path

import numpy as np
import torch

CKPT = Path(r"D:\DVCon\task-aware-object-detection-pipeline\checkpoints"
            r"\finetuned_text\minilm_task_text_best.pt")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path(__file__).resolve().parent

# Deployed runtime bound. inference.py used MAX_LENGTH=32; "task: serve wine"
# is ~6 WordPiece tokens. 8 is a documented reduction, recorded in the manifest.
MAX_TOKENS = 8

H, L, HEADS, FFN = 384, 6, 12, 1536

COCO80 = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
 "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
 "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
 "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
 "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
 "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
 "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
 "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
 "remote","keyboard","cell phone","microwave","oven","toaster","sink",
 "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
 "toothbrush"]


# --------------------------------------------------------------- forward ---
def layer_norm(x, w, b, eps=1e-12):
    m = x.mean(-1, keepdim=True)
    v = x.var(-1, unbiased=False, keepdim=True)
    return (x - m) / torch.sqrt(v + eps) * w + b


def encode(sd, ids, mask):
    """Manual MiniLM forward from the state_dict. No HF model needed."""
    p = "encoder."
    e = sd[p + "embeddings.word_embeddings.weight"][ids]
    e = e + sd[p + "embeddings.position_embeddings.weight"][:ids.shape[1]]
    e = e + sd[p + "embeddings.token_type_embeddings.weight"][0]
    x = layer_norm(e, sd[p + "embeddings.LayerNorm.weight"],
                      sd[p + "embeddings.LayerNorm.bias"])

    add = (1.0 - mask[:, None, None, :].float()) * -10000.0
    for i in range(L):
        q = p + f"encoder.layer.{i}."
        Q = x @ sd[q + "attention.self.query.weight"].T + sd[q + "attention.self.query.bias"]
        K = x @ sd[q + "attention.self.key.weight"].T   + sd[q + "attention.self.key.bias"]
        V = x @ sd[q + "attention.self.value.weight"].T + sd[q + "attention.self.value.bias"]
        B, T, _ = Q.shape
        d = H // HEADS
        sh = lambda t: t.view(B, T, HEADS, d).transpose(1, 2)
        s = (sh(Q) @ sh(K).transpose(-1, -2)) / (d ** 0.5) + add
        ctx = (s.softmax(-1) @ sh(V)).transpose(1, 2).reshape(B, T, H)
        a = ctx @ sd[q + "attention.output.dense.weight"].T + sd[q + "attention.output.dense.bias"]
        x = layer_norm(a + x, sd[q + "attention.output.LayerNorm.weight"],
                              sd[q + "attention.output.LayerNorm.bias"])
        h = x @ sd[q + "intermediate.dense.weight"].T + sd[q + "intermediate.dense.bias"]
        h = torch.nn.functional.gelu(h)
        o = h @ sd[q + "output.dense.weight"].T + sd[q + "output.dense.bias"]
        x = layer_norm(o + x, sd[q + "output.LayerNorm.weight"],
                              sd[q + "output.LayerNorm.bias"])

    m = mask.unsqueeze(-1).expand(x.size()).float()
    emb = (x * m).sum(1) / torch.clamp(m.sum(1), min=1e-9)
    return emb / torch.clamp(emb.norm(dim=-1, keepdim=True), min=1e-9)


def score(sd, t_emb, o_emb):
    f = torch.cat([t_emb, o_emb, (t_emb - o_emb).abs(), t_emb * o_emb], -1)
    h = torch.relu(f @ sd["head.0.weight"].T + sd["head.0.bias"])
    return torch.sigmoid((h @ sd["head.3.weight"].T + sd["head.3.bias"]).squeeze(-1))


# ------------------------------------------------------------ quantize ----
def q8(t):
    a = t.detach().float().cpu().numpy()
    s = float(np.abs(a).max()) / 127.0
    if s == 0.0:
        s = 1.0
    return np.clip(np.rint(a / s), -127, 127).astype(np.int8), s


def main():
    if not CKPT.exists():
        sys.exit(f"checkpoint not found: {CKPT}")
    print("loading checkpoint ...")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k: v.float() for k, v in sd.items()}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- fixed COCO label embeddings (input-independent -> offline legal) ---
    print("embedding 80 fixed COCO labels ...")
    texts = [f"object: {c}" for c in COCO80]
    enc = tok(texts, padding="max_length", truncation=True,
              max_length=MAX_TOKENS, return_tensors="pt")
    with torch.no_grad():
        obj = encode(sd, enc["input_ids"], enc["attention_mask"])

    # ---- host golden for one test prompt (validation only) -----------------
    task = "serve wine"
    te = tok([f"task: {task}"], padding="max_length", truncation=True,
             max_length=MAX_TOKENS, return_tensors="pt")
    with torch.no_grad():
        t_emb = encode(sd, te["input_ids"], te["attention_mask"])
        sc = score(sd, t_emb.expand(80, -1), obj)
    order = torch.argsort(sc, descending=True)
    print(f'\nhost golden, task="{task}":')
    for i in order[:8]:
        print(f"   {COCO80[i]:<16s} {sc[i]:.6f}")

    # ---- build the INT8 package -------------------------------------------
    print("\nquantizing ...")
    tensors, scales = [], {}

    def add(name, t):
        q, s = q8(t)
        tensors.append((name, q))
        scales[name] = s

    add("word_emb", sd["encoder.embeddings.word_embeddings.weight"])
    add("pos_emb", sd["encoder.embeddings.position_embeddings.weight"][:MAX_TOKENS])
    add("tok_type", sd["encoder.embeddings.token_type_embeddings.weight"])
    for i in range(L):
        q = f"encoder.encoder.layer.{i}."
        for tag, key in (("q", "attention.self.query"), ("k", "attention.self.key"),
                         ("v", "attention.self.value"), ("o", "attention.output.dense"),
                         ("f1", "intermediate.dense"), ("f2", "output.dense")):
            add(f"L{i}.{tag}.w", sd[q + key + ".weight"])
    add("head0.w", sd["head.0.weight"])
    add("head3.w", sd["head.3.weight"])
    add("obj_emb", obj)

    # FP32 side tables (small: LayerNorm params, all biases, scales)
    fp32 = {}
    for k, v in sd.items():
        if (".bias" in k) or ("LayerNorm" in k):
            fp32[k] = v.detach().numpy().astype(np.float32)

    blob = bytearray()
    index = []
    for name, q in tensors:
        index.append({"name": name, "dtype": "int8", "shape": list(q.shape),
                      "offset": len(blob), "scale": scales[name]})
        blob += q.tobytes()
    for name, a in fp32.items():
        index.append({"name": name, "dtype": "fp32", "shape": list(a.shape),
                      "offset": len(blob), "scale": 1.0})
        blob += a.tobytes()

    (OUT / "team43_minilm_int8.bin").write_bytes(bytes(blob))

    manifest = {
        "model": MODEL_NAME, "checkpoint": str(CKPT),
        "hidden": H, "layers": L, "heads": HEADS, "ffn": FFN,
        "max_tokens": MAX_TOKENS, "note_max_length_original": 32,
        "vocab_size": int(sd["encoder.embeddings.word_embeddings.weight"].shape[0]),
        "coco_classes": COCO80,
        "quantization": "per-tensor symmetric int8, INT32 accumulate",
        "package_bytes": len(blob),
        "tensors": index,
        "board_ops": ["tokenize", "embedding lookup", "LayerNorm", "softmax",
                      "GELU", "residual", "mean-pool", "L2-norm", "sigmoid"],
        "accelerator_ops": ["Q/K/V/O projections", "FFN1", "FFN2",
                            "QK^T", "A*V", "head.0", "head.3"],
    }
    (OUT / "team43_minilm_manifest.json").write_text(json.dumps(manifest, indent=2))

    golden = {"task": task,
              "scores": {COCO80[i]: float(sc[i]) for i in range(80)},
              "top8": [COCO80[i] for i in order[:8]],
              "task_emb_first8": [float(v) for v in t_emb[0][:8]]}
    (OUT / "team43_minilm_golden.json").write_text(json.dumps(golden, indent=2))

    print(f"\nwrote team43_minilm_int8.bin       {len(blob):,} bytes")
    print("wrote team43_minilm_manifest.json")
    print("wrote team43_minilm_golden.json")


if __name__ == "__main__":
    main()
