#!/usr/bin/env python3
"""Write vocab.txt (one WordPiece token per line, ordered by token id).

    python dump_vocab.py

The board tokenizes the runtime task prompt with this file, so it is part of
the deployment package. It is model metadata, not runtime-input-derived.
"""
from pathlib import Path
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent / "vocab.txt"

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
vocab = tok.get_vocab()                      # {token: id}
ordered = sorted(vocab.items(), key=lambda kv: kv[1])

# sanity: ids must be contiguous 0..N-1
ids = [i for _, i in ordered]
assert ids == list(range(len(ids))), "vocab ids are not contiguous"

OUT.write_text("\n".join(t for t, _ in ordered) + "\n", encoding="utf-8")

print(f"wrote {OUT}")
print(f"tokens        : {len(ordered)}")
print(f"bytes         : {OUT.stat().st_size}")
for name in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"):
    print(f"  {name:<8s} id={vocab.get(name)}")
print("check tokenisation of the demo prompt:")
enc = tok("task: serve wine", padding="max_length", truncation=True,
          max_length=8, return_tensors=None)
print("  ids  :", enc["input_ids"])
print("  mask :", enc["attention_mask"])
print("  toks :", tok.convert_ids_to_tokens(enc["input_ids"]))
