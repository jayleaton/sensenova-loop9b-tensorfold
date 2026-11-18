#!/usr/bin/env python3
"""Make the CUDA serving tree: a converted tree's files linked, the BF16 vision tower as one new shard, a routed config.

    python3 scripts/serve-tree.py ENGINE_DIR QUANT_DIR BF16_DIR DEST [YARN]

QUANT_DIR is the converter's output (tools/sensenova_loop_mlx.py), BF16_DIR the published checkpoint. The converter
keeps model_type `sensenova_loop` and drops `model.visual.*`; the CUDA server opens the family by model_type `qwen3_5`
and serves the tower in BF16. YARN (for example 4) stretches the config's YaRN factor over the trained window:
4 gives 1,048,576 positions. --rope-yarn cannot do this, because the checkpoint's config already names YaRN (factor 2).
DEST must not exist; QUANT_DIR and BF16_DIR are only read.
"""
import json
import os
import struct
import sys

if len(sys.argv) not in (5, 6):
    sys.exit(__doc__)
engine, quant, bf16, dst = sys.argv[1:5]
yarn = float(sys.argv[5]) if len(sys.argv) == 6 else None
if os.path.lexists(dst):
    sys.exit(f"{dst} already exists")
sys.path.insert(0, os.path.join(engine, "tools"))
from checkpoint_view import view, writable  # noqa: E402

view(quant, dst)


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


with open(os.path.join(bf16, "model.safetensors.index.json")) as f:
    wmap = json.load(f)["weight_map"]
names = sorted(k for k in wmap if k.startswith("model.visual."))
meta, blobs, at = {}, [], 0
for name in names:
    shard = os.path.join(bf16, wmap[name])
    h, base = header(shard)
    t = h[name]
    a, b = t["data_offsets"]
    with open(shard, "rb") as f:
        f.seek(base + a)
        blobs.append(f.read(b - a))
    meta[name] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [at, at + b - a]}
    at += b - a
meta["__metadata__"] = {"format": "pt"}
head = json.dumps(meta, separators=(",", ":")).encode()
head += b" " * (-len(head) % 8)
shard = "model-visual.safetensors"
with open(writable(dst, shard, [quant, bf16]), "wb") as f:
    f.write(struct.pack("<Q", len(head)) + head)
    for blob in blobs:
        f.write(blob)

with open(os.path.join(quant, "model.safetensors.index.json")) as f:
    ix = json.load(f)
ix["weight_map"].update({n: shard for n in names})
ix.setdefault("metadata", {})["total_size"] = ix.get("metadata", {}).get("total_size", 0) + at
with open(writable(dst, "model.safetensors.index.json", [quant, bf16]), "w") as f:
    json.dump(ix, f, indent=1)

with open(os.path.join(quant, "config.json")) as f:
    c = json.load(f)
c["sensenova_architecture"] = c.get("model_type")
c["model_type"] = "qwen3_5"
if yarn:
    t = c["text_config"]
    rope = t.get("rope_parameters") or t["rope_scaling"]
    rope["factor"] = yarn
    t["max_position_embeddings"] = int(rope["original_max_position_embeddings"] * yarn)
with open(writable(dst, "config.json", [quant, bf16]), "w") as f:
    json.dump(c, f, indent=2)
print(dst, len(names), "tower tensors,", at >> 20, "MiB; model_type qwen3_5; yarn", yarn or "as stored")
