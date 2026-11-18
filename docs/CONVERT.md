# Convert BF16 to MLX affine

On CUDA, TensorFold serves a converted checkpoint: a plain BF16 Qwen3.5 tree is decision-only there and is refused for
serving. On Metal, the dense loader reads BF16 too, but the measured profile serves 4-bit. Both backends load the same
converted tree. The converter is `tools/sensenova_loop_mlx.py`, carried in both patch series
(`patches/cuda/0008`, `patches/metal/0002`-`0004` and `0026`), so it is in every patched engine checkout. The Metal
copy also writes the vision tower; the CUDA copy is text only until CUDA images are qualified.

## Commands

```bash
pip install numpy torch safetensors          # CPU torch is enough
hf download sensenova/SenseNova-Loop-9B --revision 329a65f701ba8c48b73c30d0cbaa6f0449733673 \
  --local-dir SenseNova-Loop-9B
python3 ENGINE/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx4 --bits 4 --group 64
python3 ENGINE/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx8 --bits 8 --group 64
```

`ENGINE` is the directory `scripts/fetch-engine.sh` or `scripts/build-engine.sh` wrote. The converter reads one tensor
at a time, writes 1.8 GB shards plus `model.safetensors.index.json`, copies the tokenizer, chat template and generation
config, and records a round-trip check of its first quantized tensors in `tensorfold-convert.json`.

To test the norm classification: `cd ENGINE/tools && python3 -m pytest sensenova_loop_mlx_test.py`.

## What it writes

| source | written as |
| --- | --- |
| 2-D projections, `embed_tokens`, `lm_head` | MLX affine codes: `u32` words, `bf16` scales and biases per group of 64, `scale * code + bias` |
| `input_layernorm`, `post_attention_layernorm`, `model.norm`, `q_norm`, `k_norm` | `1 + w` in bf16 (`Qwen3_5RMSNorm` stores a zero-centred weight) |
| `linear_attn.norm` | as stored (`Qwen3_5RMSNormGated` multiplies by `w`) |
| `linear_attn.conv1d.weight` `[qkv, 1, kernel]` | `[qkv, kernel, 1]` |
| `A_log`, `dt_bias` (f32) | bf16 |
| mHC routers `mhc_attn`, `mhc_ffn` | fp32, unchanged |
| names | `model.language_model.*` to `language_model.model.*`, `lm_head.*` to `language_model.lm_head.*` |
| `model.visual.*` | Metal series (`0026`): `vision_tower.*`, 333 tensors in bf16, unquantized (0.85 GiB); CUDA series: left out |
| `loop_gate.logits` | left out: unused at inference |

Codes follow `zig/src/core/affine4_host.zig`'s `quantizeMlx` word for word: the group edge with the larger magnitude
sets the signed scale and the bias, the step is `(max - min) / (2^bits - 1)`, and codes round half away from zero.

The norm mapping decides whether the model works: without the `1 + w` fold those norms run about 20x small, and the
error shows at layer 0.
