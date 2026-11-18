# Metal: Mac Studio M5 Ultra

The dense `qwen27` Metal family executes the loop plan and the two-stream mHC, with YaRN RoPE added for this
checkpoint. The series also carries the recipe TensorFold keeps for it,
`docs/recipes/sensenova-loop-9b-metal.md`, in the patched checkout.

| | |
| --- | --- |
| Hardware | Mac Studio M5 Ultra, macOS 27.2 |
| Engine | upstream TensorFold `6d0a128b0cde35aa091bda933577cfced3519b8a` (on `main`, 2026-11-17) + [`patches/metal/`](../patches/metal/) `0001`-`0026` |
| Engine tree | `f882fabdaefda68b13e03fb2eba66390d7a15942`: the measured engine (`0001`-`0025`) with `0026`'s formatting fixes, token for token the same code |
| Weights | MLX affine 4-bit g64 ([CONVERT](CONVERT.md)) |

## Build

On the Mac, with Xcode's macOS SDK and Metal toolchain and Zig 0.17.0:

```bash
scripts/build-engine.sh metal engine-metal
```

That runs `scripts/fetch-engine.sh metal engine-metal`, then `zig build native -Dcpu=apple_m1` in `engine-metal`. The
`apple_m1` target keeps the binary compatible with M1 through M5; the numbers below are from an M5 Ultra only.

## Serve

```bash
python3 engine-metal/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx4
scripts/serve.sh metal engine-metal loop9b-mlx4
```

which runs, with the profile in [`config/metal/serve.args`](../config/metal/serve.args):

```bash
engine-metal/zig-out/native/bin/tensorfold-native serve loop9b-mlx4 --name loop9b --port 8090 \
  --context 2048 --prompt-cache-gib 0 --no-drafts
```

Load takes about 1.7 s for 4.72 GiB of weights. Startup logs the plan:
`build: plan.linear 42 slots 8 cap 2048 steps 56 mhc true`; each visit owns its KV and recurrent state, 14 attention
and 42 linear slots. Raise `--context` for longer prompts; admission caps it at 262,144 although the checkpoint
advertises 524,288 positions.

`--no-drafts` is the only supported mode: the checkpoint ships no drafter, and the loop's second visit is not a
speculative path. Do not enable MTP or DFlash2 against it.

## Results

Measured on a Mac Studio M5 Ultra (macOS 27.2), 4-bit, drafts off.

### Task gate

Against transformers at the pinned revision, run on the **same 4-bit weights** (`attn_implementation=sdpa`), 64
greedy tokens from three fixed prompts:

| prompt | tokens identical | transformers fp32 vs its own bf16, same weights |
| --- | --- | --- |
| code completion | **64/64** | 39/64 |
| arithmetic reasoning | **64/64** | 48/64 |
| plain prose | 19/64 | 19/64 |

Top-1 agrees on all three. The prose prompt first breaks at token 19, on a rank-2 candidate 1.19 logits behind. The
reference itself diverges from its fp32 twin at the same token, so 19/64 there is the measurement's resolution, not a
defect. Its per-execution error curve is continuous rather than stepped, like the prompts that pass. A 512-token prompt
agrees for 14 of its first 16 tokens. Going above that floor would need the fp32 run as the reference.

### Speed

Cold prefill through the served API, `--prompt-cache-gib 0`, `max_tokens 1`, counting only requests that report
`cached_tokens 0`; one fresh server per arm, chunk 2048 (the M5 default), 2 repetitions that agree to 0.3%. Prompts are
the first N tokens of the engine's `docs/recipes/*.md` in name order.

| prompt tokens | looped | 56 executions, no mHC window | plain 32-layer | stock Qwen3.5-9B | mlx-lm 0.32.2, stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | **2,709** | 2,789 | 4,815 | 4,824 | 4,721 |
| 2,048 | **2,752** | 2,803 | 4,837 | 4,841 | 4,987 |
| 8,192 | **2,717** | 2,769 | 4,850 | 4,856 | 4,846 |
| 32,768 | **2,461** | 2,505 | 4,426 | 4,429 | 4,305 |

The recipe carried in the series (`docs/recipes/sensenova-loop-9b-metal.md`) records another pass of the looped column,
2,710 / 2,754 / 2,718 / 2,460, within 0.1% of these. The controls come from `tf-qwen27-run --max-tokens 0` on the same
binary; the looped column matches it within 0.5%.
Stock is `mlx-community/Qwen3.5-9B-4bit` @ `8b2b98c0`. Prefill sits at 97-98% of the no-window column, and that column is
the plain path times 32/56: each execution costs what a plain layer does, and the GEMMs run at mlx-lm's rate.

Decode and short-prompt latency, `tools/bench_openai.py`, 64 tokens, 2 repetitions, temperature 0:

| metric | looped | plain 32-layer control |
| --- | ---: | ---: |
| decode | **59.8-60.0 tok/s** | 123.7 tok/s |
| TTFT, 14-token prompt | 0.019 s | 0.010 s |

`bench_openai.py`'s `prefill_tps` divides the prompt by TTFT, so with a 14-token prompt it measures one pass's
latency, not prefill throughput.

The mHC router kernel set the decode rate. Each change below is bit-identical (same prompt hashes at 512 to 32k, same
decoded and served tokens):

| router change | decode tok/s | prefill 8k tok/s |
| --- | ---: | ---: |
| one simdgroup, outputs in turn | 11.3 | |
| outputs side by side, one simdgroup an output | 36.5 | 2,596 |
| two streams fixed at compile time, lane loops unrolled by 8 | 62.7 | 2,694 |
| `merge` with two streams fixed at compile time | 63.6 | 2,720 |

`tools/router_fp64.py` checks the router's reduction orders against FP64 on 528 real rows: all at fp32 rounding.
Decode is 60 tok/s against 83 for 56 executions without the window; at one row the router is a single threadgroup.

### Not covered

- Images and video: the vision tower is not converted or loaded, and image parts are refused ([In progress](../README.md#in-progress)).
- Contexts past 262,144 tokens.
- Concurrent streams: every Metal number here is one stream.
- Other Apple chips.
