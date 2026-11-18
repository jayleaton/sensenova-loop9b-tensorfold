# Metal: Mac Studio M5 Ultra

The dense `qwen27` Metal family executes the loop plan and the two-stream mHC, with YaRN RoPE added for this
checkpoint. The series also carries the recipe TensorFold keeps for it,
`docs/recipes/sensenova-loop-9b-metal.md`, in the patched checkout.

| | |
| --- | --- |
| Hardware | Mac Studio M5 Ultra, macOS 27.2 |
| Engine | upstream TensorFold `6d0a128b0cde35aa091bda933577cfced3519b8a` (on `main`, 2026-11-17) + [`patches/metal/`](../patches/metal/) `0001`-`0039` |
| Engine tree | `bd326a4c58374fa9063f4e011e6004696ee53640`: the measured text engine (`0001`-`0025`), the image path (`0026`-`0033`), small-Mac support (`0034`-`0038`) and `0039`'s formatting fixes |
| Weights | MLX affine 4-bit g64, vision tower bf16 ([CONVERT](CONVERT.md)) |

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

The converted tree carries the vision tower (5.57 GiB in all); without `--vision` the tower is not loaded. The text
numbers below were measured on a text-only tree of 4.72 GiB. Load takes about 1.7 s for 4.72 GiB of weights. Startup logs the plan:
`build: plan.linear 42 slots 8 cap 2048 steps 56 mhc true`; each visit owns its KV and recurrent state, 14 attention
and 42 linear slots. Raise `--context` for longer prompts; admission caps it at 262,144 although the checkpoint
advertises 524,288 positions.

`--no-drafts` is the only supported mode: the checkpoint ships no drafter, and the loop's second visit is not a
speculative path. Do not enable MTP or DFlash2 against it.

## Images

Image input is supported on Metal. The checkpoint's 27-block Qwen3.5-VL tower (hidden 1152, 16 heads of 72,
intermediate 4304, a conv3d patch embed `[1152, 3, 2, 16, 16]`, merge 2, merger to 4096) is the tower TensorFold
already reads for Qwen3.8-27B, at this model's 4096 output width. The engine's existing `core/qwen_vl` reader takes it;
no tower kernel is new.

```bash
PROFILE=vision scripts/serve.sh metal engine-metal loop9b-mlx4
# engine-metal/zig-out/native/bin/tensorfold-native serve loop9b-mlx4 --name loop9b --port 8090 \
#   --context 4096 --prompt-cache-gib 0 --no-drafts --vision
```

An OpenAI `image_url` part works as-is. PNG needs nothing; JPEG needs libjpeg-turbo 3.1.4.1
(`TENSORFOLD_LIBTURBOJPEG`). `--vision` loads the tower with 1.78 GiB of scratch for 4096 image tokens a request.
Image rows enter at the embedding, and the loop and mHC then run over the whole sequence, image rows included. Every
row of one image shares a single text position and walks its merge grid in M-RoPE's `h` and `w`; text after the image
resumes past the grid's larger side.

### Tower parity

Against transformers on the same checkpoint, a chart resized to 16x36 patches (144 merged rows):

| comparison | max | mean | cosine |
| --- | ---: | ---: | ---: |
| engine vs transformers bf16 | 3.63 | 3.9e-03 | **0.99869** |
| transformers bf16 vs transformers fp32 (the noise floor) | 3.86 | 4.2e-03 | 0.99784 |
| engine vs transformers fp32 | 1.64 | 2.7e-03 | 0.99952 |

The engine is closer to the fp32 reference than the bf16 reference is, so the remainder is accumulation order, not a
port defect. All 27 blocks are checked the same way: engine rms tracks the reference within 0.2% at every block, and
per-block cosine stays at or above 0.9989. Nothing steps; the last block's absolute error rises only because the
tower's own activations rise there (reference rms 1.5 to 186). The M-RoPE positions equal `get_rope_index` on
**513 of 513 cells**: all three axes over all 171 prompt rows, image rows and the text after them.

### What it reads

Asked for the labels on a chart of this model's release benchmarks, it reads all seven: AutomationBench,
Terminal-Bench 2.1, NL2Repo, AIDA-Bench, BIRD, x²-Banking, GAIA. The known limitation: asked to name *models* on a
chart that has no model legend, it answers with names that are not on it, where the reference declines.

### Speed with an image

| metric | value |
| --- | ---: |
| prompt tokens | 164 (144 image rows + 20 text) |
| prefill | 1,397-1,519 tok/s |
| TTFT | 0.11-0.12 s |
| decode | 30.7 tok/s |

An image costs prefill in proportion to its rows. These numbers were taken before the router and merge changes
(`0023`, `0024`) that brought text decode from 33 to 60 tok/s, and they have not been re-measured on this engine.

### Dumps and one fix

`TF_QWEN_VISION_DUMP=1` writes an image's merged rows, `TF_QWEN_VISION_BLOCKS=1` each block's residual and
`TF_QWEN_ROPE_DUMP=1` the prompt's `(t, h, w)` planes, all beside the server and off by default. The tower is created
with `gpa.create` and assigned field by field, so a struct default never applies: a dump field left out segfaulted
an image request until `0031` initialised it at load. `tools/vision_golden.py` is the transformers side of these
checks.

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

## Small Macs

**16 GB Macs: supported, with the same 4-bit tree.** Patch `0034` makes that possible: startup used to keep a fixed
8 GiB margin, which refused every 16 GB Mac whatever the weights. The margin is now an eighth of Metal's working set
(1 GiB at least, 8 GiB from a 64 GiB set), the fit charges KV and recurrent state per execution, and a frame that
cannot fit 512 prompt rows steps down to 128 before it refuses.

**8 GB Macs: not yet supported.** 2-bit packs fit, but they lose code and reasoning badly, as stock Qwen3.5-9B does
at 2 bits. A follow-up is being tried.

These numbers come from the M5 Ultra with Metal's working set capped by `TF_QWEN27_WORKING_SET_BYTES` to what Metal
grants a 16 GB Mac (11453251584) or an 8 GB Mac (5726623744), not from a 16 GB or 8 GB machine. Each run is
`--parallel 1 --prompt-cache-gib 0 --no-drafts`. Memory is the peak process footprint
(`tensorfold:process_footprint_peak_bytes`) after one request of about half the context.

| tier | pack | disk | peak 4k | peak 8k | peak 32k | prefill 8k | decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **16 GB** | **`q4`, the tree above** | **5.07 GB** | **7.60 GiB** | **7.82 GiB** | **9.13 GiB** | **2,719 tok/s** | **58 tok/s** |
| 16 GB | `s16` (6-8 bit attention, embed and head) | 5.70 GB | 8.22 GiB | 8.43 GiB | 9.75 GiB | 1,394 tok/s | 49 tok/s |
| 8 GB | `s8-b` (2-bit MLP) | 3.23 GB | 4.65 GiB | refused | refused | 839 tok/s | 46 tok/s |
| 8 GB | `s8-d` (2-bit, smallest) | 3.02 GB | 4.45 GiB | 4.67 GiB | refused | 833 tok/s | 45 tok/s |

Quality, as teacher-forced top-1 agreement over 64 positions on the three task-gate prompts against each model's own
BF16 continuation (prose / code / reasoning):

| pack | SenseNova-Loop-9B | stock Qwen3.5-9B, same conversion |
| --- | --- | --- |
| `q4` | **57 / 62 / 61** | 64 / 64 / 58 |
| `s16` | 58 / 61 / 62 | |
| `s8-b` | 50 / 44 / 53 | 64 / 44 / 44 |
| `s8-d` | 52 / 38 / 47 | 61 / 45 / 44 |

`s16` adds nothing over `q4` that the gate can resolve, at half the prefill, so `q4` is the 16 GB pack. At 2 bits the
code and reasoning loss matches the stock control's, so it is ordinary 2-bit loss and not the loop's, but it is still
a collapse: greedy code leaves calls unclosed or repeats the function. 3 bits fits neither tier. Only 4-bit group-64
weights take the tuned prompt and decode kernels, so smaller packs also run slower.

The `s16` and `s8` recipes and the converter options behind them (`--recipe`, `--clip-search`, 3/5/6-bit codes) are in
the patched checkout: `docs/small-quant-recipes/`, `docs/small-quant-candidates.md` and the recipe's own "Small Macs"
section. The small-Mac runs used a text-only tree; this repo's converter also writes the 0.85 GiB vision tower,
which is loaded only with `--vision`. Images on a 16 GB Mac are not measured.

### Not covered

- Video: the processor's video path is not wired, and video parts are refused.
- More than one image a request: the path is written for it, untested here.
- Contexts past 262,144 tokens.
- Concurrent streams: every Metal number here is one stream.
- Other Apple chips; 16 GB and 8 GB Macs only through the working-set cap above.
