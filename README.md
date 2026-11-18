Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# SenseNova-Loop-9B on TensorFold: DGX Spark and Mac Studio

A recipe for serving [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) with
[TensorFold](https://github.com/ashhart/TensorFold) on two backends: CUDA on one NVIDIA DGX Spark (TP=1) or a 16 GB RTX
card, and Metal on Apple Silicon. It ships an exact upstream TensorFold pin and a patch series for each backend, a converter from the
published BF16 checkpoint to the engine's MLX affine format, serving profiles, and the measurements behind them.
No weights are included or published: you download the checkpoint from SenseNova and convert it yourself.

## Credits

- **Model: [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) by
  [SenseNova](https://huggingface.co/sensenova)** ([GitHub](https://github.com/OpenSenseNova/SenseNova-Loop)). The
  looped decoder, the two-stream mHC and the reference `modeling_sensenova_loop.py` that every number here is checked
  against are theirs. The model is licensed under the
  [Apache License 2.0](https://github.com/OpenSenseNova/SenseNova-Loop/blob/main/LICENSE); download it under the terms
  on its model card.
- **Engine: [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart (ashhart) and the TensorFold
  contributors**, Apache License 2.0 (code written before 0.6.0 keeps its MIT notice). The `qwen3_5` CUDA family and
  the dense `qwen27` Metal family, their Gated DeltaNet, hybrid-attention, YaRN and MLX-affine code, and the
  Qwen3.5-VL tower are upstream work this recipe builds on. The patches here extend it; they are not upstream changes.

- **Shared-round drafter pricing:** upstream TensorFold PR [#2475](https://github.com/ashhart/TensorFold/pull/2475)
  by [jrb-ai](https://github.com/jrb-ai).
- **EXL3:** [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp, built on
  [QTIP](https://github.com/Cornell-RelaxML/qtip) (Cornell RelaxML).

More in [CREDITS](docs/CREDITS.md); license terms in [LICENSING](docs/LICENSING.md) and [NOTICE](NOTICE).

| | |
| --- | --- |
| Model | [`sensenova/SenseNova-Loop-9B`](https://huggingface.co/sensenova/SenseNova-Loop-9B) @ `329a65f701ba8c48b73c30d0cbaa6f0449733673`, BF16, Apache-2.0 |
| Architecture | Qwen3.5 dense hybrid attention; physical layers `[4, 28)` run twice (56 executions from 32 weight sets), wrapped in a two-stream mHC ([MODEL](docs/MODEL.md)) |
| Weights served | MLX affine g64, converted with [`tools/sensenova_loop_mlx.py`](docs/CONVERT.md) from the patch series: 8-bit on CUDA, 4-bit on Metal |
| CUDA | 1x DGX Spark (GB10), TP=1: upstream TensorFold [`038e4a278`](config/cuda/tensorfold.rev) + [`patches/cuda/`](patches/cuda/) `0001`-`0019` |
| RTX | 16 GB RTX cards (tested on an RTX 5070 Ti, WSL2): the CUDA pin and series + [`patches/rtx/`](patches/rtx/) `0001`-`0002`, EXL3 packs |
| Metal | Apple Silicon (measured on M5 Ultra and M4 Pro): upstream TensorFold [`6d0a128b0`](config/metal/tensorfold.rev) + [`patches/metal/`](patches/metal/) `0001`-`0044` |
| Input | text and images on both ([CUDA](docs/CUDA.md#serve), [METAL](docs/METAL.md#images)) |
| Drafts | copy drafts on by default on both: drafts from a stream's own context, verified, so tokens are unchanged |

The two series sit on different upstream commits, so each backend keeps its own pin. Each pin is the upstream commit
its backend was measured on, and each series rebuilds the measured engine tree exactly
(`config/<backend>/tensorfold.tree` is checked after every apply). The CUDA series also rebases cleanly onto the Metal
pin, but that tree has not been measured. See [ENGINE](docs/ENGINE.md).

> **Experimental.** Both backends run the full 56-execution model end to end and keep the reference's top-1 token. CUDA
> misses the planned 0.05-logit parity gate and has no task gate yet; Metal passes its task gate on code and reasoning,
> and its prose prompt stops at the reference's own noise floor. Read [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) before you
> rely on it.

## What works today

**CUDA, DGX Spark: 8-bit, 1M context, images, four chats.** The served build is `patches/cuda/` `0001`-`0019`
([CUDA](docs/CUDA.md#serve)): the converter's 8-bit text weights, the BF16 vision tower, YaRN factor 4, fp8 KV, a 24
GiB prompt cache and an NVMe disk tier.

| | |
| --- | --- |
| decode | 18.0 tok/s one chat; 13.2-14.6 tok/s each with four chats |
| prefix reuse | a 74K prompt's second turn: TTFT 0.66 s against 52.4 s cold |
| tools | strict tool gate pass (`required` and `auto`) |
| soak | 45 requests over 12 rotating conversations, 0 errors |
| drafts on vs off | identical token ids at 2 and 4 chats |

**Metal: 4-bit with copy drafts.** M5 Ultra 59.5-60.0 tok/s without drafts; with copy drafts a repeated paragraph
runs at 362 tok/s, a code edit at 87 and prose at 94, all 16 reply pairs token-identical to drafts off. An M4 Pro
MacBook Pro (24 GB) decodes at 26.7 tok/s ([METAL](docs/METAL.md#copy-drafts)).

**RTX 16 GB: EXL3 packs.** Tested on an RTX 5070 Ti 16 GB in WSL2 with the CUDA series plus `patches/rtx/`
([RTX](docs/RTX.md)): the 6.0-bit pack decodes at 46.6 tok/s at 4k and 43.9 at 32k with drafts off, prefills at 2,232
tok/s, passes 36/36 tool calls and a 30-minute soak, and its gate logits equal the DGX Spark's run of the same pack.
The 4.0-bit pack decodes at 59.1 tok/s. No packs are published; you quantize them yourself.

**Tool calls need `libtfgrammar`.** `tool_choice: "required"`, a named function and `response_format` need
`libtfgrammar` (`.so` on Linux, `.dylib` on macOS) beside the server, built from xgrammar v0.2.8 with
`zig build grammar`; without it those requests get HTTP 400 ([CUDA](docs/CUDA.md#required-and-named-tool-calls),
[METAL](docs/METAL.md#serve)).

## Results

**Parity against HF transformers** (BF16 reference, `attn_implementation="sdpa"`, `trust_remote_code=True`):

| | CUDA, DGX Spark | Metal, Mac Studio M5 Ultra |
| --- | --- | --- |
| Engine weights | BF16 checkpoint as published | 4-bit, against transformers on the same 4-bit weights |
| Last-row logits, 56 executions | `max\|Δlogit\|` 2.203, top-1 match, top-5 set 5/5 | top-1 match on all three prompts |
| Greedy agreement | 30/31 teacher-forced | 64/64 code, 64/64 arithmetic, 19/64 prose |
| Planned gate | `max\|Δlogit\|` ≤ 0.05: **not met** (mHC drift) | 64/64 on two of three prompts; prose stops at the reference's own bf16/fp32 noise floor (19/64) |

**Images on Metal.** The vision tower sits inside the model's own bf16 noise: engine against bf16 max 3.63, cosine
0.99869, where transformers bf16 against fp32 is 3.86 / 0.99784 and the engine against fp32 is 1.64 / 0.99952. All 27
blocks track the reference within 0.2% rms, M-RoPE positions match `get_rope_index` on 513 of 513 cells, and it reads
all seven labels of a benchmark chart. Asked to name models on a chart with no legend, it invents names
([METAL](docs/METAL.md#images)).

**Small Macs.** A 16 GB Mac runs the same 4-bit tree: 5.07 GB on disk, peak 7.60 / 7.82 / 9.13 GiB at 4k / 8k /
32k, 2,719 tok/s prefill at 8k and 58 tok/s decode, measured on the M5 Ultra with Metal's working set capped to a
16 GB Mac's. 8 GB Macs are not yet supported ([METAL](docs/METAL.md#small-macs)).

**Quantized on CUDA:** q8 cosine 0.980 with 18/31 greedy agreement, q4 19/31. A stock
`mlx-community/Qwen3.5-2B-MLX-4bit` through the same binary scores cosine 0.927 and 4/15 against its own BF16, so this
is ordinary quantization noise. These trees came from an earlier conversion; see [KNOWN-ISSUES](docs/KNOWN-ISSUES.md).

**Task gate.** Metal, 64 greedy tokens a prompt against transformers on the same 4-bit weights: **code 64/64,
reasoning 64/64, prose 19/64**. 19/64 is exactly where transformers' own fp32 run and its bf16 run part on that prompt,
so it is the reference's noise floor. The prefill kernels leave every token hash unchanged. CUDA: not run yet; its
fidelity evidence is the BF16 parity above and the q8 cosine against the stock control.

**Speed** (4-bit, one stream unless noted; the CUDA column is the earlier 4-bit, 32K profile on `0001`-`0009`):

| | CUDA, DGX Spark | Metal, Mac Studio M5 Ultra |
| --- | --- | --- |
| Decode | 63.2 tok/s (`fibonacci-raw`), 4 streams: 174.7 tok/s aggregate | 59.8-60.0 tok/s |
| TTFT, 14-token prompt | | 0.019 s |
| Cold prefill, 512 / 2k / 8k / 32k tokens | 1,908 / 2,081 / **2,048** / 1,890 tok/s | 2,709 / 2,752 / 2,717 / 2,461 tok/s |
| Same weights, loop and mHC removed (8k) | 3,833 tok/s | 4,850 tok/s |

The cost is expected: 56 layer executions where a plain pass has 32, plus the mHC. CUDA prefill at 8k is 2,048 tok/s
against a bound of 32/56 of the plain path (2,190 tok/s), i.e. 93.5% of it. Full tables and conditions are in
[CUDA](docs/CUDA.md#results) and [METAL](docs/METAL.md#results).

## Quick start

Needs `git`, Zig 0.17.0 (the engine's `.zig-version`), and Python 3.11+ with `numpy`, `torch` and `safetensors` for the
converter. The checkpoint is gated on Hugging Face: accept its terms there first.

```bash
git clone https://github.com/jayleaton/sensenova-loop9b-tensorfold.git
cd sensenova-loop9b-tensorfold
hf download sensenova/SenseNova-Loop-9B --revision 329a65f701ba8c48b73c30d0cbaa6f0449733673 \
  --local-dir SenseNova-Loop-9B
```

**CUDA, on a DGX Spark** (CUDA 13 toolkit; [CUDA](docs/CUDA.md)):

```bash
scripts/build-engine.sh cuda engine-cuda                     # pin + patches/cuda, then zig build native -Dnvcc=...
python3 engine-cuda/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx8 --bits 8
python3 scripts/serve-tree.py engine-cuda loop9b-mlx8 SenseNova-Loop-9B loop9b-q8-vis 4   # + vision tower, YaRN 4
scripts/serve.sh cuda engine-cuda loop9b-q8-vis              # 1M context, 4 chats, fp8 KV, --vision, port 8090
```

**Metal, on a Mac** (Xcode with the macOS SDK and Metal toolchain; [METAL](docs/METAL.md)):

```bash
scripts/build-engine.sh metal engine-metal                   # pin + patches/metal, then zig build native -Dcpu=apple_m1
python3 engine-metal/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx4
scripts/serve.sh metal engine-metal loop9b-mlx4              # --context 33024 --parallel 4, copy drafts on
PROFILE=vision scripts/serve.sh metal engine-metal loop9b-mlx4   # image input: --context 4096 ... --vision
```

Then:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"loop9b","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":512}'
```

Thinking is on by default (the checkpoint's chat template); send `"chat_template_kwargs": {"enable_thinking": false}`
to turn it off. A `--thinking-budget` at or above `max_tokens` never closes the thinking block (the reply ends with
`length`); keep it under `max_tokens` or add `--thinking-reserve N`. Build `libtfgrammar` for required and named tool
calls (see above). To check the patch series without building, run `scripts/fetch-engine.sh cuda|metal|rtx DIR`: it fetches
the pin, applies the series and checks the resulting tree.

## Not yet

- **DFlash2 drafter:** waiting on SenseNova's published drafter weights.
- **Benchmarks:** coming.

## Documentation

| | |
| --- | --- |
| [MODEL](docs/MODEL.md) | the checkpoint: loop schedule, mHC, cache sizing, memory |
| [CONVERT](docs/CONVERT.md) | BF16 to MLX affine: commands and the mappings that matter |
| [CUDA](docs/CUDA.md) | DGX Spark: build, the served 1M profile, parity, speed |
| [METAL](docs/METAL.md) | Apple Silicon: build, serve, copy drafts, fidelity, speed |
| [RTX](docs/RTX.md) | EXL3 packs on a 16 GB RTX card: results on an RTX 5070 Ti, pack choice, WSL2 steps |
| [ENGINE](docs/ENGINE.md) | the pins and both patch series, patch by patch |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [CREDITS](docs/CREDITS.md) / [LICENSING](docs/LICENSING.md) | who built what, and under which terms |

## Licensing

This recipe's scripts, patches and docs: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The
TensorFold source the patches modify keeps TensorFold's license and notices, and the model keeps SenseNova's terms:
[`docs/LICENSING.md`](docs/LICENSING.md).
