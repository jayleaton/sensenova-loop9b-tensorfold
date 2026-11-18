Follow me on X for more updates: https://x.com/jayleaton

Support me here: https://buymeacoffee.com/jayleaton

# SenseNova-Loop-9B on TensorFold: DGX Spark and Mac Studio

A recipe for serving [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) with
[TensorFold](https://github.com/ashhart/TensorFold) on two backends: CUDA on one NVIDIA DGX Spark (TP=1), and Metal on
a Mac Studio M5 Ultra. It ships an exact upstream TensorFold pin and a patch series for each backend, a converter from
the published BF16 checkpoint to the engine's MLX affine format, serving profiles, and the measurements behind them.
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

More in [CREDITS](docs/CREDITS.md); license terms in [LICENSING](docs/LICENSING.md) and [NOTICE](NOTICE).

| | |
| --- | --- |
| Model | [`sensenova/SenseNova-Loop-9B`](https://huggingface.co/sensenova/SenseNova-Loop-9B) @ `329a65f701ba8c48b73c30d0cbaa6f0449733673`, BF16, Apache-2.0 |
| Architecture | Qwen3.5 dense hybrid attention; physical layers `[4, 28)` run twice (56 executions from 32 weight sets), wrapped in a two-stream mHC ([MODEL](docs/MODEL.md)) |
| Weights served | MLX affine g64, 4-bit (or 8-bit), converted with [`tools/sensenova_loop_mlx.py`](docs/CONVERT.md) from the patch series |
| CUDA | 1x DGX Spark (GB10), TP=1: upstream TensorFold [`038e4a278`](config/cuda/tensorfold.rev) + [`patches/cuda/`](patches/cuda/) `0001`-`0009` |
| Metal | Mac Studio M5 Ultra: upstream TensorFold [`6d0a128b0`](config/metal/tensorfold.rev) + [`patches/metal/`](patches/metal/) `0001`-`0034` |
| Input | text on both; **images on Metal** ([METAL](docs/METAL.md#images)); CUDA images [in progress](#in-progress) |

The two series sit on different upstream commits, so each backend keeps its own pin. Each pin is the upstream commit
its backend was measured on, and each series rebuilds the measured engine tree exactly
(`config/<backend>/tensorfold.tree` is checked after every apply). The CUDA series also rebases cleanly onto the Metal
pin, but that tree has not been measured. See [ENGINE](docs/ENGINE.md).

> **Experimental.** Both backends run the full 56-execution model end to end and keep the reference's top-1 token. CUDA
> misses the planned 0.05-logit parity gate and has no task gate yet; Metal passes its task gate on code and reasoning,
> and its prose prompt stops at the reference's own noise floor. Read [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) before you
> rely on it.

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

**Quantized on CUDA:** q8 cosine 0.980 with 18/31 greedy agreement, q4 19/31. A stock
`mlx-community/Qwen3.5-2B-MLX-4bit` through the same binary scores cosine 0.927 and 4/15 against its own BF16, so this
is ordinary quantization noise. These trees came from an earlier conversion; see [KNOWN-ISSUES](docs/KNOWN-ISSUES.md).

**Task gate.** Metal, 64 greedy tokens a prompt against transformers on the same 4-bit weights: **code 64/64,
reasoning 64/64, prose 19/64**. 19/64 is exactly where transformers' own fp32 run and its bf16 run part on that prompt,
so it is the reference's noise floor. The prefill kernels leave every token hash unchanged. CUDA: not run yet; its
fidelity evidence is the BF16 parity above and the q8 cosine against the stock control.

**Speed** (4-bit, one stream unless noted):

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
python3 engine-cuda/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx4 --bits 4
scripts/serve.sh cuda engine-cuda loop9b-mlx4                # --parallel 4 --context 32768, port 8090
```

**Metal, on a Mac** (Xcode with the macOS SDK and Metal toolchain; [METAL](docs/METAL.md)):

```bash
scripts/build-engine.sh metal engine-metal                   # pin + patches/metal, then zig build native -Dcpu=apple_m1
python3 engine-metal/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx4
scripts/serve.sh metal engine-metal loop9b-mlx4              # --context 2048 --prompt-cache-gib 0 --no-drafts
PROFILE=vision scripts/serve.sh metal engine-metal loop9b-mlx4   # image input: --context 4096 ... --vision
```

Then:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"loop9b","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":512}'
```

Thinking is on by default (the checkpoint's chat template); send `"chat_template_kwargs": {"enable_thinking": false}`
to turn it off. To check the patch series without building, run `scripts/fetch-engine.sh cuda|metal DIR`: it fetches
the pin, applies the series and checks the resulting tree.

## In progress

- **CUDA image input.** The CUDA engine has the Qwen3.5-VL tower code at this geometry, but GPU qualification for
  this checkpoint is pending, and the CUDA series' converter still leaves the tower out.
- **CUDA task gate.** The same three-prompt greedy gate Metal passes, on CUDA.

## Documentation

| | |
| --- | --- |
| [MODEL](docs/MODEL.md) | the checkpoint: loop schedule, mHC, cache sizing, memory |
| [CONVERT](docs/CONVERT.md) | BF16 to MLX affine: commands and the mappings that matter |
| [CUDA](docs/CUDA.md) | DGX Spark: build, serve, parity, speed |
| [METAL](docs/METAL.md) | Mac Studio M5 Ultra: build, serve, fidelity, speed |
| [ENGINE](docs/ENGINE.md) | the pins and both patch series, patch by patch |
| [KNOWN-ISSUES](docs/KNOWN-ISSUES.md) | what is not solved |
| [CREDITS](docs/CREDITS.md) / [LICENSING](docs/LICENSING.md) | who built what, and under which terms |

## Licensing

This recipe's scripts, patches and docs: **Apache License 2.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The
TensorFold source the patches modify keeps TensorFold's license and notices, and the model keeps SenseNova's terms:
[`docs/LICENSING.md`](docs/LICENSING.md).
