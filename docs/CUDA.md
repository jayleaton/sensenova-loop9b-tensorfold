# CUDA: one DGX Spark, TP=1

The `qwen3_5` CUDA family executes the loop plan and the two-stream mHC on one GB10. No tensor parallelism is needed.

| | |
| --- | --- |
| Hardware | 1x NVIDIA DGX Spark (GB10, 128 GB unified), CUDA 13.0 |
| Engine | upstream TensorFold `038e4a27872eaa018adc67b1b5ddffd16820c498` (on `main`, 2026-11-17) + [`patches/cuda/`](../patches/cuda/) `0001`-`0022` |
| Engine tree | `d5938b12bbcc1294e6a24600111cbd4c8d86f159`: the serving engine (`0001`-`0022`, without `0008`, is the build behind the numbers below, byte for byte) plus the converter (`0008`); the rows marked `0019` were measured on `0001`-`0019`, whose tokens are the same |
| Weights served | MLX affine 8-bit g64 text weights ([CONVERT](CONVERT.md)) and the BF16 vision tower, YaRN factor 4 (1,048,576 positions) |

## Build

On the Spark, with the CUDA 13 toolkit and Zig 0.17.0:

```bash
scripts/build-engine.sh cuda engine-cuda
```

That runs `scripts/fetch-engine.sh cuda engine-cuda` (fetch the pin, `git am` the series, check the tree), then in
`engine-cuda`:

```bash
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc        # zig-out/native/bin/tensorfold-native
zig build qwen35-cuda -Dnvcc=/usr/local/cuda/bin/nvcc   # tf-qwen35-cuda, the parity tool
```

Set `NVCC=` to use another `nvcc`. `NVCC=none` builds the host code without kernel images; use it only to check that the
series compiles on a machine without CUDA.

### Required and named tool calls

`tool_choice: "required"`, a named function and `response_format` need `libtfgrammar` beside the server; without it
the server answers those requests with HTTP 400 and serves everything else. Build it from xgrammar v0.2.8:

```bash
git clone --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar.git xgrammar-src
(cd engine-cuda && zig build grammar -Dxgrammar="$PWD/../xgrammar-src")   # zig-out/native/lib/libtfgrammar.so
```

Check it: a request with tools and `"tool_choice": "required"` returns `finish_reason: "tool_calls"`, not HTTP 400.

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "loop9b",
  "messages": [{"role": "user", "content": "Look up the population of Paris."}], "tool_choice": "required",
  "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object",
    "properties": {"q": {"type": "string"}}, "required": ["q"]}}}],
  "chat_template_kwargs": {"enable_thinking": false}}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["finish_reason"])'
```

### Images

PNG needs nothing. JPEG needs libjpeg-turbo 3.1.4.1 and WebP needs libwebp 1.6.0, found on `LD_LIBRARY_PATH`.

## Serve

The served tree is the converter's 8-bit text weights plus the checkpoint's BF16 vision tower as one more shard, with
the config's YaRN stretched from factor 2 to factor 4 over the trained 262,144 positions (Qwen3.5's 1M recipe).
`--rope-yarn 4` is refused because the checkpoint's config already names YaRN, so the stretch lives in the tree's
config. `scripts/serve-tree.py` makes it without copying the text weights:

```bash
python3 engine-cuda/tools/sensenova_loop_mlx.py --source SenseNova-Loop-9B --out loop9b-mlx8 --bits 8
python3 scripts/serve-tree.py engine-cuda loop9b-mlx8 SenseNova-Loop-9B loop9b-q8-vis 4
scripts/serve.sh cuda engine-cuda loop9b-q8-vis
```

which runs, with [`config/cuda/serve.args`](../config/cuda/serve.args) and
[`config/cuda/serve.env`](../config/cuda/serve.env):

```bash
TENSORFOLD_MEMORY_RESERVE_GIB=36 TF_QWEN_KV_PAGES=256 \
engine-cuda/zig-out/native/bin/tensorfold-native serve loop9b-q8-vis --name loop9b --port 8090 \
  --parallel 4 --context 1048576 --kv-cache fp8 --max-tokens 32768 --thinking-budget 4096 --decode-share 0.5 \
  --prompt-cache-gib 24 --spill-gib 256 --learn-daily-gib 256 --checkpoint-slots 8 --vision
```

- **KV:** fp8, 28.4 KiB a token. The engine keeps one stream's full 1M window and the other three share the rest of
  the page pool, so every chat can reach 1M and four chats fit at once with at least 256K each. A request whose prompt
  and `max_tokens` do not fit waits in the queue rather than being cut short.
- **Memory:** `TENSORFOLD_MEMORY_RESERVE_GIB=36` keeps the 24 GiB host prompt cache inside unified memory.
- **Disk tier:** idle conversations spill to NVMe (`--spill-gib 256`, `--checkpoint-slots 8`) and are read back ahead
  of admission; `--learn-daily-gib 256` bounds a day's writes.
- **Thinking:** on by default (the checkpoint's chat template). `--thinking-budget 4096` closes a runaway reasoning
  block at 4,096 tokens; a request may ask for its own `thinking_budget`. With `tool_choice` required or named, the
  budget also leaves room for the call. Send `chat_template_kwargs {"enable_thinking": false}` to turn thinking off.

**Thinking budget and answer room.** `--thinking-budget N` closes the thinking block after N reasoning tokens, but
`max_tokens` counts reasoning and answer together: a budget at or above the request's `max_tokens` never closes
before the reply runs out, so the reply ends with `finish_reason: "length"` and no answer. Keep the budget well under
`max_tokens`, or add `--thinking-reserve N`, which keeps the request's `max_tokens` for the answer and lets reasoning
take up to N tokens more (capped by the context window). The startup log says which of the two `max_tokens` means.

- **Copy drafts** are on: a stream drafts from its own context when it repeats text, verified by the target, so the
  tokens are the same with drafts on or off.
- Sampling defaults come from the checkpoint's `generation_config.json` (temperature 1.0, top_p 0.95).

### Serving results

Measured on the build above, the tree above and these flags, on one DGX Spark:

| check | result |
| --- | --- |
| decode, one chat (`0019`) | 18.0 tok/s |
| decode, 4 chats at once (`0019`) | 13.2-14.6 tok/s each |
| copy drafts, 1, 2 and 4 code chats at once, aggregate | 15.3, 33.3 and 54.5 tok/s, against 14.5, 28.0 and 54.0 without drafts |
| decode beside a 35,614-token prompt being read (`0019`) | 11.0 tok/s (that prompt's TTFT 32.5 s) |
| prefix reuse, a 74K prompt's second turn (`0019`) | 73,728 of 74,255 tokens from cache, TTFT 0.66 s against 52.4 s cold |
| drafts on vs off, 2 and 4 chats at once | token ids identical in all 6 replies (768 tokens each, 951-1,330 copied rows) |
| strict tool gate (`required` and `auto`, text and images) | pass |
| an OpenCode request (its system prompt and 12 tools, `tool_choice` auto) that needs a file | one well-formed `read` call |
| soak, 4 workers, 12 rotating conversations (`0019`) | 45 requests, 0 errors, 0 wrong answers; required tool turns 3/3 |
| disk tier during the soak | 39 states saved, 19 read back, 0 refusals, 0 restore fallbacks |
| thinking budget | a runaway-prone prompt stopped thinking at 4,097 tokens |

8-bit is the served width because 4-bit loses too much on this model through the CUDA affine path: against the
engine's own BF16, q8 keeps top-1 on 246 of 249 positions (cosine 0.9963 at worst), q4 on 220.

The results below come from the earlier 4-bit, 32K profile on `0001`-`0009`, and have not been re-measured on the
serving series.

## Results

Measured on one DGX Spark. References are HF `transformers 5.18.0` with torch 2.15.0 on the same GPU, bf16,
`attn_implementation="sdpa"`, `trust_remote_code=True`.

### Parity (BF16)

```bash
engine-cuda/zig-out/native/bin/tf-qwen35-cuda MODEL step IDS OUT       # last-row logits per step
engine-cuda/zig-out/native/bin/tf-qwen35-cuda MODEL rows IDS_FILE OUT   # final-norm rows
```

`max|Δlogit|` on the last prompt row of a 23-token prompt, against the reference truncated to the same depth:

| depth | max abs | top-1 | top-5 |
| --- | ---: | --- | --- |
| 4 physical layers | 0.16 | match | |
| 8 | 0.16 | match | |
| 16 | 0.23 | match | |
| 24 | 0.31 | match | same set |
| 32, loop 1, no mHC | 0.375 | match | same order |
| **32, loop 2, mHC (56 executions)** | **2.203** | **match** | **same set (5/5)** |

Teacher-forced greedy agreement over the reference's 32-token continuation: **30/31** at BF16. The plain path is exact
to bf16 rounding (the depth series grows slowly to 0.375); the loop and mHC carry 2.2 logits of drift, above the
planned 0.05 gate ([KNOWN-ISSUES](KNOWN-ISSUES.md)). It was re-verified after the series' refactors (`0002`-`0004`)
with the same result.

### Quantized fidelity

| comparison, same binary | max abs | cosine | greedy agreement |
| --- | ---: | ---: | --- |
| q8 values in BF16 storage vs the BF16 reference | 0.161 | 1.000 | |
| q8 g64 through the affine kernels | 4.414 | 0.980 | 18/31 |
| q4 g64 through the affine kernels | 4.328 | 0.980 | 19/31 |
| control: stock `mlx-community/Qwen3.5-2B-MLX-4bit` vs its publisher's BF16 | 4.28 | 0.927 | 4/15 |

The converted values are faithful (cosine 1.000 in BF16 storage). The 0.02 cosine the affine path gives up is what the
format costs, and the stock conversion of the same family gives up more. Quantization is judged by cosine and
side-by-side quality, not by exact token agreement. These trees came from an earlier conversion of the same format;
see [KNOWN-ISSUES](KNOWN-ISSUES.md#cuda).

### Speed

Served through TensorFold's `tools/bench_openai.py` and `tools/bench_concurrent.py`, 4-bit g64, `--parallel 4
--context 32768`, temperature 0, 2 repetitions:

| case | decode | detail |
| --- | --- | --- |
| `fibonacci-raw` | 63.3 tok/s (63.2 as shipped, below) | `chunk_tpot` 15.8 ms |
| `gpu-chat-no-think` | 19.0 tok/s | `itl_p50` 52.2 ms |
| concurrency 1, code | 54.2 tok/s | equal to solo |
| concurrency 4, code | 46.5 tok/s a stream, **174.7 aggregate** (193.2 steady) | 4/4 equal to solo, 0 failed |
| concurrency 4, chat | 17.3 tok/s a stream, 66.5 aggregate | 4/4 equal to solo |
| concurrency 8, code | 168.4 aggregate (saturated) | equal to solo, 0 failed |

Re-measured on the engine as shipped (after the series' later changes): `fibonacci-raw` **63.2 tok/s**,
`gpu-chat-no-think` 19.0 tok/s.

Cold prefill, `--parallel 4 --context 65536 --prompt-cache-gib 0`, exact token arrays (seeded pseudo-random ids),
`max_tokens 1`, `cached_tokens 0`, 2 repetitions:

| prompt tokens | 512 | 2k | 8k | 32k |
| --- | ---: | ---: | ---: | ---: |
| **SenseNova-Loop-9B q4, as shipped** | **1,908** | **2,081** | **2,048** | **1,890** |
| same, before the batched router (`0001`-`0005`) | 1,792 | 1,949 | 1,934 | 1,790 |
| same checkpoint, loop and mHC removed (32 plain layers) | 3,525 | 3,870 | 3,833 | 3,530 |
| stock `mlx-community/Qwen3.5-9B-MLX-4bit` @ `938d891`, same binary | 3,500 | 3,880 | 3,840 | 3,545 |

The plain path equals the stock model, so the port adds nothing to it. 56 executions bound the looped model at 32/56
of the plain one (2,190 tok/s at 8k), and this engine reaches **93.5%** of that bound.

Patch `0006` closed most of the gap. Before it the router ran one block a row: each row re-read the 256 KiB fp32 router
matrix and its own streams nine times, and it took 10.6% of an 8k prefill's kernel time. A block now routes 8 rows at
two streams, reading each weight column once for all its rows. Router time per 8k prefill went from 537 to 174 ms,
the prompt GEMM is 69.4% of kernel time, and all mHC kernels together are about 7%. The change is exact: BF16 parity is
still `max|Δlogit|` 2.203125 with the same top-1 and top-5 set, and the `step` and `prompt` logits are bit-identical to
the kernel before it.

Single-token `step` (one process a run, no batching): 7.5 tok/s at 56 executions, 14.3 at 32 layers, 59.2 at 4 layers.
56/32 = 1.75 is exactly the measured ratio, so each execution costs the same at any depth. These are not serving
numbers.

### Engine checks

- `zig fmt --check build.zig zig`: clean with `0009`. The measured tree (`0001`-`0007`) had three files zig fmt rewrites;
  `0009` is that rewrite and changes no code.
- `python3 tools/zig/lean_check.py`: 221 findings at the pin and 221 with the series (0 new); `tools/pr_check.py`: 0.
- mHC kernel contract (`0007`, `zig/tests/cuda/contract_q35_mhc.zig`): five cases (router branch, router pre/post/mix,
  merge in place, enter, leave) at hidden 4096, rows 1 to 4096, FP64 references and one planted control each:
  `SUMMARY 5 kernels: 5 pass gates 1-4, 0 fail; 5 controls: 5 red as planted`.
- `zig build test test-golden`: 2938/2963 passed, 24 skipped, 1 failure. The failure is `cli_config_test` refusing
  `response_format` because `libtfgrammar.so` was absent from that checkout, which is environmental.
- `zig build test-qwen35-launch-args-host`: pass.
