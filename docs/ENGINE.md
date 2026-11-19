# The engine: pins and patch series

Each backend builds upstream [TensorFold](https://github.com/ashhart/TensorFold) at an exact commit with this recipe's
patch series applied in order. The pin, the series length and the resulting tree are recorded in `config/<backend>/`:

| config | pin (`tensorfold.rev`) | series | `tensorfold.tree` |
| --- | --- | --- | --- |
| `config/cuda/` (served) | `038e4a27872eaa018adc67b1b5ddffd16820c498` | `patches/cuda/` `0001`-`0022` | `d5938b12bbcc1294e6a24600111cbd4c8d86f159` |
| `config/cuda-v6/` (validated, not yet served) | same | `patches/cuda/` `0001`-`0027` | `9bac6ecbdef1c03861d824e930799edcb3b3de16` |
| `config/rtx/` | same | `patches/cuda/` `0001`-`0022`, then `patches/rtx/` `0001`-`0002` | `0f61f4723db864b510f9a278b6dd5d679b344acb` |
| `config/metal/` | `6d0a128b0cde35aa091bda933577cfced3519b8a` | `patches/metal/` `0001`-`0044` | `d33e2d93ea37f0b3ed1f208599a24ac00b3de16e` |

Every repo is `https://github.com/ashhart/TensorFold.git`. A config with `tensorfold.patches` applies its own directory
`0001`..`LAST`; one with `tensorfold.series` applies each `DIR:LAST` word in order (`cuda-v6`: `cuda:0027`; `rtx`:
`cuda:0022 rtx:0002`).

Both pins are on upstream `main` (2026-11-17); `038e4a278` is 36 commits older than `6d0a128b0`.
`scripts/fetch-engine.sh BACKEND DEST [SOURCE]` fetches the pin, applies `patches/BACKEND/0001`..`LAST` with `git am`
(each patch keeps its author; the committer and dates are fixed, so every machine gets the same commits) and refuses
any result whose tree differs from `tensorfold.tree`. SOURCE may be a local TensorFold checkout or bundle that holds
the pin.

## Why two pins

The CUDA work started from `038e4a278` and the Metal work from `6d0a128b0`. Each pin is the commit its backend was
measured on, so each series gives back the measured tree exactly. The two series touch disjoint files (CUDA:
`zig/src/families/qwen3_5/`, its kernels and contract; Metal: `zig/src/families/qwen27/`, `zig/kernels/metal/qwen27/`
and a few shared core, build and Metal runtime files), and the CUDA series (as first measured) also applied cleanly on `6d0a128b0`. That tree is not measured, so
it is not the pin.

## CUDA series, `patches/cuda/`

| patch | change |
| --- | --- |
| `0001` | `qwen3_5` CUDA: execute a looped checkpoint's plan and its two-stream mHC: the execution plan from `loop_settings`, KV and recurrent state sized per execution (14 attention, 42 Gated DeltaNet), the four mHC kernels (`branch`, `merge`, `enter`, `leave`), HF-checkpoint tolerances |
| `0002` | the mHC kernels in their own image and module, the loop plan in its own module |
| `0003` | resolve the mHC image only when the model has streams; accept an f32 `dt_bias` |
| `0004` | plan accessors refuse a geometry without a valid loop; an f32 `linear_attn.norm` is widened like `dt_bias` |
| `0005` | the mHC kernels registered with the shared kernel-contract gates (this first contract does not compile; `0007` replaces it) |
| `0006` | the mHC router routes 8 rows a block at two streams, reading its weights and each row once: 8k prefill 1,934 to 2,048 tok/s, bit-identical logits |
| `0007` | the mHC contract runs: five cases at the served width, FP64 references, a planted control each; registered in `contract_family_sets.zig` |
| `0008` | `tools/sensenova_loop_mlx.py` and its test, identical to the Metal series' converter after its `0004` |
| `0009` | formatting only: three `qwen3_5` files as `zig fmt` writes them |
| `0010` | `tools/checkpoint_view.py`: a probe tree that links its source for reading and never writes through a link (`scripts/serve-tree.py` uses it) |
| `0011` | an image prompt through the looped decoder on CUDA, and the `image` verb of `tf-qwen35-cuda` |
| `0012` | a later pipeline stage's first norm folds the residual again; the CUDA host suite builds |
| `0013` | a required or named tool call closes the think block in time for the call |
| `0014` | a family's default decode share; `qwen3_5` CUDA takes 0.5 |
| `0015` | `--spill-gib`'s evictions are not charged to the daily write budget |
| `0016` | the CUDA disk tier's staging permit is sized for the longest kept state |
| `0017`-`0019` | shared rounds' copy windows start at three rows and widen as copies land; never where rows are priced or for a drafter's proposals |
| `0020`-`0022` | a shared round's copy rows fill the room left in the backend's 16-row tile (a round past it costs about twice as much), split across its chats by recent acceptance; a single chat and a drafter are unchanged |
| `0023` | the mHC streams collapse at the loop seam on every visit, as the reference does; `TF_Q35_MHC_CARRY=1` keeps the old carry for A/B. Changes served tokens (closer to the reference) |
| `0024` | `TF_QWEN_PREFILL_FP8=1`: kv8 whole-cache prompt attention on e4m3 tensor cores. Opt-in, off by default: it flips 2-3% of top-1 tokens |
| `0025`, `0026` | `TF_QWEN_PREFILL_TUNE=1`: the same prompt attention with one barrier an item (same bits); the integer widening `0025` also tried measured slower and `0026` drops it |
| `0027` | the one-barrier prompt attention is the default (same bits); `TF_QWEN_PREFILL_TUNE=0` restores the second barrier |

`0001`-`0022` is the served engine (`config/cuda/`). `0023`-`0027` are the validated next build (`config/cuda-v6/`):
the seam fix changes tokens, the rest leaves them bit-identical to `0023` alone, and nothing past `0022` is served yet.

The tree after `0001`-`0022` is the served CUDA engine with the converter added: its `zig/` source is that build's,
byte for byte, and the converter (`0008`) is the only difference. `zig/` after `0001`-`0007` is identical to the engine
behind the 4-bit results; `0008` adds only `tools/`, and `0009` only
reformats (no blank line at two files' ends, one double blank line, one struct field a line). Both trees as measured
failed `zig fmt --check`; CUDA `0009` and Metal `0039` are that rewrite, and each leaves every source token unchanged.

## Metal series, `patches/metal/` (44 files, +2725 / -154)

| patch | change |
| --- | --- |
| `0001` | `qwen27`: YaRN RoPE, SenseNova-Loop registration and a loop schedule shared in `zig/src/core/loop_schedule.zig` |
| `0002`, `0004` | the converter, and the zero-centred norm fold with its test |
| `0003` | per-layer residual dumps from the real-weight gate |
| `0005` | run a looped checkpoint's 56 executions and its two-stream mHC on Metal |
| `0006`, `0008`, `0015` | keep the new sources inside TensorFold's lean rules (600 lines a file, one-line comments) |
| `0007`, `0012`, `0014` | the mHC window's residual in its own stream planes; fold the pending residual and the window input in place |
| `0009`, `0011`, `0016`, `0017` | taps and a narrow tape at the window's own width, to measure the window seam |
| `0010` | prove the mHC merge in place, repeated and per row |
| `0013` | share the router's coefficients through threadgroup memory |
| `0018`, `0019`, `0021`, `0025` | `docs/recipes/sensenova-loop-9b-metal.md`: the recipe, its noise floor, the router's cost and cold prefill |
| `0020`, `0023`, `0024` | the router's outputs side by side, the router and the merge with two streams fixed at compile time; each bit-identical |
| `0022` | `tools/router_fp64.py`: the router's FP64 reference and the reduction orders it admits |
| `0026` | the converter carries the vision tower: `model.visual.*` as `vision_tower.*`, 333 tensors in bf16 |
| `0027` | `tools/vision_golden.py`: the transformers side of the vision checks |
| `0028`, `0033` | the recipe's `## Images` section: serving, tower parity, what it reads, speed with an image |
| `0029`, `0030` | dump the tower's merged rows and each block's residual (`TF_QWEN_VISION_DUMP`, `TF_QWEN_VISION_BLOCKS`) |
| `0031` | initialise the vision dump at load (a struct default never applies to the tower; an image request segfaulted) |
| `0032` | dump the prompt's M-RoPE planes (`TF_QWEN_ROPE_DUMP`) for a `get_rope_index` check |
| `0034` | a startup margin that scales with the working set (an eighth, 1 to 8 GiB), every execution charged, 512 to 128 prompt rows before refusing |
| `0035` | converter: per-module recipes, an opt-in least-error range search, MLX's bitstream for 3/5/6-bit codes; MTP skipped |
| `0036`, `0038` | small-Mac candidates and recipes (`docs/small-quant-*`), and the recipe's "Small Macs" section |
| `0037` | converter: a stock Qwen3.5 f32 gated norm stored as bf16 (SenseNova's is bf16 already; this is for the control) |
| `0039` | formatting and comments only, the same tokens: four `qwen27` files as `zig fmt` writes them |
| `0040` | the mHC state pointers in Metal's `device` address space, so macOS 26 (MSL 4.0) and 15 (MSL 3.2) compile the library |
| `0041` | every execution's state kept after a shared round, not only the physical layers (2 or more streams) |
| `0042` | lone streams propose copy drafts without a drafter |
| `0043` | copy drafts on by default, `--no-drafts` turns them off; EXL3 checkpoints would keep them opt-in. One conflict against this series, in `qwen27_served.zig`: the incoming `copy_policy` line is taken, and since this tree has no EXL3 loader `model.exl3` is set false |
| `0044` | the engine's Metal recipe: serve with copy drafts and `libtfgrammar`, and keep answer room with `--thinking-reserve` |

`0001`-`0025` are the archive's `sensenova/loop9b-metal-prefill` (`79222dc1a`); `0026`-`0033` are
`sensenova/loop9b-metal-vision` (`12b5090b8`), which forked before the last three prefill commits, replayed on top;
`0034`-`0038` are `sensenova/loop9b-metal-small` (`4ed6c731d`), forked at `79222dc1a`, replayed after them. The
converter keeps both the tower and the recipes, and the candidates doc says the tower is carried.
Their engine source equals the vision branch's plus the prefill branch's router and merge kernels. Two changes from the
archive: the 28 dump files the vision branch committed (`vision-*.bin`, 36 MB of activations) are left out, and the
recipe keeps the prefill branch's M5 Ultra wording where the vision branch's text still said M3 Ultra.

Each patch is one engine commit with its own message. TensorFold asks for one change per pull request, a kernel-contract
`SUMMARY` for each new GPU kernel, zero new lean-checker findings and an FP64 reference for any changed reduction
order; the patch messages and [CUDA](CUDA.md#engine-checks) / [METAL](METAL.md) record which of those receipts
exist. The series are this recipe's, not upstream pull requests.

## RTX series, `patches/rtx/` (on top of the CUDA series through `0022`)

| patch | change |
| --- | --- |
| `0001` | the EXL3 loader reads SenseNova-Loop's loop and mHC (the loop's KV and recurrent state, the f32 mHC routers, `loop_gate`); the EXL3 capability list compiles and keeps Qwen's decisions; the quantize tools and one resumable chain at 4.0/5.0/6.0 bpw (Hessians from the real looped model, exllamav3 conversion, the extras attached) |
| `0002` | `zig/src/native/cuda.zig` back to TensorFold's 600-line rule: a family's own model_type and its aliases in one registry pass |

`0001` squashes the archive's `sensenova/loop9b-exl3` engine commits (`bb2c8e4f8`, `863dbb692`, `42e0a733d`); one test
conflict against the CUDA series (`cuda_loop_plan_test.zig`) keeps both tests. On `cuda:0022` the RTX tree's engine
source equals the build the RTX copy-draft and host-cache numbers were measured on; the only differences are the
converter in `tools/` and the equivalent registry loop of `0002`. RTX does not take `0023`-`0027` yet.
