# The engine: pins and patch series

Each backend builds upstream [TensorFold](https://github.com/ashhart/TensorFold) at an exact commit with this recipe's
patch series applied in order. The pin, the series length and the resulting tree are recorded in `config/<backend>/`:

| file | CUDA | Metal |
| --- | --- | --- |
| `tensorfold.repo` | `https://github.com/ashhart/TensorFold.git` | same |
| `tensorfold.rev` | `038e4a27872eaa018adc67b1b5ddffd16820c498` | `6d0a128b0cde35aa091bda933577cfced3519b8a` |
| `tensorfold.patches` | `0009` | `0039` |
| `tensorfold.tree` | `31714680052c2be236d6162d450c96b1f61f3bb6` | `bd326a4c58374fa9063f4e011e6004696ee53640` |

Both pins are on upstream `main` (2026-11-17); `038e4a278` is 36 commits older than `6d0a128b0`.
`scripts/fetch-engine.sh BACKEND DEST [SOURCE]` fetches the pin, applies `patches/BACKEND/0001`..`LAST` with `git am`
(each patch keeps its author; the committer and dates are fixed, so every machine gets the same commits) and refuses
any result whose tree differs from `tensorfold.tree`. SOURCE may be a local TensorFold checkout or bundle that holds
the pin.

## Why two pins

The CUDA work started from `038e4a278` and the Metal work from `6d0a128b0`. Each pin is the commit its backend was
measured on, so each series gives back the measured tree exactly. The two series touch disjoint files (CUDA:
`zig/src/families/qwen3_5/`, its kernels and contract; Metal: `zig/src/families/qwen27/`, `zig/kernels/metal/qwen27/`
and a few shared core, build and Metal runtime files), and the CUDA series also applies cleanly on `6d0a128b0`. That tree is not measured, so
it is not the pin.

## CUDA series, `patches/cuda/` (21 files, +1185 / -28)

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

`zig/` after `0001`-`0007` is identical to the measured CUDA engine; `0008` adds only `tools/`, and `0009` only
reformats (no blank line at two files' ends, one double blank line, one struct field a line). Both trees as measured
failed `zig fmt --check`; CUDA `0009` and Metal `0039` are that rewrite, and each leaves every source token unchanged.

## Metal series, `patches/metal/` (41 files, +2656 / -103)

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
