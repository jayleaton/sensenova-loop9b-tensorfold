# HIP: ROG Ally X (Radeon 780M, gfx1103)

> **Tested on one ROG Ally X** (Ryzen Z1 Extreme, Radeon 780M, 24 GB LPDDR5X with the BIOS's 8 GB UMA carve-out),
> on Omarchy (Arch Linux) with Arch's ROCm 7.2.4 packages. Other AMD iGPUs, other distributions and other carve-out
> sizes are untested. Read [the limits](#what-is-not-tested) before you rely on it.

| | |
| --- | --- |
| Engine | upstream TensorFold `813ff4ccbf9d6610e991a96d53fda2b628f56347` (on `main`, 2026-11-19, version 1.2.10) + [`patches/hip/`](../patches/hip/) `0001`-`0018` |
| Engine tree | `083d74e21f2678630829c59af6ee008a963331fe` ([`config/hip/`](../config/hip/)), the tree the measured binary was built from |
| Device | ROG Ally X (RC72LA, BIOS RC72LA.313): Ryzen Z1 Extreme, Radeon 780M (RDNA3, `gfx1103`, 12 CUs), 8 GB UMA carve-out |
| OS | Omarchy 4.5.0 (Arch), kernel `7.2.15-1-omarchy`, Mesa 26.2.5, `linux-firmware-amdgpu` 20260916 |
| ROCm | 7.2.4 from Arch `extra`: `rocm-hip-sdk 7.2.4-1`, `hip-runtime-amd 7.2.4-1`, `rocm-device-libs 2:7.2.4-2`, `comgr 2:7.2.4-2`, `rocm-llvm 2:7.2.4-2` (HIP 7.2.53211, AMD clang 22) |
| Weights | [`jayleaton/SenseNova-Loop-9B-MLX-4bit`](https://huggingface.co/jayleaton/SenseNova-Loop-9B-MLX-4bit), the Metal 4-bit tree, revision `0afa03c99f858625f4f8c5e3158d6ef6fdfad5bc`; text only on HIP |

The measured binary was `zig-out/native-hip/bin/tensorfold-native`, `tensorfold-native 1.2.10`, sha256
`5e2a1f7d7a1a081525e26a696b404d9f7bbb376c90ea70a353ecb11ab5faec37`; your build of the same tree may hash differently.

## Results on the 780M

One stream, temperature 0, thinking off, prompt cache off, an 8,192-token window. Measured 2026-11-19 on the 8 GB
carve-out.

| prompt | decode | first token |
| ---: | ---: | ---: |
| 14 tokens (raw completion, 256 tokens, 3 runs) | **10.8 tok/s** (10.80 median) | 0.31 s |
| 31 tokens (chat, 256 tokens, 3 runs) | 10.8 tok/s | 0.58 s |
| 4,812 tokens | 10.2 tok/s | 23.2 s |
| 6,479 tokens | 10.0 tok/s | 31.7 s |
| 7,428 tokens | 9.8 tok/s | 37.2 s |

Prompts read at about 200 tok/s (prompt length over first-token time: 207, 204 and 200 tok/s at the three long
prompts). The server's first reply ("What is 17 * 23?", four tokens) answered `391`; its 8.1 tok/s is a four-token
reply and not a decode rate.

| check | result |
| --- | --- |
| `zig/tests/qwen35/serve_check.py`, 64 tokens | 11 ok, 3 fail. Every exactness check passes, greedy and at t 0.8: solo == together, serial together == serial solo, the same twice, decoders unchanged by a long prompt, a long prompt among decoders == alone. The three failures test nothing at this profile: `drafted == serial` twice (0 drafts accepted, no drafter) and `resumed == fresh` (prompt cache off, 0 tokens resumed) |
| mHC kernels on gfx1103, `hip-kernel-contract` | `tf_mhc_branch` (both variants), `tf_mhc_merge`, `tf_mhc_enter`, `tf_mhc_leave`: nonfinite, repeat, shapes and sums pass against the FP64 reference, and each planted control is red as planted |
| engine on the 780M, `tf-qwen35-test check` on stock Qwen3.5-4B (`lmstudio-community/Qwen3.5-4B-MLX-4bit`) | 46 passed, 0 failed, 6 skipped in 222.7 s; graphs replayed 2,859 of 3,055 rounds |
| the looped tree on the device, `tf-qwen35-test upload` | 32 layers, 1,023 buffers, 4,818 MiB read back equal |

The startup banner, at the flags below:

```
HIP memory budget from the card's memory (firmware VRAM carve-out): 7.7 of 8.0 GiB free, 0.5 GiB reserve
HIP gfx1103 device 0, rank 0 of 1: model 4.71 GiB; 1 stream at once, 0.53 GiB each at a 8192-token window, of 2.5 GiB left after a 0.5 GiB reserve
serving SenseNova-Loop-9B-MLX-4bit at http://127.0.0.1:8090/v1 (... drafts: copies only; context: 8192; loaded in 19.5s)
```

No teacher-forced fidelity gate against BF16 ran on HIP. The weights are the Metal 4-bit tree, whose quality is
measured on the Mac ([METAL](METAL.md)); the HIP engine's mHC kernels match the FP64 reference above.

## Setup from scratch

Each step is what was run on the Ally X. Run them as your own user unless a step says `sudo`.

### 1. ROCm

```bash
sudo pacman -Sy --needed rocm-hip-sdk rocminfo
rocminfo | grep -m1 gfx1103            # Name: gfx1103 (Marketing Name: AMD Ryzen Z1 Extreme)
```

That one command pulls the whole ROCm 7.2.4 stack above (`rocm-llvm` alone is 7.0 GiB installed). `/dev/kfd` and
`/dev/dri/renderD128` were already open to the desktop user (mode 0666); nothing else was configured.

**gfx1103 is targeted natively.** The build passes `-Dhip-arch=gfx1103` (clang `--offload-arch=gfx1103`), and
TensorFold already lists `gfx1103` with the RDNA3 chips. Do **not** set `HSA_OVERRIDE_GFX_VERSION`; it was never set
on the tested machine. The engine refuses to start if the GPU it finds differs from the build's `-Dhip-arch`.

### 2. GPU memory: the TTM limit

The 780M has no memory of its own. The BIOS's UMA setting reserves a carve-out the driver reports as VRAM (8 GB here,
left at that), and the rest of system memory is reachable as GTT, which the kernel's TTM limit caps at about half of
RAM (7.65 GiB here). The recipe raised the TTM limit to 16 GiB:

```
ttm.pages_limit=4194304 ttm.page_pool_size=4194304        # 4194304 pages of 4 KiB = 16 GiB
```

**On a Limine + UKI boot (Omarchy), `/etc/kernel/cmdline` is ignored.** `limine-entry-tool` builds the kernel command
line from `KERNEL_CMDLINE[...]` in `/etc/default/limine` and `/etc/limine-entry-tool.d/*.conf`, and reads
`/etc/kernel/cmdline` only when `KERNEL_CMDLINE` is unset, which Omarchy's `/etc/default/limine` never leaves it.
The first attempt appended the values to `/etc/kernel/cmdline` and ran `mkinitcpio -P`; `mkinitcpio` warned that it
does not update Limine's entries, and the boot entry never carried them. What works is a drop-in and
`limine-mkinitcpio`:

```bash
echo 'KERNEL_CMDLINE[default]+=" ttm.pages_limit=4194304 ttm.page_pool_size=4194304"' |
  sudo tee /etc/limine-entry-tool.d/ttm.conf
sudo limine-mkinitcpio
sudo /usr/lib/limine/limine-entry-tool --get-cmdline | grep -o 'ttm[^ ]*'   # both values, before you reboot
```

After a reboot:

```bash
grep -o 'ttm[^ ]*' /proc/cmdline
cat /sys/module/ttm/parameters/pages_limit /sys/module/ttm/parameters/page_pool_size    # 4194304 4194304
cat /sys/class/drm/card*/device/mem_info_gtt_total
```

| setting | state |
| --- | --- |
| `ttm.pages_limit=4194304`, `ttm.page_pool_size=4194304` on the boot command line | verified after a reboot |
| `/sys/module/ttm/parameters/pages_limit` and `page_pool_size` | 4194304 each, verified after a reboot |
| GTT total | 15,674 MiB after a reboot, not 16 GiB: `amdgpu` caps GTT at the memory Linux sees (15.3 GiB beside the 8 GB carve-out) |
| UMA carve-out | 8 GB, unchanged; no `amdgpu.*` parameters |

**The numbers above did not use GTT.** They were measured before that reboot, with the limit raised at runtime
(`echo 4194304 | sudo tee /sys/module/ttm/parameters/pages_limit page_pool_size`), and the engine budgeted from the
8 GB carve-out (the banner's `firmware VRAM carve-out` line). Serving from GTT, and any window past what the carve-out
holds, has not run.

### 3. Zig and the engine

Zig 0.17.0 (TensorFold's `.zig-version`), from ziglang.org:

```bash
mkdir -p ~/.local/bin && cd ~/.local
curl -LO https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz
tar -xJf zig-x86_64-linux-0.17.0.tar.xz && ln -sf ~/.local/zig-x86_64-linux-0.17.0/zig ~/.local/bin/zig
```

Then, in this recipe's checkout:

```bash
scripts/build-engine.sh hip engine-hip
```

which fetches the pin, applies `patches/hip/` `0001`-`0018`, checks the tree, and runs:

```bash
zig build -Dhip=true native-hip -Dhip-arch=gfx1103 -Dhipcc=/opt/rocm/bin/hipcc \
  -Dhip-device-lib=/opt/rocm/amdgcn/bitcode
```

The binary is `engine-hip/zig-out/native-hip/bin/tensorfold-native`. The measured binary was built in place with
`nice -n 19 zig build -j4 ...` and the same flags. `scripts/build-engine.sh hip` itself was then run on the Ally X from
a fresh clone (2026-11-20, under `taskset -c 0-5 nice -n 19` to keep the desktop usable): it reproduced the tree above
and built in 378 s.

**The device-library path.** Without `-Dhip-device-lib`, TensorFold passes hipcc
`--rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode`, which Arch does not have, and the first build failed with
`clang++: error: cannot find ROCm device library`. Arch's `rocm-device-libs` installs the bitcode in
`/opt/rocm/amdgcn/bitcode` (`pacman -Qo /opt/rocm/amdgcn/bitcode/ockl.bc`). The flag is the whole fix: no symlink,
environment variable or file change. `scripts/build-engine.sh` passes it by default; set `HIP_DEVICE_LIB` (and
`HIPCC`, `HIP_ARCH`) to override.

### 4. The model

```bash
uv tool install "huggingface_hub[cli]"
hf download jayleaton/SenseNova-Loop-9B-MLX-4bit --local-dir loop9b-mlx4
```

The tested download resolved to revision `0afa03c99f858625f4f8c5e3158d6ef6fdfad5bc` (5,982,954,634 bytes in four
shards; the engine loads 4.71 GiB of text weights). Add `--revision 0afa03c99f858625f4f8c5e3158d6ef6fdfad5bc` to pin
it. This is the same tree the Macs serve; the vision tower it carries is not loaded on HIP. The gated BF16 checkpoint
is not needed.

### 5. Serve

```bash
scripts/serve.sh hip engine-hip loop9b-mlx4
```

which runs, with the profile in [`config/hip/serve.args`](../config/hip/serve.args) and
[`serve.env`](../config/hip/serve.env):

```bash
TENSORFOLD_MEMORY_RESERVE_GIB=0.5 engine-hip/zig-out/native-hip/bin/tensorfold-native serve loop9b-mlx4 \
  --name loop9b --port 8090 --context 8192 --parallel 1 --prompt-cache-gib 0
```

The measured serve was that command without `--name` (the model was served under its directory name), under
`nice -n 19`. Every flag is there for the 8 GB carve-out:

| flag | why |
| --- | --- |
| `TENSORFOLD_MEMORY_RESERVE_GIB=0.5` | the default 1 GiB reserve leaves too little beside the 4.71 GiB model; at 0.5, 2.5 GiB is left for the window |
| `--context 8192 --parallel 1` | one 8K window takes 0.53 GiB |
| `--prompt-cache-gib 0` | the prompt cache on (`--prompt-cache-gib 1.2`, the most that fit) ended in a GPU hang; see [KNOWN-ISSUES](KNOWN-ISSUES.md#hip) |

Copy drafts are on by default (`drafts: copies only`). Thinking is on by the chat template's default; send
`"chat_template_kwargs": {"enable_thinking": false}` to turn it off. `libtfgrammar` was not built on the Ally X, so
`tool_choice: "required"` and named tool calls return HTTP 400; tool calls are untested on HIP.

### 6. Check it

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "loop9b",
  "messages": [{"role": "user", "content": "What is 17 * 23?"}], "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}}'
python3 engine-hip/zig/tests/qwen35/serve_check.py http://127.0.0.1:8090 loop9b 64
```

The first answers `391`. `serve_check` should print 11 `ok` lines and the three expected failures above
(`drafted == serial` twice, `resumed == fresh`); any other `FAIL` is a real difference.

## `patches/hip/`

On upstream `813ff4c`. `0001`-`0011` are the CUDA series' backend-neutral patches, applied as recorded; `0012`-`0018`
port the looped model to TensorFold's `qwen3_5` HIP path.

| patch | change | from |
| --- | --- | --- |
| `0001`-`0003` | the converter, `checkpoint_view.py`, a required or named tool call closes the think block | CUDA `0008`, `0010`, `0013` |
| `0004` | `--spill-gib`'s evictions are not charged to the daily write budget | CUDA `0015` |
| `0005`-`0010` | shared rounds' copy windows and rows | CUDA `0017`-`0022` |
| `0011` | a family's default decode share; the CUDA-family hunks do not exist on this pin, so the HIP registration declares `0.5` (`0017`) | CUDA `0014`, edited |
| `0012` | `qwen3_5`: a looped checkpoint's plan and its two-stream mHC, read from `loop_settings` and `mhc`; `sensenova_loop` admitted (56 executions over 32 layers) | new |
| `0013` | the tower's static YaRN rope, resolved as its config declares it | new |
| `0014` | a windowed layer's mHC routers load with their shapes checked | new |
| `0015` | HIP: every cache slot names an execution, not a layer (14 attention, 42 Gated DeltaNet) | new |
| `0016` | HIP: the four mHC kernels (`branch`, `merge`, `enter`, `leave`) in their own image, a port of the CUDA kernels, with their kernel contract | new |
| `0017` | HIP: the looped forward, YaRN rope kernels and the `sensenova_loop` registration (text only) | new |
| `0018` | the YaRN table builds where the core module lives | new |

`0001`-`0018` give back the measured tree exactly. Two later commits on the same work (the mHC routers following
the layer body, and the contract's gate-5 records for the mHC image) were made after the measured serve and are not
in this series.

## What is not tested

- **Serving after the reboot, and from GTT.** The TTM limit is live after a reboot, but no GPU job has run on that
  layout yet (the serve check after the reboot is pending); every number here used the 8 GB carve-out. Windows past
  8,192 tokens, more than one stream and other carve-out sizes are untested.
- **Other AMD GPUs.** Only the 780M (`gfx1103`). Other RDNA3 iGPUs (the 760M, the 890M's `gfx1150`) and discrete
  cards need their own `HIP_ARCH` and have not run.
- **The prompt cache, tool calls, images and drafters.** The prompt cache hung the GPU; tool calls need
  `libtfgrammar` and were not run; the HIP port refuses `--vision`; no neural drafter exists for the checkpoint.
- **A full `hip-kernel-contract` run.** The mHC rows pass; the full registry did not finish
  ([KNOWN-ISSUES](KNOWN-ISSUES.md#hip)).
- **Unattended boot and long soaks.** No soak test ran, and serving after an unattended boot has not been tried.
