# HIP: ROG Ally X (Radeon 780M, gfx1103)

> **Tested on one ROG Ally X** (Ryzen Z1 Extreme, Radeon 780M, 24 GB LPDDR5X with the BIOS's 8 GB UMA carve-out),
> on Omarchy (Arch Linux) with Arch's ROCm 7.2.4 packages. Other AMD iGPUs, other distributions and other carve-out
> sizes are untested. Read [the limits](#what-is-not-tested) before you rely on it.

| | |
| --- | --- |
| Engine | upstream TensorFold `813ff4ccbf9d6610e991a96d53fda2b628f56347` (on `main`, 2026-11-19, version 1.2.10) + [`patches/hip/`](../patches/hip/) `0001`-`0020` |
| Engine tree | `493472234c7de38a888643d9c1a93713fb12acb2` ([`config/hip/`](../config/hip/)), the tree the 2026-11-20 runs were built from |
| Device | ROG Ally X (RC72LA, BIOS RC72LA.313): Ryzen Z1 Extreme, Radeon 780M (RDNA3, `gfx1103`, 12 CUs), 8 GB UMA carve-out |
| OS | Omarchy 4.5.0 (Arch), kernel `7.2.15-1-omarchy`, Mesa 26.2.5, `linux-firmware-amdgpu` 20260916 |
| ROCm | 7.2.4 from Arch `extra`: `rocm-hip-sdk 7.2.4-1`, `hip-runtime-amd 7.2.4-1`, `rocm-device-libs 2:7.2.4-2`, `comgr 2:7.2.4-2`, `rocm-llvm 2:7.2.4-2` (HIP 7.2.53211, AMD clang 22) |
| Weights | [`jayleaton/SenseNova-Loop-9B-MLX-4bit`](https://huggingface.co/jayleaton/SenseNova-Loop-9B-MLX-4bit), the Metal 4-bit tree, revision `0afa03c99f858625f4f8c5e3158d6ef6fdfad5bc`; text only on HIP |
| Memory | `ttm.pages_limit=ttm.page_pool_size=1966080` (GTT 7.5 GiB, below the carve-out); see [GPU memory](#2-gpu-memory-keep-gtt-below-the-carve-out) |

## Results on the 780M

One stream, greedy, an 8,192-token window, the engine inside the 8 GB carve-out. Two runs agree within 3%: 2026-11-19
on `0001`-`0018` (prompt cache off) and 2026-11-20 on `0001`-`0020` on the 1966080 boot (256-token replies, three runs
each; no GPU hang on that boot).

| prompt | decode, 2026-11-19 | decode, 2026-11-20 | first token |
| ---: | ---: | ---: | ---: |
| short (14 / 21 tokens) | **10.8 tok/s** | **10.9 tok/s** (10.87 without drafts) | 0.31 / 0.44 s |
| 4.8K (4,812 / 4,774 tokens) | 10.2 tok/s | 9.9 tok/s without drafts | 23.2 / 25.0 s |
| 6,479 tokens | 10.0 tok/s | | 31.7 s |
| 7.1K-7.4K (7,428 / 7,075 tokens) | 9.8 tok/s | 9.76 tok/s | 37.2 / 37.7 s |

Prompts read at about 190-200 tok/s (prompt length over first-token time: 191 at 4,774 tokens on 2026-11-20; 207, 204
and 200 at the three long prompts on 2026-11-19). Drafted and plain replies are token-identical in every pair.

**Prompt cache (`--prompt-cache-gib 1`).** A 5,246-token prompt resumes equal to fresh: first token 27.7 / 28.1 s cold,
**0.61 / 0.59 s** warm, the resumed reply's tokens equal to the cold one's both times. Decode with the cache on is
10.6 tok/s at a short prompt. Copy drafts with the cache on: a code edit 33.6 tok/s against 10.3 without drafts;
re-quote 12.5 against 13.0, chat 10.6 against 10.6; every pair token-identical. The cache hung the GPU on 2026-11-19
only because of the 16 GiB GTT setting ([GPU memory](#2-gpu-memory-keep-gtt-below-the-carve-out)); at 1966080 it is clean.

| check | result |
| --- | --- |
| `zig/tests/qwen35/serve_check.py` | **12 ok with the prompt cache off, and 12 ok with it on** (2026-11-20). Every exactness check passes, greedy and at t 0.8: solo == together, serial together == serial solo, the same twice, decoders unchanged by a long prompt, a long prompt among decoders == alone. The non-ok lines test nothing here: `drafted == serial` twice (0 drafts accepted, no drafter), and with the cache off `resumed == fresh` (nothing resumed). With the cache on, `resumed == fresh` needs a second cache-off server, which does not fit the carve-out beside the first; the cache check above shows the same property |
| mHC and YaRN kernels on gfx1103, `hip-kernel-contract` | `tf_mhc_branch` (both variants), `tf_mhc_merge`, `tf_mhc_enter`, `tf_mhc_leave`: nonfinite, repeat, shapes and sums pass against the FP64 reference, and each planted control is red as planted; 28 PASS lines name the mHC and YaRN kernels. The full run cannot finish on this machine ([KNOWN-ISSUES](KNOWN-ISSUES.md#hip)) |
| engine on the 780M, `tf-qwen35-test check` on stock Qwen3.5-4B (`lmstudio-community/Qwen3.5-4B-MLX-4bit`) | 46 passed, 0 failed, 6 skipped in 222.7 s; graphs replayed 2,859 of 3,055 rounds |
| the looped tree on the device, `tf-qwen35-test upload` | 32 layers, 1,023 buffers, 4,818 MiB read back equal |

At the profile's flags the engine budgets from the carve-out: one stream at 0.53 GiB beside the 4.71 GiB model, the
default 1.0 GiB reserve, loaded in 21 s.

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

### 2. GPU memory: keep GTT below the carve-out

> **Warning: do not raise GTT above the UMA carve-out.** On the Ally X with the 8 GB carve-out, a TTM limit of 16 GiB
> (`ttm.pages_limit=4194304`) brought GTT up at 15.3 GiB, larger than the carve-out. The engine then budgets from host
> memory, its buffers land in GTT, and the GPU **hangs under load**: 3 GPU resets (`MES failed to respond to
> msg=REMOVE_QUEUE`) in 5 minutes on a clean boot. With both TTM values at **1966080** (GTT 7.5 GiB, below the
> carve-out) the engine stays in the carve-out and **38 requests ran with 0 hangs**.

The 780M has no memory of its own. The BIOS's UMA setting reserves a carve-out the driver reports as VRAM (8 GB here,
left at that), and the rest of system memory is reachable as GTT, which the kernel's TTM limit caps (7.65 GiB here by
default). The engine's HIP admission budgets from the carve-out only while GTT is smaller than it; once GTT is larger,
it budgets from host memory and allocates in GTT. Keep GTT below the carve-out:

```
ttm.pages_limit=1966080 ttm.page_pool_size=1966080        # 1966080 pages of 4 KiB = 7.5 GiB, below the 8 GB carve-out
```

The kernel default (7.65 GiB here) is also below the carve-out, so leaving the TTM limit alone works too. Set the
values explicitly if your kernel's default could exceed the carve-out.

| TTM limit (both values) | GTT | where the engine allocates | result under load |
| --- | --- | --- | --- |
| `4194304` (16 GiB) | 15.3 GiB | GTT (`mem_info_gtt_used` 73 MiB to 5.9 GiB at load, VRAM about 0.57 GiB) | 3 GPU resets in 5 minutes, after 6, then 3, then 0 long prefills |
| `1966080` (7.5 GiB) | 7.5 GiB | the carve-out (`mem_info_vram_used` 6.3 GiB, GTT idle) | 38 requests, 0 hangs; 23 at prompts of 1.5K tokens or more (12 at 4,395 tokens, 6 at 7,257), no MES lines in the kernel log |

Those runs used the same HIP engine with the stock Qwen3.5-9B 4-bit tree at `--context 8192 --parallel 1
--prompt-cache-gib 0`. `HSA_ENABLE_SDMA=0` still hung, and so did a smaller prompt step (`TF_POLICY=prefill_step=256`).
Only this device, ROCm 7.2.4 and kernel `7.2.15` were tested; machines with a minimal carve-out, where the model
must live in GTT, were not.

**On a Limine + UKI boot (Omarchy), `/etc/kernel/cmdline` is ignored.** `limine-entry-tool` builds the kernel command
line from `KERNEL_CMDLINE[...]` in `/etc/default/limine` and `/etc/limine-entry-tool.d/*.conf`, and reads
`/etc/kernel/cmdline` only when `KERNEL_CMDLINE` is unset, which Omarchy's `/etc/default/limine` never leaves it.
A first attempt appended the values to `/etc/kernel/cmdline` and ran `mkinitcpio -P`; `mkinitcpio` warned that it
does not update Limine's entries, and the boot entry never carried them. What works is a drop-in and
`limine-mkinitcpio`:

```bash
echo 'KERNEL_CMDLINE[default]+=" ttm.pages_limit=1966080 ttm.page_pool_size=1966080"' |
  sudo tee /etc/limine-entry-tool.d/ttm.conf
sudo limine-mkinitcpio
sudo /usr/lib/limine/limine-entry-tool --get-cmdline | grep -o 'ttm[^ ]*'   # both values, before you reboot
```

After a reboot:

```bash
grep -o 'ttm[^ ]*' /proc/cmdline
cat /sys/module/ttm/parameters/pages_limit /sys/module/ttm/parameters/page_pool_size    # 1966080 1966080
cat /sys/class/drm/card*/device/mem_info_gtt_total /sys/class/drm/card*/device/mem_info_vram_total   # GTT below VRAM
```

| setting | state |
| --- | --- |
| `ttm.pages_limit=1966080`, `ttm.page_pool_size=1966080` on the boot command line | verified after a reboot |
| GTT total | 7.5 GiB, below the 8 GB carve-out, verified after a reboot |
| the engine's budget | the carve-out branch (`mem_info_vram_used` 6.3 GiB under load, GTT idle) |
| UMA carve-out | 8 GB, unchanged; no `amdgpu.*` parameters |

**The SenseNova numbers above ran inside the carve-out.** On 2026-11-19 the limit had been raised at runtime, but GTT
stayed at its boot size (7.65 GiB), below the carve-out; the 2026-11-20 runs, the prompt cache included, ran on the
1966080 boot with no GPU hang.

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

which fetches the pin, applies `patches/hip/` `0001`-`0020`, checks the tree, and runs:

```bash
zig build -Dhip=true native-hip -Dhip-arch=gfx1103 -Dhipcc=/opt/rocm/bin/hipcc \
  -Dhip-device-lib=/opt/rocm/amdgcn/bitcode
```

The binary is `engine-hip/zig-out/native-hip/bin/tensorfold-native`. The measured binaries were built in place with
`nice -n 19 zig build -j4 ...` and the same flags. `scripts/build-engine.sh hip` itself ran on the Ally X from a fresh
clone with the series through `0018` (2026-11-20, under `taskset -c 0-5 nice -n 19` to keep the desktop usable): it
reproduced that tree and built in 378 s.

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

which runs, with the profile in [`config/hip/serve.args`](../config/hip/serve.args):

```bash
engine-hip/zig-out/native-hip/bin/tensorfold-native serve loop9b-mlx4 \
  --name loop9b --port 8090 --context 8192 --parallel 1 --prompt-cache-gib 1
```

The measured serves were that command without `--name` (the model was served under its directory name). The flags
fit the 8 GB carve-out with the default 1.0 GiB reserve:

| flag | why |
| --- | --- |
| `--context 8192 --parallel 1` | one 8K window takes 0.53 GiB beside the 4.71 GiB model |
| `--prompt-cache-gib 1` | a 1 GiB prompt cache in the carve-out; a 5,246-token prompt comes back in 0.6 s instead of 28 s. `--prompt-cache-gib 0` turns it off |

Keep GTT below the carve-out ([GPU memory](#2-gpu-memory-keep-gtt-below-the-carve-out)): the prompt cache hung the
GPU on 2026-11-19 under the 16 GiB TTM setting, and runs clean at 1966080.

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

The first answers `391`. `serve_check` should print 12 `ok` lines and the expected non-ok lines above
(`drafted == serial` twice, and `resumed == fresh`, which cannot run against one server); any other `FAIL` is a real
difference.

## `patches/hip/`

On upstream `813ff4c`. `0001`-`0011` are the CUDA series' backend-neutral patches, applied as recorded; `0012`-`0020`
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
| `0019` | the mHC routers load with the layer body they belong to | new |
| `0020` | the kernel contract's gate-5 records for the mHC image and the YaRN rope | new |

`0001`-`0020` give back the tree of the 2026-11-20 runs exactly. The 2026-11-19 runs were on `0001`-`0018`
(tree `083d74e21f2678630829c59af6ee008a963331fe`), and the two runs' decode agrees within 3%.

## What is not tested

- **Serving from GTT hangs the GPU** ([above](#2-gpu-memory-keep-gtt-below-the-carve-out)); keep GTT below the carve-out.
  Windows past 8,192 tokens, more than one stream and other carve-out sizes are untested.
- **Other AMD GPUs.** Only the 780M (`gfx1103`); no other RDNA chip has been checked. Other iGPUs (the 760M, the
  890M's `gfx1150`) and discrete cards need their own `HIP_ARCH` and have not run.
- **Tool calls, images and drafters.** Tool calls need `libtfgrammar` and were not run; the HIP port refuses `--vision`; no neural drafter exists for the checkpoint.
- **A full `hip-kernel-contract` run.** The mHC and YaRN rows pass; the full registry needs about 28 GiB and cannot
  finish on the Ally X ([KNOWN-ISSUES](KNOWN-ISSUES.md#hip)).
- **Unattended boot and long soaks.** No soak test ran, and serving after an unattended boot has not been tried.
