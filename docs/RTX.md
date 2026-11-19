# RTX 16 GB: EXL3 packs on the CUDA engine

> **Tested on an RTX 5070 Ti 16 GB (WSL2).** The packs were sized on a DGX Spark under a 16 GB memory cap, then run on
> the card: the gate's logits equal the Spark's run of the same pack, and the card's own windows, speed, tool calls
> and a 30-minute soak are below. Other 16 GB cards and native Windows are untested.

| | |
| --- | --- |
| Engine | upstream TensorFold `038e4a278` + [`patches/cuda/`](../patches/cuda/) `0001`-`0022` + [`patches/rtx/`](../patches/rtx/) `0001`-`0002` |
| Engine tree | `0f61f4723db864b510f9a278b6dd5d679b344acb` ([`config/rtx/`](../config/rtx/)), checked 2026-11-19 |
| Card tested | RTX 5070 Ti 16 GB (Blackwell, `sm_120`), WSL2 Ubuntu 24.04, the Windows desktop on the same card |
| Weights | EXL3 packs quantized from the BF16 checkpoint with exllamav3 ([quantize](#quantize)) |

`patches/rtx/`:

| patch | change |
| --- | --- |
| `0001` | the EXL3 loader reads SenseNova-Loop's loop and mHC; the capability list keeps Qwen's decisions; the quantize chain: Hessians from the real looped model, exllamav3 conversion at 4.0/5.0/6.0 bpw, the extras attached |
| `0002` | `zig/src/native/cuda.zig` back to 600 lines: a family's own model_type and its aliases in one registry pass |

## Results on the RTX 5070 Ti

| pack | largest fp8-KV window on the card | gate cos / top-1 (298 positions) | decode, drafts off | prefill | tool calls | other |
| --- | ---: | --- | --- | --- | --- | --- |
| **`sn-6.0`** | 245,760 | 0.99931 / 290, logits equal to the Spark's run of the pack | 46.6 / 45.7 / 43.9 tok/s at 4k / 8k / 32k (best run; 44-45 and 39-40 over five quiet A/B rounds) | 2,232 / 2,203 / 2,055 tok/s; first token 1.8 s at 4k, 16.3 s at 32k | 36/36 | greedy repeats 10/10; a 30-minute soak, 241 requests, 0 failures, no memory growth |
| `sn-4.0` | 319,488 | 0.99486 / 279 | 59.1 / 60.1 / 54.4 tok/s | 2,364 / 2,279 / 2,126 tok/s | not run | |
| `sn-5.0` | 286,720 | 0.99779 / 290, equal to the Spark | 50.5 / 51.2 / 44.0 tok/s | 1,950 / 1,885 / 1,789 tok/s | | peak 9.32 GiB at a 32k bf16-KV window |

Served at the flags below, `sn-6.0` peaked at 12.49 GiB in the engine and 15.76 of the card's 15.92 GiB with the
Windows desktop's share. At a 32k bf16-KV window the engine's peak equals the Spark's (10.36 GiB): the BF16 embedding
rows (1.89 GiB) stay on the card there and move to host memory only when a long window needs the room.

The table's speed, tool-call and soak rows were measured with the EXL3 code on the CUDA series through `0009` (the
prefill build). The copy-draft and host-cache rows below were measured on the card with this recipe's RTX tree, the
EXL3 code on the served CUDA series `0001`-`0022`; a five-round A/B on that tree decoded `sn-6.0` at 44.9 / 39.7 tok/s
at 4k / 32k against 44.1 / 38.9 on the older one. The longest prompt held on the card is 191,692 tokens (`sn-6.0`, the
212,992 fp8 window). The card's clock decays under sustained load: the same binary fell from 48.9 to 38.4 tok/s over
consecutive serves, so quote a quiet, cool card.

### Copy drafts on the card

Copy drafts are on by default (drafts from a chat's own context, verified, so tokens are unchanged). `sn-6.0`, one
chat, a quiet card, drafts off against on:

| reply | off | on |
| --- | ---: | ---: |
| re-quote a passage | 45.5 tok/s | 660.1 tok/s |
| code edit | 46.5 | 533.8 |
| chat | 45.1 | 119.4 |
| summarise | 45.2 | 87.4 |

16 of 16 reply pairs are token-identical on and off, on both `sn-6.0` and `sn-4.0`. At `--parallel 4 --context 32768`
with the desktop busy, aggregate tok/s with drafts on:

| chats | re-quote | code edit |
| ---: | ---: | ---: |
| 1 | 115.0 | 153.5 |
| 2 | 126.6 | 175.1 |
| 4 | 128.2 | 178.3 |

Re-quote with drafts off aggregates 24.4 / 42.2 / 71.0 at 1 / 2 / 4 chats on that run.

### Host-RAM prompt cache

Serve without `--prompt-cache-gib 0` and the engine keeps finished prompt states in pinned host RAM (default 8 GiB, 0
device bytes), so a returning conversation restores its prefix instead of reading it again. `sn-6.0` at the 212,992
fp8 window:

| prefix | cold | restored |
| ---: | ---: | ---: |
| 26,805 tokens | 24.2 s | 2.15 s (11x) |
| 109,313 tokens | 133.6 s | 4.48 s (30x) |

Restore is PCIe-bound, about 0.9-1.0 GiB/s through WSL2. The device peak is 12.49 GiB with the cache on or off; host
RSS was 9.8 GiB with four states held, so give WSL2 enough memory in `.wslconfig`. The disk tier works too
(`--prompt-cache-gib 1 --spill-gib 4 --prompt-cache-disk DIR`: 3 evictions, 0 failures, a spilled state back in 3.03 s).
With the cache on, `sn-6.0` fp8 still fits `--parallel 4 --context 32768` (peak 12.94 GiB; 40,960 refused) and
`--parallel 2 --context 81920` (13.31 GiB; 90,112 refused). Only kept prompt states, the embedding fallback and the
disk tier's staging live in host RAM: **a live chat's KV window cannot**, so the window is still bounded by the card.

**Pick: `sn-6.0` at 16 GB**, next to q8 on quality. **`sn-4.0` for speed** or the longest window.

## Quality and size, measured on the Spark

| pack | size | cosine / top-1 vs BF16 (249 positions) | device peak, 32k, bf16 KV | largest fp8-KV window, 15.0 / 14.5 GiB cap |
| --- | ---: | --- | ---: | --- |
| `sn-4.0` | 6.31 GB | 0.99393 / 236 | 8.51 GiB | 294,912 / 278,528 |
| `sn-5.0` | 7.17 GB | 0.99807 / 244 | 9.31 GiB | 270,336 / 253,952 |
| `sn-6.0` | 8.29 GB | 0.99937 / 244 | 10.35 GiB | 229,376 / 212,992 |
| MLX q8 (for scale) | | 0.99975 / 246 | | |

15.0 GiB is what a 16 GB card leaves after the driver context; 14.5 GiB is a Windows desktop whose display also holds
part of the card. The card itself admitted larger windows than the 14.5 GiB cap (the table above), since what the
desktop holds varies; the serve flags keep the 14.5 GiB window so they fit with a busier desktop.

## Build

On Linux or WSL2 with the CUDA 13 toolkit and Zig 0.17.0:

```bash
scripts/build-engine.sh rtx engine-rtx        # SM=120 by default (RTX 50); set SM= for another card
```

Without root, point the build at a user-level toolkit with `NVCC=$CUDA_DIR/bin/nvcc` ([WSL2 steps](#windows-wsl2-steps)).
Structured output and `tool_choice` required or named need `libtfgrammar` beside the server
([CUDA](CUDA.md#required-and-named-tool-calls)).

## Serve

```bash
scripts/serve.sh rtx engine-rtx sn-6.0        # --parallel 1 --kv-cache fp8 --context 212992 --prompt-cache-gib 0
```

The profile turns the host prompt cache off so it fits a machine with little RAM. To keep returning conversations in
host RAM, or to serve several chats, run directly:

```bash
engine-rtx/zig-out/native/bin/tensorfold-native serve sn-6.0 --port 8090 --parallel 1 --kv-cache fp8 --context 212992   # 8 GiB host cache
engine-rtx/zig-out/native/bin/tensorfold-native serve sn-6.0 --port 8090 --parallel 4 --kv-cache fp8 --context 32768    # 4 chats
engine-rtx/zig-out/native/bin/tensorfold-native serve sn-6.0 --port 8090 --parallel 2 --kv-cache fp8 --context 81920    # 2 chats
```

The other two fits, run directly:

```bash
engine-rtx/zig-out/native/bin/tensorfold-native serve sn-4.0 --port 8090 --parallel 1 --kv-cache fp8 --context 278528 --prompt-cache-gib 0
engine-rtx/zig-out/native/bin/tensorfold-native serve sn-6.0 --port 8090 --parallel 1 --context 32768 --prompt-cache-gib 0   # bf16 KV
```

If the desktop needs more of the card, cap the engine's budget with `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=14.5`.

## Windows: WSL2 steps

Run the engine in WSL2 Ubuntu 24.04. A native Windows build exists upstream (`docs/WINDOWS.md`), but these packs were
only run on Linux and WSL2, which is that same path with the Windows driver serving the GPU.

1. **Driver and WSL.** On Windows, the current NVIDIA display driver (`nvidia-smi` reports CUDA 13.x); no CUDA toolkit
   on the Windows side. Then `wsl --install -d Ubuntu-24.04`, reboot if asked, `wsl --update`, and inside Ubuntu
   `sudo apt update && sudo apt install -y build-essential git curl`. Install only the toolkit in WSL, never a Linux
   display driver or the `cuda`/`cuda-drivers` metapackages:

   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
   sudo apt install -y cuda-toolkit-13-0
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && . ~/.bashrc
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
   nvcc --version
   ```

   **Without sudo**, take the same toolkit's parts from NVIDIA's redist tarballs (`redistrib_13.0.3.json` lists each
   one's path and sha256): `cuda_nvcc`, `cuda_crt`, `cuda_cudart`, `cuda_cccl`, `libnvvm`, `cuda_cuobjdump`,
   `cuda_nvdisasm`, `cuda_profiler_api` and `libnvptxcompiler`, each extracted with `--strip-components=1` into one
   directory, `$CUDA_DIR` (for example `cuda-13.0` under your home directory); then build with
   `NVCC=$CUDA_DIR/bin/nvcc`. The card was tested this way.

   If the build is killed for memory, set `memory=24GB` (or half the machine's RAM) in `%USERPROFILE%\.wslconfig`.
2. **Zig 0.17.0:**

   ```bash
   mkdir -p ~/.local/bin && cd ~/.local
   curl -LO https://ziglang.org/download/0.17.0/zig-x86_64-linux-0.17.0.tar.xz
   tar -xJf zig-x86_64-linux-0.17.0.tar.xz && ln -sf ~/.local/zig-x86_64-linux-0.17.0/zig ~/.local/bin/zig
   ```

3. **Build** with this recipe: `scripts/build-engine.sh rtx engine-rtx` (several minutes of nvcc, no GPU needed). It
   fetches the pinned upstream commit, applies both series and checks the tree.
4. **The pack** goes on ext4 under your home directory, never under `/mnt/c`, where the checkpoint reader's `O_DIRECT`
   does not reach. Check every file against the pack's `sha256` list (13 entries).
5. **Serve** as above. It listens on `http://127.0.0.1:8090/v1`; add `--host 0.0.0.0` to reach it from Windows. Close
   other GPU work first: a busy desktop halved decode on the first run.
6. **Check:** one greedy chat, and the card's peak while it runs:

   ```bash
   curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "loop9b",
     "messages": [{"role": "user", "content": "Write a Python function fib(n) that returns the n-th Fibonacci number."}],
     "max_tokens": 200, "temperature": 0}'
   ```

   Read the card's memory from Windows (Task Manager's dedicated GPU memory, or `nvidia-smi` in a Windows terminal).
   **WSL's `nvidia-smi` prints a garbage used figure near 2^44 MiB once the card is nearly full**; the engine admits
   from the driver's free memory, so a startup refusal naming the budget while the desktop is busy means the card is
   full, not that the flags are wrong. The engine's own peak is on `/metrics` with `--metrics-open`. Scoring the
   teacher-forced gate yourself needs `numpy` (`sudo apt install -y python3-numpy`).

## Quantize

exllamav3 cannot trace this model (`sensenova_loop` is not a registered architecture, and its `qwen3_5` forward has no
loop or mHC), so the chain captures Hessians from the real looped model with transformers, one per unique weight summed
over both visits, and lets exllamav3 quantize from them. The steps, pinned versions and recipes are in the patched
engine: `docs/exl3-sensenova-quantize.md`, `docs/exl3-recipes/` and `tools/exl3_sensenova_chain.sh`. No packs are
published.
