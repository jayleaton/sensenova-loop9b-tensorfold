# Known issues

## All backends

- **No neural drafter.** The checkpoint ships no MTP or draft weights. Copy drafts (drafts taken from a stream's own
  context) are on and keep tokens unchanged; a DFlash2 drafter waits on published weights.
- **No greedy task gate on CUDA yet.** Metal passes code 64/64 and reasoning 64/64 with prose at the reference's noise
  floor ([METAL](METAL.md#task-gate)); CUDA has BF16 parity, quantized top-1, the serving checks and task benchmarks on
  the 4-bit tree ([CUDA](CUDA.md#task-benchmarks)), not the greedy task gate.
- **Tool calls under `auto`: prose instead of a call** (measured on CUDA). On BFCL 116/200, and 72 of the 84 misses answer in prose with
  no tool markup at all. `tool_choice: "required"` or a named function (with `libtfgrammar`) forces the call. Choosing
  a tool at a distance under `auto` is weak; it is the model, not the quantization.
- **Agentic coding success is low** (measured on CUDA). OpenCode finished 3 of 10 tasks with four sessions at once and 2 of 5 alone; most misses
  hit the task's time cap, 9.3% of tool calls failed (mostly `edit` calls whose `oldString` did not match), and long
  digit strings can run away (a made-up `fib(250)`).
- **Images: block fonts.** The model cannot read text drawn in a 5x7 pixel block font; ordinary fonts read fine.
- **Images: invented model names.** Asked to name models on a chart that has no model legend, it answers with names
  that are not on the chart; reading the text that is there is correct.

## CUDA

- **mHC drift above the gate.** Over 56 executions the engine is 2.203 `max|Δlogit|` from the BF16 reference (top-1
  and the top-5 set match; greedy 30/31). The planned gate is 0.05. The v6 seam fix (`0023`, validated, not yet
  served) brings mean KL from 0.0244 to 0.0186 over 736 positions ([CUDA](CUDA.md#v6-the-loop-seam-and-one-barrier-prefill)).
- **v6 is validated, not served.** `config/cuda-v6/` passes the token check, the strict tool gate and 231K recall; its
  decode was not re-measured (unchanged by construction) and it has not run as the live server.
- **FP8 prompt attention flips tokens.** `TF_QWEN_PREFILL_FP8=1` flips 2-3% of top-1 tokens; it is off by default.
- **Images on CUDA.** Ordinary fonts read (4/4 in one request); one image alone was misread once; at most 4 images a
  request; no parity test for the image path in the series yet.
- **Copy drafts at four chats.** With four chats decoding at once, copy drafts are about 2% slower than none (52.9
  against 54.1 tok/s aggregate on 256-token code replies); at one and two chats they are faster. Tokens are the same
  either way.
- **4-bit on CUDA loses too much**: top-1 on 220 of 249 positions against BF16, where 8-bit keeps 246. Serve 8-bit.
  The task benchmarks ran on the 4-bit tree; the served 8-bit tree was not benchmarked.
- **Contexts past the trained 262,144 positions** run on the YaRN factor-4 stretch. A 1,027,507-token needle was found
  on an earlier build of the same series; long-context reading quality between factor 2 and 4 is still being compared.
- **Prefill is 93.5% of the loop's bound** (2,048 of 2,190 tok/s at 8k, 4-bit). The rest is the mHC kernels.

## RTX

- **No live KV in host RAM.** The host prompt cache keeps finished prompt states, not a running chat's KV window, so
  the window is bounded by the card (212,992 fp8 tokens for `sn-6.0` at one chat).
- **Measured on one card in WSL2.** RTX 5070 Ti 16 GB only; native Windows and other cards are untested. `sn-4.0`'s
  tool calls were not run on the card. Decode varies with desktop load and clock decay (about 39-47 tok/s for `sn-6.0`).

## HIP

- **GPU hangs with the prompt cache on.** A serve with `--prompt-cache-gib 1.2` hung the 780M on its first long
  prompt (`MES failed to respond to msg=REMOVE_QUEUE`, then a GPU reset), and so did a second full
  `hip-kernel-contract` run a few seconds later. The profile serves with the cache off, which ran without a hang;
  the cause is not found.
- **Two upstream prompt-attention kernels fail a contract case on gfx1103.** `tf_fa_paged_bf16` and
  `tf_fa_wide_bf16` return a finite value for 255 of 49,152 outputs where the FP64 reference gives NaN (the
  `nonfinite` case). Neither is changed by this series. The full contract run stopped before its summary.
- **One 8K stream.** The 8 GB UMA carve-out holds the 4.71 GiB model and one 8,192-token window with a 0.5 GiB
  reserve. The TTM limit is raised and live after a reboot (GTT 15,674 MiB), but the engine budgeted from the
  carve-out in every measured run; serving from GTT, longer windows and more streams are untested.
- **No fidelity gate against BF16 on HIP.** The mHC kernels pass their FP64 contract and `serve_check` passes its
  exactness checks; no teacher-forced KL or top-1 comparison has run on the 780M.
- **Text only, no tool-call grammar.** `--vision` is refused on HIP, and `libtfgrammar` was not built, so required and
  named tool calls were not tested.
- **Measured on one Ally X.** The 780M (`gfx1103`) only; other AMD iGPUs and discrete cards, other distributions and
  other carve-out sizes are untested, as are unattended boot and long soaks.

## Metal

- **Prose greedy agreement 19/64**, at the reference's own bf16/fp32 noise floor ([METAL](METAL.md#task-gate)).
- **Decode is 60 tok/s against 83 for 56 executions without the window**: at one row the router is a single
  threadgroup.
- **8 GB Macs are not supported yet.** 2-bit packs fit but lose code and reasoning; 16 GB numbers come from a
  working-set cap on the M5 Ultra, not a 16 GB machine.
- **Images: one a request and no video.** More than one image a request is untested; video parts are refused.
- **Image speed predates the router changes**: 30.7 tok/s decode with an image was measured before `0023`/`0024`.
- **Chips measured:** M5 Ultra and M4 Pro. Others are untested.
