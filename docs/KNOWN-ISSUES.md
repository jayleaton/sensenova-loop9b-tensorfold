# Known issues

## Both backends

- **No task gate on CUDA yet.** Metal passes code 64/64 and reasoning 64/64 with prose at the reference's noise floor
  ([METAL](METAL.md#task-gate)); CUDA has BF16 parity and quantized cosine, not the greedy task gate.
- **Text only.** The converter leaves out the vision tower, so neither backend serves images. On Metal the tower is
  being wired and has an open OCR defect; nothing is claimed for image input until it is fixed and measured.
- **No drafter.** The checkpoint ships no MTP or draft weights, so there is no speculative decoding.
- **Context.** The checkpoint advertises 524,288 positions; the measured profiles use 32K (CUDA) and 2K (Metal), and
  Metal admission caps at 262,144. Longer contexts are not measured.

## CUDA

- **mHC drift above the gate.** Over 56 executions the engine is 2.203 `max|Δlogit|` from the BF16 reference (top-1
  and the top-5 set match; greedy 30/31). The planned gate is 0.05. The suspected cause is rounding order in `merge`:
  the reference rounds `mix` and `post` to bf16 and rounds the two addends separately, and the engine accumulates in
  fp32 and rounds once.
- **The quantized CUDA receipts used an earlier conversion.** The q8/q4 fidelity numbers in [CUDA](CUDA.md) were
  measured on trees from an earlier converter that wrote the same format but also folded `1 + w` into the 24
  `linear_attn.norm` tensors. The converter shipped here leaves that gated norm as stored, which is what the CUDA
  loader reads unshifted in both the BF16 and the MLX path, and what the Metal results use. Speed does not depend on
  it; the quantized fidelity on CUDA has not been re-measured with this converter yet.
- **Prefill is 93.5% of the loop's bound** (2,048 of 2,190 tok/s at 8k). The rest is the mHC kernels, about 7% of
  kernel time.

## Metal

- **Prose greedy agreement 19/64**, at the reference's own bf16/fp32 noise floor ([METAL](METAL.md#task-gate)).
- **Decode is 60 tok/s against 83 for 56 executions without the window**: at one row the router is a single
  threadgroup.
- **One stream.** Concurrency is not measured.
- **M5 Ultra only.** Other Apple chips are not measured.
