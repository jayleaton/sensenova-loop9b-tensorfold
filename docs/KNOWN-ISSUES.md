# Known issues

## Both backends

- **No neural drafter.** The checkpoint ships no MTP or draft weights. Copy drafts (drafts taken from a stream's own
  context) are on and keep tokens unchanged; a DFlash2 drafter waits on published weights.
- **No task gate on CUDA yet.** Metal passes code 64/64 and reasoning 64/64 with prose at the reference's noise floor
  ([METAL](METAL.md#task-gate)); CUDA has BF16 parity, quantized top-1 and the serving checks, not the greedy task gate.
- **Images: block fonts.** The model cannot read text drawn in a 5x7 pixel block font; ordinary fonts read fine.
- **Images: invented model names.** Asked to name models on a chart that has no model legend, it answers with names
  that are not on the chart; reading the text that is there is correct.

## CUDA

- **mHC drift above the gate.** Over 56 executions the engine is 2.203 `max|Δlogit|` from the BF16 reference (top-1
  and the top-5 set match; greedy 30/31). The planned gate is 0.05.
- **Copy drafts at four chats.** With four chats decoding at once, copy drafts are about 2% slower than none (52.9
  against 54.1 tok/s aggregate on 256-token code replies); at one and two chats they are faster. Tokens are the same
  either way.
- **4-bit on CUDA loses too much**: top-1 on 220 of 249 positions against BF16, where 8-bit keeps 246. Serve 8-bit.
- **Contexts past the trained 262,144 positions** run on the YaRN factor-4 stretch. A 1,027,507-token needle was found
  on an earlier build of the same series; long-context reading quality between factor 2 and 4 is still being compared.
- **Prefill is 93.5% of the loop's bound** (2,048 of 2,190 tok/s at 8k, 4-bit). The rest is the mHC kernels.

## Metal

- **Prose greedy agreement 19/64**, at the reference's own bf16/fp32 noise floor ([METAL](METAL.md#task-gate)).
- **Decode is 60 tok/s against 83 for 56 executions without the window**: at one row the router is a single
  threadgroup.
- **8 GB Macs are not supported yet.** 2-bit packs fit but lose code and reasoning; 16 GB numbers come from a
  working-set cap on the M5 Ultra, not a 16 GB machine.
- **Images: one a request and no video.** More than one image a request is untested; video parts are refused.
- **Image speed predates the router changes**: 30.7 tok/s decode with an image was measured before `0023`/`0024`.
- **Chips measured:** M5 Ultra and M4 Pro. Others are untested.
