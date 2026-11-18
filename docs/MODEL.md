# The checkpoint

[`sensenova/SenseNova-Loop-9B`](https://huggingface.co/sensenova/SenseNova-Loop-9B) at revision
`329a65f701ba8c48b73c30d0cbaa6f0449733673`, by SenseNova, Apache-2.0. Every fact below comes from the checkpoint's own
`config.json`, safetensors headers, model card and reference `modeling_sensenova_loop.py`.

## Architecture

A Qwen3.5 dense hybrid-attention backbone (`model_type` `sensenova_loop`, inner text config `qwen3_5_text`) with two
additions TensorFold had not executed on that family: a shared-weight loop and a two-stream, input-dependent mHC.

| property | value |
| --- | --- |
| physical layers / executions | 32 / **56**: `[0,4)` + `[4,28)` twice + `[28,32)` (`loop_settings` `{start: 4, end: 28, counts: [2]}`) |
| attention schedule | 3 Gated DeltaNet : 1 full attention, `full_attention_interval: 4` |
| hidden / heads / KV heads / head dim | 4096 / 16 / 4 / 256, `attn_output_gate` |
| linear attention | 16 key heads x 128, 32 value heads x 128, conv kernel 4, fp32 recurrent state |
| MLP / vocabulary | SwiGLU 12288 / 248,320, untied head |
| RoPE | YaRN, theta 1e7, factor 2.0 over 262,144 (to 524,288), partial 0.25 (64 of 256), M-RoPE interleaved `[11,11,10]` |
| mHC | 2 streams over `[4, 28)`, 20 log-space Sinkhorn iterations, fp32 routers `[8, 8192]` + `[8]` per site |
| vision | Qwen3.5-VL tower: depth 27, hidden 1152, patch 16, temporal patch 2, merge 2, output 4096 |
| MoE / MTP | none; `mtp_num_hidden_layers: 0` and no `mtp.*` tensors, so there is no drafter |

## Loop and mHC: the invariants

- Weights are shared across the two passes; cache state is not. Each visit owns its KV or recurrent slot: **14**
  attention slots (not 8) and **42** Gated DeltaNet slots (not 24). Sizing by physical layer reads the other pass's
  state and still produces fluent text, so it fails silently.
- Position embeddings are computed once and reused by both passes.
- mHC per site: `x̂ = x * rsqrt(mean(x²) + 1e-6)` over the two flattened streams, `logits = W·x̂ + b` in fp32;
  `pre = sigmoid(logits[0:2])`, `post = 2 * sigmoid(logits[2:4])`, `mix = sinkhorn_20(logits[4:8] as 2x2)`. The layer runs
  once, on `branch = pre · streams`; the merge is `streams = mix @ streams + post * branch`.
- The streams collapse to their mean at physical layer 27; the second pass re-enters at layer 4 with two identical
  streams.
- The routers stay fp32: never quantized, never cast.
- `loop_gate.logits` is read by nothing at inference.

## Size and cache

| component | size |
| --- | ---: |
| BF16 checkpoint (text, vision, fp32 routers) | 17.53 GiB |
| MLX affine 8-bit g64, text only | 9.3-9.7 GiB |
| MLX affine 4-bit g64, text only | 4.7-5.6 GiB (measured trees: 4.72 GiB on Metal, 5.55 GiB on CUDA) |
| MLX affine 4-bit g64 with the bf16 vision tower | 5.57 GiB (Metal) |
| KV per token, bf16 | 56 KiB (14 attention executions x 4 heads x 256 x K,V x 2 B) |
| recurrent state per stream | about 90 MiB (42 Gated DeltaNet executions), constant in context |

KV for 32K / 128K / 256K tokens is 1.75 / 7.0 / 14.0 GiB in bf16. With an 8-bit KV the card's "4-bit about 4.9 GB, 256K
about 15 GiB" follows from the 14 attention executions.
