# Credits

- **[SenseNova](https://huggingface.co/sensenova): [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B)**
  ([GitHub](https://github.com/OpenSenseNova/SenseNova-Loop)), the model, its looped decoder and two-stream
  manifold-constrained hyper-connections, the checkpoint at revision `329a65f701ba8c48b73c30d0cbaa6f0449733673`, and the
  reference `modeling_sensenova_loop.py` that defines the arithmetic every parity number here is checked against.
  Apache License 2.0 ([license](https://github.com/OpenSenseNova/SenseNova-Loop/blob/main/LICENSE)).
- **[Ash Hart (ashhart) / TensorFold](https://github.com/ashhart/TensorFold)** and the TensorFold contributors: the
  engine and server. The `qwen3_5` CUDA family and the dense `qwen27` Metal family, the Gated DeltaNet and
  hybrid-attention kernels, the YaRN table builder (`zig/src/core/rope_yarn.zig`), the MLX affine readers and
  `quantizeMlx`, the Qwen3.5-VL tower, the kernel-contract harness and the `tools/bench_openai.py` /
  `tools/bench_concurrent.py` clients are upstream work. The converter reproduces the engine's own
  `quantizeMlx` (`zig/src/core/affine4_host.zig`). Apache License 2.0;
  code written before 0.6.0 keeps its MIT notice.
- [Hugging Face transformers](https://github.com/huggingface/transformers) 5.18.0 and
  [PyTorch](https://github.com/pytorch/pytorch) 2.15.0: the reference runs.
- [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.32.2: the Metal prefill comparison.
- [`mlx-community/Qwen3.5-9B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit),
  [`mlx-community/Qwen3.5-9B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) and
  [`mlx-community/Qwen3.5-2B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-2B-MLX-4bit): the stock controls.
