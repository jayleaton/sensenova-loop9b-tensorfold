# Licensing

| part | license | notes |
| --- | --- | --- |
| this recipe's scripts, configs and docs | Apache-2.0 | [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE) |
| `patches/` | Apache-2.0 | modifications of TensorFold source; the TensorFold code they modify and quote as diff context stays under TensorFold's license |
| TensorFold | Apache-2.0 from 0.6.0; code written before 0.6.0 keeps its MIT notice | fetched from upstream at build time, not vendored; its `LICENSE`, `NOTICE`, `LICENSES/` and `THIRD_PARTY_NOTICES.md` are in every patched checkout |
| SenseNova-Loop-9B weights | Apache-2.0, per its [model card](https://huggingface.co/sensenova/SenseNova-Loop-9B) and [license](https://github.com/OpenSenseNova/SenseNova-Loop/blob/main/LICENSE) | not included; download them from SenseNova under those terms. The converted and quantized trees on Hugging Face ([README](../README.md#weights-on-hugging-face)) keep this license and credit SenseNova |

Both upstream licenses allow redistributing modifications and documentation. Apache-2.0 asks that a redistribution
keeps the license and any NOTICE, marks changed files, and keeps copyright and attribution notices. This repository
keeps TensorFold's NOTICE text in its own [`NOTICE`](../NOTICE). Each patch is a `git format-patch` record of a commit
with its author, date and message, which states the change. Applying the series yields a tree with TensorFold's
license files unchanged. SenseNova publishes no NOTICE file for the model.

If you redistribute converted weights, they remain SenseNova's model under Apache-2.0: keep the license, credit
SenseNova and say that the weights were converted.
