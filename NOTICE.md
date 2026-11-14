# Third-party components

This repository's own code is MIT (see LICENSE). It builds on, and the output you produce contains or runs, the following. Check each licence yourself before you ship.

| Component | Used for | Licence (as published by its authors; verify) |
|---|---|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (Hugging Face) | the base model you fine-tune. The model you ship is a derivative of it. | Apache-2.0 |
| [PyTorch](https://pytorch.org) | training | BSD-style |
| [Hugging Face transformers](https://github.com/huggingface/transformers), [optimum](https://github.com/huggingface/optimum) | training, tokenizer, ONNX export | Apache-2.0 |
| [ONNX](https://onnx.ai), [ONNX Runtime](https://onnxruntime.ai) (Python and `onnxruntime-web`) | model format, graph editing, running the model in the browser | Apache-2.0 / MIT |
| [transformers.js](https://github.com/huggingface/transformers.js) | tokenizer in the browser (the model itself runs through ONNX Runtime directly) | Apache-2.0 |

Your own `knowledge/` content is yours; the sample bakery data in this repo is invented and may be reused freely.
The fine-tuned model inherits whatever restrictions apply to the base model and to any data you train on.
