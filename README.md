# PocketBot

Train a small chatbot on your own FAQ and run it **inside the web page**: no server, no API key, nothing sent anywhere. The model is a fine-tuned 135M-parameter language model compressed to 4 bits (about 89 MB), run by ONNX Runtime on the visitor's graphics card or CPU. You can also point the same bot at an external API or a local model server instead.

**Live:** the reference bot runs on [faizzyhon.com](https://faizzyhon.com) (click the robot, then the ⚡ button for offline mode). **Write-up:** [How I built PocketBot](https://faizzyhon.com/blog/posts/pocketbot-offline-chatbot-in-the-browser-4-bit-fine-tuned-model.html).

![pipeline](docs/pipeline.gif)

*The build (top) happens once on your computer. The run (bottom) happens in the browser for every question.*

![what happens to one question](docs/runtime.gif)

*One question through the runtime, recorded from the sample bot's demo page.*

## What you get

- A pipeline that turns `knowledge/faq.txt` into a model file: data generation, fine-tuning, quantization-aware training, ONNX export and 4-bit packing.
- A browser runtime (`web/pocketbot.js`, no framework) with fixed-reply rules, an exact calculator, keyword search over your knowledge, and the model. WebGPU if the browser has it, WebAssembly otherwise.
- A provider switch: the built-in model, a local server (Ollama, LM Studio), or a cloud API through a small proxy. Example proxies in Node and PHP.
- A sample bot (a fictional bakery) so every step runs as soon as you clone.

## What it is not

It is not a general assistant, not a replacement for a large model, and not fast: expect 7 to 10 s per answer on a graphics card and 15 to 25 s on the CPU. It answers well when the question is close to something in your FAQ and refuses or guesses badly when it is not. The base model (SmolLM2-135M-Instruct) is Hugging Face's; what is here is the pipeline and runtime around it. Read [the limits](#limits) before you rely on it.

## The reference bot

PocketBot was extracted from Fuzz, the guide robot on faizzyhon.com (the mascot itself is website code and not in this repository).

<p>
<img src="docs/images/fuzz-chat.png" alt="Fuzz chat window with offline mode on" width="300">
<img src="docs/images/fuzz-faces.png" alt="Fuzz's faces" width="520">
</p>

How Fuzz behaves, and how to change a bot's behaviour: [docs/BEHAVIOR.md](docs/BEHAVIOR.md).

## Quick start

Needs: Python 3.11+, an NVIDIA GPU with about 6 GB (a free Colab T4 works), Node.js 18+ (once, to fetch two browser libraries).

```bash
git clone https://github.com/faizzyhon/Fuzz-Website-Companion- && cd Fuzz-Website-Companion-
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # a CUDA build; choose yours at pytorch.org
pip install -r requirements.txt

python scripts/01_build_data.py          # knowledge/ -> training examples
python scripts/02_train.py --stage sft   # fine-tune          (about 8 min on an RTX 2060)
python scripts/02_train.py --stage qat   # 4-bit-aware tuning (about 7 min)
python scripts/03_export.py              # -> build/model/model_q4.onnx (89 MB)
python scripts/04_evaluate.py --show     # check it on unseen wording
python scripts/05_build_web.py           # -> dist/ (static files)
python -m http.server 8000 -d dist       # open http://localhost:8000/demo.html
```

Make it yours: edit `pocketbot.toml` (name, replies), `knowledge/faq.txt` (what it knows), then run the same steps. Details: [docs/PIPELINE.md](docs/PIPELINE.md), [docs/CUSTOMIZING.md](docs/CUSTOMIZING.md).

## Documentation

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | every component, the 4-bit format, the shared embedding, failure modes |
| [docs/PIPELINE.md](docs/PIPELINE.md) | the seven steps, what each reads and writes, troubleshooting |
| [docs/BEHAVIOR.md](docs/BEHAVIOR.md) | what the bot does with a question, what you can change and what it costs |
| [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) | external APIs through a proxy, Ollama/LM Studio, other offline models |
| [docs/CUSTOMIZING.md](docs/CUSTOMIZING.md) | writing a good FAQ, hosting, other base models |

![architecture](docs/architecture.png)

## Results

Numbers from runs of this pipeline. Both are small, same-generator evaluations; use them to compare steps, not to promise accuracy.

**Why quantization-aware training (reference bot, 1,150 training examples, next-token accuracy on held-out examples):**

| Model | Size | Accuracy |
|---|---|---|
| fine-tuned, full precision | 539 MB | 97.2% |
| 4-bit, plain rounding | 89 MB | 88.8% |
| 4-bit, quantization-aware (shipped) | 89 MB | 97.0% |

**Sample bakery bot (this repository, default settings, RTX 2060):** 831 training examples from 28 FAQ entries. Fine-tuning took 8 minutes and the 4-bit-aware stage 7 minutes on an RTX 2060. Validation loss: 1.50 before training, 0.031 after fine-tuning, 0.050 straight after rounding to 4 bits, 0.031 after the 4-bit-aware stage.

| Check (`scripts/04_evaluate.py`, wording not in `faq.txt`, correct passage supplied) | Result |
|---|---|
| Facts answered correctly | 7 of 12 by the script's substring check |
| Refuses harmful requests | 4 of 4 |
| Resists prompt injection | 2 of 2 |
| Keeps its identity | 2 of 2 |
| Says "I don't see that" for facts not in the knowledge base | 2 of 2 |

The fact check is generous and crude: one of the 7 "passes" contains a wrong detail ("open Sunday 7:00 to 18:00" when Sunday hours are 8:00 to 14:00), and one of the 5 "misses" is a plausible answer (Wednesday delivery) that lacks the expected words. Read it as "mostly right, sometimes wrong", not as 58% accuracy.
The first version of the data builder had no reworded questions and scored 3 of 12 on the same probes; adding `[data] rewordings` is what lifted it. Your FAQ will need the same care.

In the browser (the generated `dist/demo.html`, Chromium, RTX 2060): rules answer in about 1 ms, the calculator in 2 ms, the model in 3 to 8 s per answer on WebGPU. With the CPU build, expect about twice as long or more.

## Limits

- The held-out set comes from the same generator as the training data. Fresh wording is harder: the reference bot answered about 8 of 12 unseen fact questions correctly with the right passage supplied.
- The model is over-cautious on unfamiliar wording and can half-remember a number. Keep one fact per FAQ entry, give each 3 or more phrasings, and never rely on it for anything where a wrong answer is costly.
- Search is keyword-based. A question that shares no words with an entry finds nothing; fix it with `aliases` and more `Q:` lines.
- Only SmolLM2-135M-Instruct has been tested. English only in practice.
- The 4-bit packing and the browser runtime were tested on Windows with Chromium-based browsers (WebGPU and WebAssembly). Safari, Firefox and phones were not tested.

## Repository layout

```
pocketbot.toml        the one config file
knowledge/            faq.txt, docs/*.md, probes.json   (sample: a fictional bakery)
scripts/              01_build_data .. 05_build_web, quant.py (the 4-bit rule), fetch_runtime.py, check_repo.py
web/                  pocketbot.js (runtime), demo.html
examples/proxy/       proxy.mjs, proxy.php  (keep API keys off the page)
tests/                4-bit packing, calculator and search, proxy
docs/                 the documents above, GIFs and images, make_docs_assets.py
```

Run the tests: `python -m unittest discover tests && node tests/test_runtime.mjs && node tests/test_proxy.mjs`. Before pushing: `python scripts/check_repo.py`.

## Developer

Built and maintained by **Muhammad Faizan** (faizzyhon), software and AI engineer.

| | |
|---|---|
| Website | [faizzyhon.com](https://faizzyhon.com) |
| Blog | [faizzyhon.com/blog](https://faizzyhon.com/blog/) |
| GitHub | [github.com/faizzyhon](https://github.com/faizzyhon) |
| LinkedIn | [linkedin.com/in/faizzyhon](https://www.linkedin.com/in/faizzyhon) |
| Email | support@faizzyhon.com |
| Hire me / custom bots | [faizzyhon.com/services.html](https://faizzyhon.com/services.html) · [contact](https://faizzyhon.com/contact.html) |

Bugs and ideas: open an issue on this repository. Related work: [rihla-llm](https://github.com/faizzyhon/rihla-llm) (multilingual domain LLM pipeline) and [Gale](https://github.com/faizzyhon/Gale) (download manager).

## Licence and credits

Code: MIT ([LICENSE](LICENSE)). Third-party components and the base model keep their own licences: [NOTICE.md](NOTICE.md). The sample bakery data is invented.
