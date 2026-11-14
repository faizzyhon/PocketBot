# Making it your own

## What to edit

| File | What it controls |
|---|---|
| `pocketbot.toml` `[bot]` | the name, the system prompt, and the eight fixed replies (identity, injection, harmful, unknown, off-topic, empty, greeting, thanks) |
| `knowledge/faq.txt` | the facts the bot knows, in the voice you want |
| `knowledge/docs/*.md` | longer reference text, searched at question time (not used to write training questions) |
| `knowledge/probes.json` | your own test questions |
| `pocketbot.toml` `[runtime] aliases` | words that should also search for other words |
| `web/demo.html` | the chat page. Replace it with your own UI; `web/pocketbot.js` is the part to reuse. |

## Writing a good FAQ

- **One fact per entry.** "Delivery is $4, free over $40." is one entry. Do not bundle hours, prices and policies in one answer.
- **3 or more `Q:` lines per entry, in different words.** These are the training questions and the search vocabulary. Write them the way real people type: short, lower case, with the words you would not use yourself.
- **Answers are what the bot says.** Write them in the voice you want, in one to three sentences. Keep numbers and names exact; the model copies them from the passage it is given.
- **Never put two answers that disagree in the knowledge base.** The model will learn both.
- **Add the refusals you need.** Put questions you do not want guessed into `[data] unknown_questions` (the bot should answer "I don't see that").
- **Keep the system prompt short.** Small models follow a sentence or two better than a page.

## Choosing aliases

Run the bot, ask ten questions the way a stranger would, and note which ones fail to find the right entry. For each, check which word did not match and add it: `fee = "price cost"`. Aliases only affect search, so changing them needs `05_build_web.py` and nothing else.

## Fixed replies and what they do to training

The eight replies are trained into the model as well as returned by the rules. If you change them, retrain; otherwise the model and the rules will answer the same question in different words.

## Hosting

`dist/` is static. Requirements for the host:

- Serve `.wasm` as `application/wasm`. The pipeline renames the JavaScript modules to `.js` so no special `.mjs` rule is needed.
- Serve files over HTTPS (browsers expose WebGPU and Cache Storage only on secure origins; `localhost` counts as secure).
- If you send a `Content-Security-Policy` header, `script-src` must allow `'self'` and `'wasm-unsafe-eval'`, and `connect-src` must allow `'self'`.
- Do not gzip the `.onnx` file on the fly (it does not compress); caching headers on `/model/` are worth setting.
- Cross-origin isolation (COOP/COEP headers) lets the CPU build use several threads, but it breaks most embedded third-party content, so only set it on a page you control completely.

## Using a different base model

Only `HuggingFaceTB/SmolLM2-135M-Instruct` has been run through this pipeline. The export script assumes a Llama-style architecture with tied embeddings and a hidden size divisible by the block size (32). SmolLM2-360M-Instruct fits those rules and should work, at about 220 MB after packing and slower generation; it has not been tested here. Models with untied embeddings, or other tensor names, need changes in `03_export.py`.

## Reusing only the browser runtime

`web/pocketbot.js` does not depend on the Python side except through four files: `bot.config.json`, `index.json`, `model/*`, and the `lib/` folder. Any ONNX model exported with the same input and output names (`input_ids`, `attention_mask`, `position_ids`, `past_key_values.N.{key,value}` in; `logits`, `present.N.{key,value}` out) will run.
