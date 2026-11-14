# Connecting it to external APIs or another offline model

The part you can swap is **who writes the sentence**. Everything around it stays: the rules, the calculator and the search run in the browser first, and the chosen provider receives the same prompt (system prompt + the retrieved passages + the question).

```
question -> rules -> calculator -> search -> [ provider ] -> answer
                                              |
                                              +-- local   the 4-bit model in the tab (default)
                                              +-- http    any OpenAI-compatible endpoint, or your own proxy
                                              +-- http + fallback_to_local   try the endpoint, use the tab's model if it fails
```

Set it in `pocketbot.toml`, then run `python scripts/05_build_web.py`. No retraining is needed for any of the options below, but a different provider will not know the exact wording the built-in model was trained on, so it is told the same thing in the system prompt and the passages.

## What leaves the browser

| Provider | Sent over the network | Works offline |
|---|---|---|
| `local` | nothing, after the page and the model are cached | yes |
| `http` to a server on the visitor's own machine (Ollama, LM Studio) | the question and the passages, to `localhost` | yes, if that server is running |
| `http` to your proxy and a cloud API | the question and the passages, to your proxy and then to the vendor | no |

If your questions or knowledge are private, read the vendor's data terms before using a cloud API.

## 1. A local server you run yourself (Ollama, LM Studio, llama.cpp)

These expose an OpenAI-compatible endpoint, so the `http` provider works without a proxy and without a key.

```toml
[provider]
type = "http"
endpoint = "http://localhost:11434/v1/chat/completions"   # Ollama. LM Studio: http://localhost:1234/v1/chat/completions
model = "llama3.2"                                         # whatever `ollama list` shows
fallback_to_local = true
```

- This only works for people who run that server on their own computer, so it suits you while developing, or a tool you hand to technical users. A public website's visitors do not have it.
- The server must allow your page's origin. For Ollama set `OLLAMA_ORIGINS` (for example `OLLAMA_ORIGINS=http://localhost:8000`) before starting it; LM Studio has a CORS switch in its server settings. Check the current docs of the tool you use, these settings change between versions.
- A page served over HTTPS may be blocked by the browser from calling a local server, or may show a permission prompt. Test with the plain `http://localhost` demo first.

## 2. A cloud API (OpenAI, Anthropic, Groq and similar), through a proxy

Never put an API key in the page, in `pocketbot.toml`, or in `bot.config.json`: everything in `dist/` is public, and anyone can copy a key from it and spend your money. `05_build_web.py` refuses to build if a `headers` entry looks like a credential.
The pattern is: the browser calls **your** proxy, the proxy holds the key and calls the vendor.

```toml
[provider]
type = "http"
endpoint = "https://your-site.example/proxy.php"     # or http://localhost:8787/chat for the Node proxy
model = "ignored-by-the-proxy-it-sets-its-own"
fallback_to_local = true
```

Two ready-made proxies are in `examples/proxy/`:

| File | For | Run |
|---|---|---|
| `proxy.mjs` | any machine with Node 18+ | `ALLOWED_ORIGINS=https://your-site.example API_KEY=... UPSTREAM=openai MODEL=gpt-4o-mini node examples/proxy/proxy.mjs` (`UPSTREAM=anthropic` for Anthropic's API; set `MODEL` to a model you have access to) |
| `proxy.php` | shared PHP hosting (Hostinger, cPanel) | upload it, create `proxy-secrets.php` next to it (format in the file header) and keep that file out of the web root or behind an `.htaccess` block |

What both proxies do: refuse requests whose `Origin` is not on your list, limit requests per IP, cap `max_tokens`, forward the last few messages only, and return `{reply, choices:[{message:{content}}]}` so `pocketbot.js` can read it. `tests/test_proxy.mjs` checks the Node version against a fake upstream (it does not call a real vendor).
They are examples. Before production use, set a spending limit at your API vendor, add authentication if the endpoint is not meant to be public, and log failures.

### Your own backend

Anything that accepts `POST` JSON `{model, messages:[{role, content}], max_tokens, temperature, stream:false}` and returns one of these works:

```json
{"choices":[{"message":{"content":"the answer"}}]}      // OpenAI shape
{"content":[{"text":"the answer"}]}                     // Anthropic shape
{"reply":"the answer"}                                  // the shape the example proxies add
```

CORS: respond to the browser's `OPTIONS` preflight and send `Access-Control-Allow-Origin` for your site only.

## 3. Keep the built-in model as a safety net

`fallback_to_local = true` (the default in the sample config) means: if the endpoint errors, times out (`timeout_s`) or returns nothing, the runtime downloads and uses the 4-bit model instead. An HTTP-only bot never downloads the 89 MB file unless it needs the fallback. Set `fallback_to_local = false` to see provider errors instead.

## 4. Another offline model

| Goal | How |
|---|---|
| A bigger model on the visitor's machine | Run it in Ollama or LM Studio (section 1). |
| A bigger in-browser model | Change `[model] base` to a larger Llama-style model with tied embeddings and a hidden size divisible by 32 (SmolLM2-360M-Instruct fits the rules, about 220 MB packed; **not tested** here) and run the whole pipeline. |
| A different in-browser runtime (WebLLM, transformers.js text generation, llama.cpp WASM) | Not integrated. The seam is `initLocal()` in `web/pocketbot.js`: return an object with `generate(prompt, onToken)` and the rest of the runtime works unchanged. The prompt is plain ChatML text. |
| Your own ONNX model | It must use the input/output names in `CUSTOMIZING.md` ("Reusing only the browser runtime"). |

## Testing a provider

1. Build: `python scripts/05_build_web.py`, serve: `python -m http.server 8000 -d dist`, open `demo.html`.
2. Ask a question your FAQ answers. The line under the title shows `http` or `model` for each answer, so you can see which provider answered.
3. Stop the endpoint and ask again: with `fallback_to_local = true` the answer should still arrive (slower, from the built-in model).
4. Open the browser's network tab: with the proxy, no request should go to the vendor's domain and no request should carry a key.
