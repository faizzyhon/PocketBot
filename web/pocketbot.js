/* PocketBot browser runtime. A small fine-tuned language model (4-bit, ~90 MB) runs in the page on WebGPU or WebAssembly. No server, no API.
   Every question goes through: rules -> calculator -> keyword search -> model -> source link.
   Usage:
     import { load } from './pocketbot.js';
     const bot = await load(status => console.log(status));          // downloads the model once; the browser keeps it
     const r = await bot.answer('Are you open on Sunday?', partial => ui.show(partial));   // r.text, r.via ('rules' | 'calculator' | 'model'), r.stats
   Add ?nogpu to the page URL to force the CPU build. */
const BASE = new URL('./', import.meta.url);
const U = (p) => new URL(p, BASE).href;

/* ── rules: these never reach the model (a 135M model is not reliable enough to refuse or to keep a persona) ── */
const vendors = 'gemini|bard|google|alphabet|deepmind|openai|chatgpt|gpt|claude|anthropic|llama|meta|mistral|deepseek|grok|copilot|microsoft|qwen|palm|transformer|llm';
const identityRe = new RegExp("\\b(who|what)\\b.{0,25}\\b(are|r)\\b.{0,6}\\b(you|u)\\b|\\byour (name|model|version|creator|maker|developer|vendor|company|origin|architecture|backend|engine)\\b|\\b(which|what|whats|what s) (ai|model|llm|bot|engine|company|version|api)\\b|\\bwho (made|created|built|trained|developed|owns|runs|powers|programmed|designed)\\b|\\b(are|r) (you|u) (a |an |the )?(" + vendors + ")\\b|\\b(" + vendors + ")\\b.{0,30}\\b(are|r) (you|u)\\b|\\bpowered by\\b|\\bunderlying (model|ai|llm)\\b|\\bwhich (llm|ai|model)\\b", 'u');
const attackRe = new RegExp("\\b(ignore|disregard|forget|override|bypass|skip)\\b.{0,30}\\b(previous|prior|above|earlier|all|your|the|these)\\b.{0,20}\\b(instructions?|rules?|prompts?|guidelines?|restrictions?|constraints?)\\b|\\b(system|developer|hidden|initial|original|secret) (prompt|message|instructions?)\\b|\\b(reveal|show|print|repeat|leak|dump|output|display|tell me)\\b.{0,25}\\b(prompt|instructions?|rules|configuration|api key|secret|token|password)\\b|\\b(jailbreak|dan mode|developer mode|god mode|do anything now|sudo mode|admin mode)\\b|\\b(pretend|act|roleplay|role play|imagine|simulate)\\b.{0,25}\\b(you are|you re|to be|as|that you)\\b.{0,30}\\b(not|no longer|without|unfiltered|unrestricted|another|different|" + vendors + ")\\b|\\bnew (instructions?|rules?|persona|role)\\b|\\bfrom now on\\b.{0,30}\\b(you|your)\\b", 'u');
const harmRe = /\b(write|make|build|create|generate|code|develop|give me|show me how to (make|build|write))\b.{0,40}\b(ransomware|keylogger|virus|malware|botnet|ddos|trojan|spyware|rootkit|credential stealer|phishing (page|email|site)|fake login)\b|\bhow (do|can|to|would) (i |you |we )?(hack|steal|crack|break into|get into|take over|ddos|scam)\b/i;
const hiRe = /^(hi|hello|hey|hiya|good (morning|afternoon|evening))\b[\s!.,]*(there)?[\s!.,]*$/i, thanksRe = /^(thanks|thank you|thx|cheers)\b[\s!.,a-z]*$/i;
const norm = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}\s]+/gu, ' ').replace(/\s+/g, ' ').trim();

/* ── calculator (no eval): + - * / ^ and brackets. Exported for tests. ── */
export function calc(raw) {
  let s = raw.toLowerCase().trim().replace(/^(what\s*('?s|is)|calculate|compute|evaluate|solve|how much is|whats)\s+/, '');
  s = s.replace(/×/g, '*').replace(/÷/g, '/').replace(/−/g, '-').replace(/\*\*/g, '^').replace(/[ =?.\t]+$/, '');
  if (!s || s.length > 80 || !/^[0-9+\-*/^().\s]+$/.test(s) || !/\d/.test(s) || !/[+\-*/^]/.test(s.replace(/^[-+\s]+/, ''))) return null;
  const t = s.match(/\d+(?:\.\d+)?|[-+*/^()]/g) || []; if (t.join('') !== s.replace(/\s+/g, '')) return null;
  let p = 0, bad = null;
  const peek = () => t[p];
  const prim = () => { const x = peek(); if (x === '-') { p++; return -prim(); } if (x === '+') { p++; return prim(); }
    if (x === '(') { p++; const v = expr(); if (peek() !== ')') { bad = 'syn'; return 0; } p++; return v; }
    if (x !== undefined && /^\d/.test(x)) { p++; return parseFloat(x); } bad = 'syn'; return 0; };
  const pow = () => { const b = prim(); if (peek() === '^') { p++; const e = pow(); if (Math.abs(e) > 1000) { bad = 'big'; return 0; } return Math.pow(b, e); } return b; };
  const term = () => { let v = pow(); while (peek() === '*' || peek() === '/') { const o = t[p++], r = pow(); if (o === '/' && r === 0) { bad = 'div'; return 0; } v = o === '*' ? v * r : v / r; } return v; };
  const expr = () => { let v = term(); while (peek() === '+' || peek() === '-') { const o = t[p++], r = term(); v = o === '+' ? v + r : v - r; } return v; };
  const v = expr();
  if (bad === 'div') return 'Dividing by zero is undefined.';
  if (bad || p !== t.length || !isFinite(v)) return null;
  return `${s} = ${Math.abs(v) >= 1e15 || (v !== 0 && Math.abs(v) < 1e-9) ? v.toExponential(6) : String(+v.toFixed(10))}`;
}

/* ── keyword search (BM25). Words in the heading count triple. Exported for tests. ── */
const STOP = new Set('the and for you are was with that this from have has not but can how what when where why who which does did into about your his her its our their there then than too any all get got use used using tell give please explain mean means between difference vs is of to in on at by an or as be do it if me my we so a i'.split(' '));
const nrm = (s) => ' ' + s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim() + ' ';
const count = (hay, needle) => { let n = 0, i = hay.indexOf(needle); while (i !== -1) { n++; i = hay.indexOf(needle, i + needle.length); } return n; };
export function terms(q, alias = {}) {
  const words = []; for (const w of nrm(q).trim().split(' ')) { words.push(w); if (alias[w]) words.push(...alias[w].split(' ')); }
  const out = new Set();
  for (let t of words) { if (t.length < 2 || STOP.has(t)) continue; if (t.length > 3 && t.endsWith('s') && !t.endsWith('ss')) t = t.slice(0, -1); out.add(t); }
  return [...out];
}
export function retrieve(query, chunks, { max = 4, faqWeight = 1.1, minRel = 0.45, minCov = 0.4, aliases = {} } = {}) {
  const T = terms(query, aliases); if (!T.length || !chunks.length) return [];
  const docs = [], df = {}; let lenSum = 0;
  chunks.forEach((c, i) => {
    const body = c._b || (c._b = nrm(c.x)), head = c._h || (c._h = nrm(c.t + ' ' + c.h));
    const d = { len: Math.max(1, count(body, ' ') - 1), b: {}, h: {} };
    for (const t of T) { d.b[t] = count(body, ' ' + t); d.h[t] = count(head, ' ' + t); if (d.b[t] || d.h[t]) df[t] = (df[t] || 0) + 1; }
    docs[i] = d; lenSum += d.len;
  });
  const N = chunks.length, avg = lenSum / N, idf = {}; let idfSum = 0;
  for (const t of T) { idf[t] = Math.log(1 + (N - (df[t] || 0) + 0.5) / ((df[t] || 0) + 0.5)); if (df[t]) idfSum += idf[t]; }   // words found nowhere cannot match, so they do not count against coverage
  if (!idfSum) return [];
  const k1 = 1.4, b = 0.75, scored = [];
  docs.forEach((d, i) => {
    let s = 0, hit = 0;
    for (const t of T) { const tf = d.b[t] + 3 * d.h[t]; if (!tf) continue; hit += idf[t]; s += idf[t] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * d.len / avg)); }
    if (s <= 0 || hit / idfSum < minCov) return;
    const c = chunks[i]; scored.push([i, s * (c.w || 1) * (c.k === 'qa' ? faqWeight : 1), hit / idfSum]);
  });
  scored.sort((a, b) => b[1] - a[1]);
  const best = { qa: 0, pg: 0 }, take = { qa: 0, pg: 0 }, per = {}, out = [];
  for (const [i, s, cov] of scored) {   // two pools (FAQ entries, document passages) so neither crowds out the other
    const c = chunks[i], g = c.k === 'qa' ? 'qa' : 'pg';
    if (!best[g]) best[g] = s;
    if (s < minRel * best[g] || take[g] >= 3 || out.length >= max) continue;
    if (g === 'pg' && (per[c.t + c.u] || 0) >= 2) continue;
    per[c.t + c.u] = (per[c.t + c.u] || 0) + 1; take[g]++; out.push({ c, s, cov });
  }
  return out.sort((a, b) => b.s - a.s);
}
const passage = (c) => c.k === 'qa' ? `Q: ${c.q || c.h}\nA: ${c.x}` : c.x;

/* ── model loading and generation ── */
async function cachedBuffer(cacheName, url, onProgress) {
  let cache = null; try { cache = await caches.open(cacheName); } catch (e) { /* private window: download every time */ }
  const hit = cache && await cache.match(url);
  if (hit) { onProgress && onProgress(1, true); return hit.arrayBuffer(); }
  const res = await fetch(url); if (!res.ok) throw new Error('model download failed: ' + res.status);
  const total = +res.headers.get('content-length') || 0, reader = res.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onProgress && onProgress(total ? got / total : 0, false); }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) { try { await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) { /* quota */ } }
  return buf.buffer;
}

/* The in-browser model: tokenizer + ONNX Runtime session + greedy generation. Created on first use, so a bot configured for an HTTP provider
   never downloads the 89 MB file unless it needs the fallback. */
async function initLocal(cfg, status) {
  const M = cfg.model, RT = cfg.runtime;
  const T = await import(U('lib/transformers.web.min.js'));
  let ort, sess, backend = 'cpu';
  async function start(gpu, model) {   // WebGPU (graphics card) when available, else WebAssembly on the CPU: two separate ONNX Runtime builds
    const o = await import(U(gpu ? 'lib/ort.webgpu.bundle.min.js' : 'lib/ort.wasm.bundle.min.js'));
    o.env.wasm.wasmPaths = gpu ? { mjs: U('lib/ort-wasm-simd-threaded.asyncify.js'), wasm: U('lib/ort-wasm-simd-threaded.asyncify.wasm') }
                               : { mjs: U('lib/ort-wasm-simd-threaded.js'), wasm: U('lib/ort-wasm-simd-threaded.wasm') };
    o.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    const opts = { executionProviders: gpu ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all' };
    if (gpu) {   // keep the attention cache in GPU memory between tokens
      opts.preferredOutputLocation = { logits: 'cpu' };
      for (let i = 0; i < M.layers; i++) { opts.preferredOutputLocation[`present.${i}.key`] = 'gpu-buffer'; opts.preferredOutputLocation[`present.${i}.value`] = 'gpu-buffer'; }
    }
    const s = await o.InferenceSession.create(model, opts); ort = o; sess = s; backend = gpu ? 'webgpu' : 'cpu';
  }
  const [tj, tc] = await Promise.all([fetch(U('model/tokenizer.json')).then(r => r.json()), fetch(U('model/tokenizer_config.json')).then(r => r.json())]);
  const tok = new T.PreTrainedTokenizer(tj, tc);
  const buf = await cachedBuffer('pocketbot-v' + M.version, U('model/model_q4.onnx') + '?v=' + M.version, (f, cached) => status(cached ? 'Loading the saved model…' : `Downloading the model… ${Math.round(f * 100)}%`));
  status('Starting the model…');
  const wantGpu = !!navigator.gpu && !/[?&]nogpu\b/.test(location.search) && await navigator.gpu.requestAdapter().then(a => !!a, () => false);
  if (wantGpu) { try { await start(true, buf); } catch (e) { console.warn('[pocketbot] WebGPU unavailable, using the CPU', e); sess = null; } }
  if (!sess) await start(false, buf);
  const L = M.layers, KV = M.kv_heads, HD = M.head_dim, enc = (t) => tok.encode(t, { add_special_tokens: false })[0], stops = new Set([enc('<|im_end|>'), enc('<|endoftext|>')]);
  const state = { aborted: false, stats: null };

  async function generate(prompt, onToken, maxNew = RT.max_new_tokens) {
    const ids = Array.from(tok.encode(prompt, { add_special_tokens: false }));
    let past = {}; for (let i = 0; i < L; i++) for (const k of ['key', 'value']) past[`past_key_values.${i}.${k}`] = new ort.Tensor('float32', new Float32Array(0), [1, KV, 0, HD]);
    let cur = ids, seen = 0; const out = []; state.aborted = false; const t0 = performance.now(); let tFirst = 0;
    for (let step = 0; step < maxNew && !state.aborted; step++) {
      const n = cur.length;
      const feed = { input_ids: new ort.Tensor('int64', BigInt64Array.from(cur, BigInt), [1, n]),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(seen + n).fill(1n), [1, seen + n]),
        position_ids: new ort.Tensor('int64', BigInt64Array.from({ length: n }, (_, i) => BigInt(seen + i)), [1, n]), ...past };
      const res = await sess.run(feed); if (step === 0) tFirst = performance.now() - t0;
      const d = res.logits.data, V = res.logits.dims[res.logits.dims.length - 1]; let best = 0, bv = -Infinity;   // greedy: the most likely next token
      for (let i = d.length - V, j = 0; j < V; i++, j++) if (d[i] > bv) { bv = d[i]; best = j; }
      if (stops.has(best)) break;
      out.push(best); seen += n; cur = [best];
      for (const k in past) if (past[k].location === 'gpu-buffer') past[k].dispose();
      past = {}; for (let i = 0; i < L; i++) for (const k of ['key', 'value']) past[`past_key_values.${i}.${k}`] = res[`present.${i}.${k}`];
      const so = tok.decode(out, { skip_special_tokens: true });
      if (/\n(Question|Context):/.test(so)) break;   // it started echoing the prompt format
      if (onToken) onToken(so);
    }
    state.stats = { backend, promptTokens: ids.length, prefillMs: Math.round(tFirst), perTokenMs: out.length > 1 ? Math.round((performance.now() - t0 - tFirst) / (out.length - 1)) : 0, newTokens: out.length };
    let txt = tok.decode(out, { skip_special_tokens: true }).replace(/\n+(Question|Context):[\s\S]*$/, '').trim();
    if (out.length >= maxNew) { const e = Math.max(txt.lastIndexOf('. '), txt.lastIndexOf('! '), txt.lastIndexOf('? '), txt.search(/[.!?]$/)); if (e > 40) txt = txt.slice(0, e + 1); }
    return txt;
  }
  if (backend === 'webgpu') {   // warm-up compiles the GPU shaders and proves the GPU path runs; if it does not, fall back to the CPU
    try { await generate('<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n', null, 2); }
    catch (e) { console.warn('[pocketbot] WebGPU run failed, using the CPU', e); await start(false, buf); }
  }
  return { generate, state, get backend() { return backend; } };
}

/* An HTTP provider: any OpenAI-compatible /v1/chat/completions endpoint (OpenAI, Ollama, LM Studio, llama.cpp server, Groq, ...) or your own proxy.
   Never put a secret API key in this file or in bot.config.json: they are public. Use a proxy that holds the key (examples/proxy/). */
async function callHttp(P, messages, signal) {
  const res = await fetch(P.endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json', ...(P.headers || {}) }, signal,
    body: JSON.stringify({ model: P.model, messages, max_tokens: P.max_tokens || 300, temperature: P.temperature ?? 0.2, stream: false }) });
  if (!res.ok) throw new Error(`provider returned ${res.status}`);
  const j = await res.json();
  const text = j.choices?.[0]?.message?.content ?? j.content?.[0]?.text ?? j.reply ?? '';   // OpenAI shape, Anthropic shape, or the {reply} shape of examples/proxy
  if (!String(text).trim()) throw new Error('provider returned no text');
  return String(text).trim();
}

export async function load(status = () => {}) {
  status('Loading…');
  const cfg = await fetch(U('bot.config.json')).then(r => r.json()), RT = cfg.runtime, RP = cfg.replies, P = cfg.provider || { type: 'local' };
  const index = await fetch(U('index.json')).then(r => r.json());
  let local = null, localObj = null;
  const getLocal = () => local || (local = initLocal(cfg, status).then(l => { status(''); localObj = l; return l; }));
  if (P.type !== 'http') await getLocal();   // an HTTP-only bot never downloads the model; with fallback it is fetched the first time it is needed
  let lastStats = null, via = P.type === 'http' ? 'http' : 'model', ctrl = null;

  async function answer(question, onToken) {
    const q = question.trim().slice(0, 600), n = norm(q);
    if (attackRe.test(n)) return { text: RP.injection, via: 'rules' };
    if (identityRe.test(n)) return { text: RP.identity, via: 'rules' };
    if (harmRe.test(q)) return { text: RP.harmful, via: 'rules' };
    if (hiRe.test(q)) return { text: RP.greeting, via: 'rules' };
    if (thanksRe.test(q)) return { text: RP.thanks, via: 'rules' };
    const c = calc(q); if (c) return { text: c, via: 'calculator' };
    const hits = P.type === 'http' && P.use_search === false ? [] : retrieve(q, index.chunks, { max: RT.retrieval_top, faqWeight: RT.faq_weight, aliases: RT.aliases || {} });
    let ctx = ''; for (const h of hits) { const p = passage(h.c); if ((ctx + p).length > RT.context_chars) break; ctx += (ctx ? '\n\n' : '') + p; }
    const user = ctx ? `Context:\n${ctx}\n\nQuestion: ${q}` : q;
    let text = '', used = 'model';
    if (P.type === 'http') {
      ctrl = new AbortController(); const timer = setTimeout(() => ctrl.abort(), (P.timeout_s || 30) * 1000);
      try { text = await callHttp(P, [{ role: 'system', content: cfg.system_prompt }, { role: 'user', content: user }], ctrl.signal); used = 'http'; if (onToken) onToken(text); }
      catch (e) { if (!P.fallback_to_local) throw e; console.warn('[pocketbot] provider failed, using the local model:', e.message || e); status('Switching to the offline model…'); }
      finally { clearTimeout(timer); }
    }
    if (!text) {
      const l = await getLocal();
      text = await l.generate(`<|im_start|>system\n${cfg.system_prompt}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n`, onToken); lastStats = l.state.stats; used = 'model';
    }
    if (!text) text = RP.empty;
    const refused = text.startsWith(RP.unknown.slice(0, 30));
    if (used === 'model' && refused && hits.length && hits[0].cov >= 0.8) {   // the small model is sometimes over-cautious: if search found a strong match, show it instead of a refusal
      const top = hits[0].c, body = top.x.replace(/\s+/g, ' ').trim(), cut = body.slice(0, 340), end = Math.max(cut.lastIndexOf('. '), cut.lastIndexOf('? '));
      text = body.length > 340 ? (end > 120 ? cut.slice(0, end + 1) : cut + '…') : body;
    }
    const page = hits.find(h => h.c.u);
    if (page && !/\]\(/.test(text)) text += `\n\nRelated: [${page.c.t}](${page.c.u})`;   // give the reader somewhere to check the answer
    via = used;
    return { text, via: used, stats: used === 'model' ? lastStats : null, sources: hits.map(h => h.c.u || 'faq') };
  }
  status('');
  return { answer, abort: () => { if (localObj) localObj.state.aborted = true; ctrl && ctrl.abort(); },
           get backend() { return localObj ? localObj.backend : 'http'; }, get stats() { return lastStats; }, get provider() { return via; }, name: cfg.name };
}
if (typeof window !== 'undefined') window.PocketBot = { load };
