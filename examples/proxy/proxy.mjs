// A tiny proxy that keeps your API key off the web page.  node examples/proxy/proxy.mjs
//   browser (pocketbot.js, provider.type = "http")  ->  this proxy (holds the key)  ->  OpenAI-compatible API or Anthropic
//
// Environment variables:
//   UPSTREAM          openai | anthropic              (default openai)
//   UPSTREAM_URL      override the API URL            (default: the vendor's chat endpoint)
//   API_KEY           the secret key. Never put it in the repo or in the page.
//   MODEL             model name sent upstream        (default gpt-4o-mini for openai; set it for anthropic)
//   ALLOWED_ORIGINS   comma list, e.g. https://example.com,http://localhost:8000   (required: requests from other sites get 403)
//   RATE_PER_MIN      requests per IP per minute      (default 20)
//   PORT              default 8787
// Point provider.endpoint in pocketbot.toml at  http://localhost:8787/chat  (or wherever you host this).
// This is an example, not a hardened service: add real authentication, logging and a spending limit at your API vendor before using it in production.
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const E = process.env, UP = E.UPSTREAM || 'openai', PORT = +E.PORT || 8787, RATE = +E.RATE_PER_MIN || 20;
const URLS = { openai: 'https://api.openai.com/v1/chat/completions', anthropic: 'https://api.anthropic.com/v1/messages' };
const upstreamUrl = E.UPSTREAM_URL || URLS[UP], model = E.MODEL || (UP === 'openai' ? 'gpt-4o-mini' : '');
const allowed = (E.ALLOWED_ORIGINS || '').split(',').map(s => s.trim()).filter(Boolean);
const hits = new Map();   // ip -> [timestamps]

function limited(ip) {
  const now = Date.now(), list = (hits.get(ip) || []).filter(t => now - t < 60000); list.push(now); hits.set(ip, list); return list.length > RATE;
}
function reply(res, code, body, origin) {
  res.writeHead(code, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': origin || 'null', 'Vary': 'Origin', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Methods': 'POST, OPTIONS' });
  res.end(JSON.stringify(body));
}

export function createServer(fetchImpl = fetch) {
  return http.createServer(async (req, res) => {
    const origin = req.headers.origin || '';
    if (!allowed.includes(origin)) return reply(res, 403, { error: 'origin not allowed' });
    if (req.method === 'OPTIONS') return reply(res, 204, {}, origin);
    if (req.method !== 'POST') return reply(res, 405, { error: 'POST only' }, origin);
    if (limited(req.socket.remoteAddress)) return reply(res, 429, { error: 'slow down' }, origin);
    let raw = ''; for await (const c of req) { raw += c; if (raw.length > 20000) return reply(res, 413, { error: 'too large' }, origin); }
    let body; try { body = JSON.parse(raw); } catch { return reply(res, 400, { error: 'bad json' }, origin); }
    const msgs = Array.isArray(body.messages) ? body.messages.filter(m => m && typeof m.content === 'string' && ['system', 'user', 'assistant'].includes(m.role)).slice(-8) : [];
    if (!msgs.length) return reply(res, 400, { error: 'no messages' }, origin);
    const max = Math.min(+body.max_tokens || 300, 400);   // the page may ask for less, never more than this
    try {
      let up;
      if (UP === 'anthropic') {
        const system = msgs.filter(m => m.role === 'system').map(m => m.content).join('\n'), rest = msgs.filter(m => m.role !== 'system');
        up = await fetchImpl(upstreamUrl, { method: 'POST', headers: { 'content-type': 'application/json', 'x-api-key': E.API_KEY || '', 'anthropic-version': '2023-06-01' },
          body: JSON.stringify({ model, max_tokens: max, system, messages: rest }) });
      } else {
        up = await fetchImpl(upstreamUrl, { method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer ' + (E.API_KEY || '') },
          body: JSON.stringify({ model, max_tokens: max, temperature: 0.2, messages: msgs }) });
      }
      if (!up.ok) return reply(res, 502, { error: 'upstream ' + up.status }, origin);
      const j = await up.json(); const text = j.choices?.[0]?.message?.content ?? j.content?.[0]?.text ?? '';
      reply(res, 200, { reply: String(text).trim(), choices: [{ message: { content: String(text).trim() } }] }, origin);   // both shapes: pocketbot.js reads either
    } catch (e) { reply(res, 502, { error: 'upstream unreachable' }, origin); }
  });
}
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {   // only when run directly, not when imported by a test
  if (!allowed.length) { console.error('Set ALLOWED_ORIGINS (for example https://example.com).'); process.exit(1); }
  if (!E.API_KEY && !E.UPSTREAM_URL) console.warn('warning: API_KEY is not set; the upstream will reject requests.');
  createServer().listen(PORT, () => console.log(`proxy on :${PORT} -> ${upstreamUrl} (origins: ${allowed.join(', ')})`));
}
