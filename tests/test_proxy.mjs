// node tests/test_proxy.mjs : the proxy forwards to a fake upstream, keeps the key server-side, and enforces origin and rate limits.
import assert from 'node:assert/strict';
process.env.ALLOWED_ORIGINS = 'https://good.example'; process.env.API_KEY = 'SECRET-123'; /* allow-secret: fake key for this test */ process.env.RATE_PER_MIN = '3'; process.env.MODEL = 'm1';
const { createServer } = await import('../examples/proxy/proxy.mjs');

let seen = null;
const fakeUpstream = async (url, init) => { seen = { url, headers: init.headers, body: JSON.parse(init.body) };
  return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: ' Hello from upstream ' } }] }) }; };
const srv = createServer(fakeUpstream); await new Promise(r => srv.listen(0, r)); const base = `http://127.0.0.1:${srv.address().port}`;
const post = (body, origin = 'https://good.example') => fetch(base, { method: 'POST', headers: { 'content-type': 'application/json', ...(origin ? { origin } : {}) }, body: JSON.stringify(body) });

let r = await post({ messages: [{ role: 'system', content: 'be brief' }, { role: 'user', content: 'hi' }], max_tokens: 9999 });
assert.equal(r.status, 200); const j = await r.json();
assert.equal(j.reply, 'Hello from upstream'); assert.equal(j.choices[0].message.content, 'Hello from upstream');
assert.equal(seen.headers.authorization, 'Bearer SECRET-123');          // key added on the server
assert.equal(seen.body.max_tokens, 400);                                 // page cannot ask for more than the cap
assert.equal(seen.body.model, 'm1');
assert.equal(r.headers.get('access-control-allow-origin'), 'https://good.example');
assert.equal((await post({ messages: [{ role: 'user', content: 'x' }] }, 'https://evil.example')).status, 403);   // other sites are refused
assert.equal((await post({ messages: [{ role: 'user', content: 'x' }] }, '')).status, 403);                        // no Origin header: refused
assert.equal((await post({ messages: [] })).status, 400);
assert.equal((await post({ messages: [{ role: 'user', content: 'x' }] })).status, 200);                            // 3rd counted request (limit 3 per minute)
assert.equal((await post({ messages: [{ role: 'user', content: 'x' }] })).status, 429);                            // 4th is refused
srv.close(); console.log('proxy tests passed');
