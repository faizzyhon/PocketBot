"""Step 7. Assemble the static folder you deploy: dist/  (model, tokenizer, search index, bot config, browser runtime, demo page).
  python scripts/05_build_web.py        then:  python -m http.server 8000 -d dist     and open http://localhost:8000/demo.html
dist/ is plain static files: any host works (GitHub Pages, Netlify, Cloudflare Pages, an Apache folder). No server code runs."""
import json, shutil, sys
from common import BUILD, DIST, ROOT, config, load_docs, load_faq

C = config(); B, R = C['bot'], C['runtime']; PV = C.get('provider', {'type': 'local'})
if not (BUILD / 'model' / 'model_q4.onnx').exists(): raise SystemExit('build/model/model_q4.onnx not found. Run scripts/03_export.py first.')
if not (ROOT / 'web' / 'vendor' / 'lib' / 'ort.wasm.bundle.min.js').exists():
    sys.path.insert(0, str(ROOT / 'scripts')); import fetch_runtime; fetch_runtime.main()

# bot.config.json is served to every visitor. Refuse to publish anything that looks like a credential in it.
import re, os
for k, v in (PV.get('headers') or {}).items():
    if re.search(r'(?i)auth|key|token|secret', k) and os.environ.get('POCKETBOT_ALLOW_PUBLIC_KEY') != '1':
        raise SystemExit(f'[provider] headers.{k} looks like a credential and bot.config.json is public. Put the key in a proxy (examples/proxy/) instead.')
if PV.get('type') == 'http' and re.match(r'https://api\.(openai|anthropic|groq)\.', PV.get('endpoint', '')):
    print('warning: [provider] endpoint is a vendor API. Calling it directly from the browser needs a key in the page. Use a proxy (examples/proxy/) instead.')

if DIST.exists(): shutil.rmtree(DIST)
(DIST / 'model').mkdir(parents=True); shutil.copytree(ROOT / 'web' / 'vendor' / 'lib', DIST / 'lib')
for f in ('model_q4.onnx', 'tokenizer.json', 'tokenizer_config.json'): shutil.copy(BUILD / 'model' / f, DIST / 'model' / f)
for f in ('pocketbot.js', 'demo.html'): shutil.copy(ROOT / 'web' / f, DIST / f)

# search index: one passage per FAQ entry (all its phrasings in the heading, so any wording matches) + the document passages
chunks = [{'k': 'qa', 't': 'FAQ', 'h': ' / '.join(e['qs']), 'q': e['qs'][0], 'x': e['a'], 'w': e['w'], 'u': ''} for e in load_faq()]
chunks += [{'u': d['u'], 't': d['t'], 'h': d['h'], 'x': d['x']} for d in load_docs()]
(DIST / 'index.json').write_text(json.dumps({'chunks': chunks}, ensure_ascii=False, separators=(',', ':')), 'utf-8')

info = json.load(open(BUILD / 'model' / 'model_info.json'))
(DIST / 'bot.config.json').write_text(json.dumps({
    'name': B['name'], 'system_prompt': B['system_prompt'],
    'replies': {k: B[k + '_reply'] for k in ('identity', 'injection', 'harmful', 'unknown', 'offtopic', 'empty', 'greeting', 'thanks')},
    'model': {'version': C['model']['version'], **info}, 'runtime': R, 'provider': PV}, ensure_ascii=False, indent=1), 'utf-8')
size = sum(f.stat().st_size for f in DIST.rglob('*') if f.is_file()) / 1e6
print(f'dist/ ready: {len(chunks)} passages, {size:.0f} MB total (model {(DIST / "model" / "model_q4.onnx").stat().st_size / 1e6:.0f} MB).')
print('try it:  python -m http.server 8000 -d dist   ->   http://localhost:8000/demo.html')
