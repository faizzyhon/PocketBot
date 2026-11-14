"""Step 6. Test the model on questions it has not seen (knowledge/probes.json), and optionally measure next-token accuracy.
  python scripts/04_evaluate.py                       # the exported 4-bit ONNX model (what ships)
  python scripts/04_evaluate.py --ckpt qat            # a PyTorch checkpoint (sft or qat) instead
  python scripts/04_evaluate.py --show                # print every answer
  python scripts/03_export.py --all-logits && python scripts/04_evaluate.py --token-accuracy
What it measures: with the right passage handed to the model ("gold" context), does it answer, refuse or stay in character correctly?
It tests the MODEL, not retrieval. Question wording must not appear in faq.txt or the score is meaningless. Read the results as a smoke test, not a benchmark."""
import argparse, json, random
import numpy as np
from transformers import AutoTokenizer
from common import BUILD, ROOT, config, faq_passage, load_faq

ap = argparse.ArgumentParser(); ap.add_argument('--model', default=str(BUILD / 'model' / 'model_q4.onnx')); ap.add_argument('--ckpt')
ap.add_argument('--show', action='store_true'); ap.add_argument('--token-accuracy', action='store_true'); args = ap.parse_args()
C = config(); B = C['bot']; tok = AutoTokenizer.from_pretrained(C['model']['base']); faq = load_faq(); random.seed(3)
probes = json.load(open(ROOT / 'knowledge' / 'probes.json', encoding='utf-8'))
prompt = lambda user: tok.apply_chat_template([{'role': 'system', 'content': B['system_prompt']}, {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)

if args.ckpt:
    import torch
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(BUILD / 'ckpt' / args.ckpt, dtype=torch.float16).cuda().eval()
    def generate(text):
        ids = tok(prompt(text), add_special_tokens=False, return_tensors='pt').to('cuda')
        with torch.no_grad(): g = hf.generate(**ids, max_new_tokens=100, do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(g[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
else:
    import onnxruntime as ort
    info = json.load(open(BUILD / 'model' / 'model_info.json')); sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    end = tok.convert_tokens_to_ids('<|im_end|>')
    def generate(text):   # greedy decoding with a KV cache: the same loop the browser runs
        cur, out, n = tok(prompt(text), add_special_tokens=False)['input_ids'], [], 0
        past = {f'past_key_values.{i}.{k}': np.zeros((1, info['kv_heads'], 0, info['head_dim']), np.float32) for i in range(info['layers']) for k in ('key', 'value')}
        for _ in range(100):
            feed = {'input_ids': np.array([cur], np.int64), 'attention_mask': np.ones((1, n + len(cur)), np.int64), 'position_ids': np.arange(n, n + len(cur), dtype=np.int64)[None], **past}
            res = sess.run(None, feed); nxt = int(res[0][0, -1].argmax())
            if nxt in (tok.eos_token_id, end): break
            out.append(nxt); n += len(cur); cur = [nxt]
            past = {f'past_key_values.{i}.{k}': res[1 + 2 * i + (k == 'value')] for i in range(info['layers']) for k in ('key', 'value')}
        return tok.decode(out, skip_special_tokens=True).strip()

def context(gold):
    e = next((e for e in faq if any(gold.lower() in q.lower() for q in e['qs'])), None)
    if e is None: raise SystemExit(f'probes.json: no FAQ question contains "{gold}"')
    parts = [faq_passage(e)] + [faq_passage(x) for x in random.sample([x for x in faq if x is not e], 1)]; random.shuffle(parts)
    return '\n\n'.join(parts)

same = lambda out, key: out[:30].lower() == B[key][:30].lower()
results, rows = {}, []
for kind in ('fact', 'unknown', 'refuse', 'injection', 'identity'):
    ok = 0
    for p in probes[kind]:
        user = f"Context:\n{context(p['gold'])}\n\nQuestion: {p['q']}" if 'gold' in p else p['q']; out = generate(user)
        good = (any(x.lower() in out.lower() for x in p['expect']) and not same(out, 'unknown_reply')) if kind == 'fact' else same(out, {'unknown': 'unknown_reply', 'refuse': 'harmful_reply', 'injection': 'injection_reply', 'identity': 'identity_reply'}[kind])
        ok += good; rows.append((kind, good, p['q'], out))
    results[kind] = f'{ok}/{len(probes[kind])}'
print('probe results:', results)
if args.show:
    for kind, good, q, out in rows: print(f"  {'ok ' if good else 'BAD'} [{kind}] {q}\n        -> {out[:160]!r}")
print('Note: exact-prefix checks for refusals mean a correct refusal in different words counts as a miss.')

if args.token_accuracy:
    allp = BUILD / 'model' / 'model_q4_all_logits.onnx'
    if not allp.exists(): raise SystemExit('run: python scripts/03_export.py --all-logits')
    info = json.load(open(BUILD / 'model' / 'model_info.json')); s = ort.InferenceSession(str(allp), providers=['CPUExecutionProvider']); hit = tot = 0
    for line in list(open(BUILD / 'data' / 'val.jsonl', encoding='utf-8'))[:60]:
        m = json.loads(line)['messages']
        p = tok(tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)['input_ids']
        a = tok(m[-1]['content'] + '<|im_end|>', add_special_tokens=False)['input_ids']; ids = (p + a)[:640]
        feed = {'input_ids': np.array([ids], np.int64), 'attention_mask': np.ones((1, len(ids)), np.int64), 'position_ids': np.arange(len(ids), dtype=np.int64)[None],
                **{f'past_key_values.{i}.{k}': np.zeros((1, info['kv_heads'], 0, info['head_dim']), np.float32) for i in range(info['layers']) for k in ('key', 'value')}}
        pred = s.run(['logits'], feed)[0][0].argmax(-1)
        for j in range(len(p) - 1, len(ids) - 1): hit += int(pred[j] == ids[j + 1]); tot += 1
    print(f'next-token accuracy on held-out examples: {hit / tot:.4f} ({tot} tokens). The held-out set comes from the same generator as the training data, so this is optimistic.')
