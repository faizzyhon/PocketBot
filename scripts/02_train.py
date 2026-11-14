"""Steps 2 and 3. Full fine-tune, then quantization-aware fine-tune of the base model on build/data/*.jsonl.
  python scripts/02_train.py --stage sft     base model        -> build/ckpt/sft
  python scripts/02_train.py --stage qat     build/ckpt/sft    -> build/ckpt/qat   (weights rounded to the 4-bit grid in every forward pass)
Options: --epochs N  --lr X  to override pocketbot.toml.   Needs a CUDA GPU (about 6 GB).
Only the assistant's words are learned (the prompt tokens are masked out of the loss). The best epoch by validation loss is kept."""
import argparse, json, math, random, time
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import BUILD, config
from quant import fq_torch

ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['sft', 'qat'], required=True)
ap.add_argument('--epochs', type=int); ap.add_argument('--lr', type=float); args = ap.parse_args()
C = config(); T = C['train']; Q = C['qat']; BASE = C['model']['base']
if not torch.cuda.is_available(): raise SystemExit('No CUDA GPU found. Training needs one (a free Colab T4 works).')
epochs = args.epochs or (T['epochs'] if args.stage == 'sft' else Q['epochs'])
lr = args.lr or (T['lr'] if args.stage == 'sft' else Q['lr'])
src = BASE if args.stage == 'sft' else str(BUILD / 'ckpt' / 'sft'); out = BUILD / 'ckpt' / args.stage
random.seed(1); torch.manual_seed(1)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).cuda(); model.config.use_cache = False

if args.stage == 'qat':
    BLK = Q['block']
    fq = lambda w: fq_torch(w, BLK)   # the rounding rule lives in quant.py, shared with the exporter
    for m in model.modules():
        if isinstance(m, nn.Linear): m.forward = (lambda x, m=m: F.linear(x, fq(m.weight), m.bias))
        elif isinstance(m, nn.Embedding): m.forward = (lambda ids, m=m: F.embedding(ids, fq(m.weight)))   # the embedding is tied to the output layer

def encode(row):
    m = row['messages']
    p = tok(tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)['input_ids']
    a = tok(m[-1]['content'] + '<|im_end|>\n', add_special_tokens=False)['input_ids']
    return (p + a)[:T['max_len']], ([-100] * len(p) + a)[:T['max_len']]
load = lambda f: [encode(json.loads(l)) for l in open(BUILD / 'data' / f, encoding='utf-8')]
train, val = load('train.jsonl'), load('val.jsonl')

def batches(data, shuffle):
    idx = list(range(len(data))); random.shuffle(idx) if shuffle else None
    for s in range(0, len(idx), T['batch']):
        b = [data[i] for i in idx[s:s + T['batch']]]; L = max(len(x) for x, _ in b)
        yield (torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in b]).cuda(), torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in b]).cuda(),
               torch.tensor([y + [-100] * (L - len(y)) for _, y in b]).cuda())

@torch.no_grad()
def val_loss():
    model.eval(); tot = n = 0
    for ids, att, lab in batches(val, False):
        with torch.autocast('cuda', dtype=torch.float16): loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        k = (lab[:, 1:] != -100).sum().item(); tot += loss.item() * k; n += k
    model.train(); return tot / n

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
steps = epochs * math.ceil(math.ceil(len(train) / T['batch']) / T['accum']); warm = max(3, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps)))))
scaler = torch.amp.GradScaler('cuda'); model.train(); best = val_loss(); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
print(f'[{args.stage}] train {len(train)}  val {len(val)}  epochs {epochs}  lr {lr}  val loss before: {best:.4f}', flush=True)
saved = False
for ep in range(epochs):
    run = cnt = 0
    for i, (ids, att, lab) in enumerate(batches(train, True)):
        with torch.autocast('cuda', dtype=torch.float16): loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / T['accum']
        scaler.scale(loss).backward(); run += loss.item() * T['accum']; cnt += 1
        if (i + 1) % T['accum'] == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
    vl = val_loss(); print(f'  epoch {ep + 1}/{epochs}  train loss {run / cnt:.4f}  val loss {vl:.4f}  ({time.time() - t0:.0f}s)', flush=True)
    if vl < best or not saved:
        best = min(best, vl); saved = True; model.config.use_cache = True
        model.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out); model.config.use_cache = False
print(f'[{args.stage}] done. best val loss {best:.4f}  ->  {out}')
