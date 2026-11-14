"""Step 4 and 5. build/ckpt/qat -> ONNX (fp32) -> one 4-bit ONNX file the browser can run.
  python scripts/03_export.py            -> build/model/model_q4.onnx  (+ tokenizer files, model_info.json)
  python scripts/03_export.py --all-logits   keeps logits for every position (only needed by 04_evaluate.py --token-accuracy)

Packing: every MatMul weight becomes 4-bit (block size from pocketbot.toml, zero point 8). The embedding table is packed ONCE and used twice:
by the token lookup (Gather + unpack) and by the output layer (MatMulNBits), because the base model ties them together. The same rounding rule
is used by 02_train.py --stage qat, so the model was trained on exactly the grid it is stored on."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np, onnx
from onnx import TensorProto as T, helper, numpy_helper
from common import BUILD, config
from quant import pack4

ap = argparse.ArgumentParser(); ap.add_argument('--all-logits', action='store_true'); ap.add_argument('--ckpt', default='qat'); args = ap.parse_args()
C = config(); BLK = C['qat']['block']; ACC = C['export']['accuracy_level']
ckpt = BUILD / 'ckpt' / args.ckpt; fp32 = BUILD / 'onnx-fp32'; out = BUILD / 'model'; out.mkdir(parents=True, exist_ok=True)
if not (ckpt / 'config.json').exists(): raise SystemExit(f'{ckpt} not found. Run 02_train.py first (stage sft, then qat).')

# 1. PyTorch -> ONNX with a KV cache (Hugging Face optimum; it checks the result against PyTorch and prints the max difference)
cli = Path(sys.executable).parent / ('optimum-cli.exe' if os.name == 'nt' else 'optimum-cli')
if not (fp32 / 'model.onnx').exists() or (fp32 / 'model.onnx').stat().st_mtime < (ckpt / 'model.safetensors').stat().st_mtime:
    subprocess.run([str(cli), 'export', 'onnx', '--model', str(ckpt), '--task', 'text-generation-with-past', '--dtype', 'fp32', str(fp32)], check=True)

# 2. pack to 4-bit
EMB = 'model.embed_tokens.weight'
m = onnx.load(str(fp32 / 'model.onnx')); g = m.graph; inits = {i.name: i for i in g.initializer}
E = numpy_helper.to_array(inits[EMB]).astype(np.float32); V, D = E.shape; NB = D // BLK
new_init, drop, nodes, count = [], {EMB}, [], 0
for n in g.node:
    if n.op_type == 'MatMul' and n.input[1] in inits and len(inits[n.input[1]].dims) == 2:
        W = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32); K, N = W.shape; bq, sc = pack4(W.T, BLK)
        tag = f'q4_{count}'; count += 1; drop.add(n.input[1])
        new_init += [numpy_helper.from_array(bq, tag + '_w'), numpy_helper.from_array(sc.reshape(-1), tag + '_s')]
        nodes.append(helper.make_node('MatMulNBits', [n.input[0], tag + '_w', tag + '_s'], list(n.output), name=n.name + '_q4', domain='com.microsoft',
                                      K=K, N=N, bits=4, block_size=BLK, accuracy_level=ACC))
    else: nodes.append(n)
eq, esc = pack4(E, BLK)
i64 = lambda v, n: numpy_helper.from_array(np.array(v, np.int64), n)
new_init += [numpy_helper.from_array(eq, 'emb_q4'), numpy_helper.from_array(esc.reshape(-1), 'emb_scales_flat'), numpy_helper.from_array(esc, 'emb_scales2d'),
             numpy_helper.from_array(np.array([4], np.uint32), 'c4_u32'), numpy_helper.from_array(np.array(8.0, np.float32), 'c8_f'),
             i64([-1], 'axis_last'), i64([0, 0, NB, BLK], 'shape_blocks'), i64([0, 0, D], 'shape_out'),
             i64([-1], 'sl_starts'), i64([2 ** 62], 'sl_ends'), i64([1], 'sl_axes')]
tr = next(n for n in nodes if n.op_type == 'Transpose' and list(n.input) == [EMB])
lm = next(n for n in nodes if n.op_type == 'MatMul' and n.input[1] == tr.output[0])
last = not args.all_logits   # generation only needs the last position's logits: slice the hidden state before the 49k-wide output layer
lm_new = ([helper.make_node('Slice', [lm.input[0], 'sl_starts', 'sl_ends', 'sl_axes'], ['last_hidden'], name='last_hidden_slice')] if last else []) + \
         [helper.make_node('MatMulNBits', ['last_hidden' if last else lm.input[0], 'emb_q4', 'emb_scales_flat'], list(lm.output), name='lm_head_q4', domain='com.microsoft',
                           K=D, N=V, bits=4, block_size=BLK, accuracy_level=ACC)]
gat = next(n for n in nodes if n.op_type == 'Gather' and n.input[0] == EMB); ids, emb_out = gat.input[1], gat.output[0]
extra = []
def node(op, ins, outs, **kw): extra.append(helper.make_node(op, ins, outs, name='emb_' + outs[0], **kw)); return outs[0]
rows = node('Gather', ['emb_q4', ids], ['emb_rows'], axis=0); sc = node('Gather', ['emb_scales2d', ids], ['emb_sc_rows'], axis=0)
x32 = node('Cast', [rows], ['emb_u32'], to=T.UINT32); hi = node('BitShift', [x32, 'c4_u32'], ['emb_hi'], direction='RIGHT')
lo = node('Sub', [x32, node('BitShift', [hi, 'c4_u32'], ['emb_hi_shl'], direction='LEFT')], ['emb_lo'])
lo_u = node('Unsqueeze', [node('Cast', [lo], ['emb_lo_f'], to=T.FLOAT), 'axis_last'], ['emb_lo_u'])
hi_u = node('Unsqueeze', [node('Cast', [hi], ['emb_hi_f'], to=T.FLOAT), 'axis_last'], ['emb_hi_u'])
flat = node('Reshape', [node('Concat', [lo_u, hi_u], ['emb_both'], axis=-1), 'shape_blocks'], ['emb_blocks'])
node('Reshape', [node('Mul', [node('Sub', [flat, 'c8_f'], ['emb_centered']), node('Unsqueeze', [sc, 'axis_last'], ['emb_sc_u'])], ['emb_scaled']), 'shape_out'], [emb_out])

final = [x for n in nodes if n is not gat and n is not tr for x in (lm_new if n is lm else [n])] + extra
keep = [i for i in g.initializer if i.name not in drop]; g.ClearField('node'); g.ClearField('initializer'); g.initializer.extend(keep); g.initializer.extend(new_init)
if not any(o.domain == 'com.microsoft' for o in m.opset_import): m.opset_import.append(helper.make_opsetid('com.microsoft', 1))
have = {i.name for i in g.initializer} | {i.name for i in g.input}; pending = list(final); ordered = []   # topological order
while pending:
    moved = False
    for n in list(pending):
        if all(i == '' or i in have for i in n.input): ordered.append(n); have.update(n.output); pending.remove(n); moved = True
    if not moved: raise SystemExit('cannot order graph at ' + pending[0].name)
g.node.extend(ordered)
dst = out / ('model_q4_all_logits.onnx' if args.all_logits else 'model_q4.onnx'); onnx.save(m, str(dst))
print(f'packed {count} layer matrices + shared embedding ({V}x{D})  ->  {dst}  {dst.stat().st_size / 1e6:.1f} MB')

# 3. what the runtime needs to know about the model
cfg = json.load(open(ckpt / 'config.json'))
info = {'layers': cfg['num_hidden_layers'], 'kv_heads': cfg['num_key_value_heads'], 'head_dim': cfg['hidden_size'] // cfg['num_attention_heads'], 'block': BLK}
(out / 'model_info.json').write_text(json.dumps(info))
for f in ('tokenizer.json', 'tokenizer_config.json'): shutil.copy(ckpt / f, out / f)
