"""Regenerates docs/pipeline.gif, docs/runtime.gif and docs/architecture.png with Pillow (pip install Pillow).
The runtime GIF replays real answers recorded from the demo page: edit docs/runtime_examples.json to re-record. python docs/make_docs_assets.py"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1224, 600
PAPER, INK, GREY, BLUE, TINT, LINE = (244, 241, 234), (17, 17, 17), (150, 146, 138), (20, 20, 204), (225, 225, 248), (207, 200, 184)
F = 'C:/Windows/Fonts/'   # Windows fonts; on Linux/macOS change to DejaVuSans*.ttf paths
title = ImageFont.truetype(F + 'segoeuib.ttf', 21); lane = ImageFont.truetype(F + 'consolab.ttf', 15)
head = ImageFont.truetype(F + 'segoeuib.ttf', 16); body = ImageFont.truetype(F + 'consola.ttf', 12); big = ImageFont.truetype(F + 'segoeuib.ttf', 15)
BW, BH, GAP, X0 = 214, 118, 26, 25

BUILD = [('1  DATA', ['knowledge/faq.txt + docs', 'become chat examples:', 'with / without context']),
         ('2  FINE-TUNE', ['02_train --stage sft', 'all weights of the base', 'model, best epoch kept']),
         ('3  QUANT-AWARE', ['02_train --stage qat', 'weights rounded to 4 bits', 'in every forward pass']),
         ('4  EXPORT', ['03_export: PyTorch -> ONNX', 'checked against PyTorch', 'weights packed to 4-bit']),
         ('5  WEB BUILD', ['05_build_web -> dist/', 'model + tokenizer + search', 'index + runtime libraries'])]
RUN = [('1  RULES', ['injection, identity,', 'harmful, hello, thanks', '-> fixed replies']),
       ('2  CALCULATOR', ['plain arithmetic', 'parsed and computed', 'exactly, no eval']),
       ('3  SEARCH', ['BM25 over index.json', 'FAQ entries + doc', 'passages, best 4']),
       ('4  MODEL', ['ONNX Runtime in the tab:', 'WebGPU, else WebAssembly', 'prompt = passages + question']),
       ('5  ANSWER', ['text + "Related" link', 'model cached by browser', 'nothing leaves the tab'])]


def box(d, i, y, step, state):
    x = X0 + i * (BW + GAP); name, lines = step; col = {'idle': GREY, 'done': INK, 'on': BLUE}[state]
    d.rectangle([x, y, x + BW, y + BH], fill=TINT if state == 'on' else PAPER, outline=col, width=3 if state == 'on' else 2)
    d.text((x + 12, y + 10), name, font=head, fill=col)
    for k, t in enumerate(lines): d.text((x + 12, y + 42 + k * 21), t, font=body, fill=INK if state != 'idle' else GREY)
    if i < 4:
        ax, ay, c = x + BW + 3, y + BH // 2, INK if state in ('done', 'on') else LINE
        d.line([ax, ay, ax + GAP - 8, ay], fill=c, width=2); d.polygon([(ax + GAP - 8, ay - 5), (ax + GAP - 8, ay + 5), (ax + GAP - 1, ay)], fill=c)


def save_gif(frames, durations, path):
    q = [f.quantize(colors=24, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print('saved', path.name, round(path.stat().st_size / 1024), 'KB,', len(frames), 'frames')


# ── pipeline.gif: build row on top, run row below, accuracy bars last ──
def pipe_frame(b, r, results=False):
    im = Image.new('RGB', (W, H), PAPER); d = ImageDraw.Draw(im)
    d.text((25, 18), 'PocketBot: from your FAQ to a chatbot that runs inside the web page', font=title, fill=INK)
    d.text((25, 52), 'BUILD  (once, on your computer)', font=lane, fill=BLUE)
    for i, s in enumerate(BUILD): box(d, i, 76, s, 'on' if i == b else ('done' if i < b else 'idle'))
    d.text((25, 226), "RUN  (every question, in the visitor's browser)", font=lane, fill=BLUE)
    for i, s in enumerate(RUN): box(d, i, 250, s, 'on' if i == r else ('done' if i < r else 'idle'))
    d.line([25, 392, W - 25, 392], fill=LINE, width=1)
    if not results:
        d.text((25, 408), 'The file built in step 4 is what the browser downloads once and caches (about 89 MB for the default base model).', font=body, fill=GREY)
        d.text((25, 432), 'Base model: SmolLM2-135M-Instruct (Hugging Face). The rest of the pipeline is in this repository.', font=body, fill=GREY)
    else:
        d.text((25, 404), 'Why step 3 exists: next-token accuracy on held-out examples (the author\'s own bot, 1,150 training examples)', font=big, fill=INK)
        for k, (lab, v, c) in enumerate([('full precision, 539 MB', 97.16, INK), ('4-bit by plain rounding, 89 MB', 88.84, GREY), ('4-bit after quant-aware training, 89 MB', 97.05, BLUE)]):
            y = 438 + k * 30; d.text((25, y + 2), lab, font=body, fill=INK); x1 = 330 + int((v - 80) / 20 * 560)
            d.rectangle([330, y, x1, y + 20], fill=c); d.text((x1 + 8, y + 2), f'{v:.1f}%', font=body, fill=INK)
        d.text((25, 536), 'Axis starts at 80%. The held-out set comes from the same generator as the training data, so read the gap between the bars,', font=body, fill=GREY)
        d.text((25, 553), 'not the absolute level. Your numbers will differ: measure your own with scripts/04_evaluate.py.', font=body, fill=GREY)
    return im

fr, du = [pipe_frame(-1, -1)], [1200]
for i in range(5): fr.append(pipe_frame(i, -1)); du.append(1100)
for i in range(5): fr.append(pipe_frame(5, i)); du.append(1100)
fr.append(pipe_frame(5, 5, True)); du.append(6000)
save_gif(fr, du, HERE / 'pipeline.gif')

# ── architecture.png: the same two rows with file names ──
a = Image.new('RGB', (W, 470), PAPER); d = ImageDraw.Draw(a)
d.text((25, 18), 'Architecture: what runs where, and which file connects the two halves', font=title, fill=INK)
d.text((25, 52), 'BUILD TIME  (Python + GPU)', font=lane, fill=BLUE)
A_BUILD = [('knowledge/', ['faq.txt, docs/*.md', 'probes.json', 'pocketbot.toml']), ('01_build_data', ['-> build/data/', '   train.jsonl, val.jsonl']), ('02_train (x2)', ['sft -> build/ckpt/sft', 'qat -> build/ckpt/qat']),
           ('03_export', ['-> build/model/', '   model_q4.onnx', '   tokenizer.json']), ('05_build_web', ['-> dist/  (static)', '   + index.json', '   + bot.config.json'])]
for i, s in enumerate(A_BUILD): box(d, i, 76, s, 'done')
d.text((25, 214), 'dist/ is the only thing you deploy', font=lane, fill=BLUE); d.line([X0 + 4 * (BW + GAP) + BW // 2, 76 + BH, X0 + 4 * (BW + GAP) + BW // 2, 238], fill=INK, width=2)
d.polygon([(X0 + 4 * (BW + GAP) + BW // 2 - 6, 232), (X0 + 4 * (BW + GAP) + BW // 2 + 6, 232), (X0 + 4 * (BW + GAP) + BW // 2, 244)], fill=INK)
d.text((25, 250), 'RUN TIME  (the visitor\'s browser tab, web/pocketbot.js)', font=lane, fill=BLUE)
A_RUN = [('rules', ['regex -> fixed replies', 'from bot.config.json']), ('calculator', ['recursive descent', 'no eval']), ('search', ['BM25', 'reads index.json']),
         ('model', ['reads model_q4.onnx', 'ONNX Runtime Web', 'WebGPU / WebAssembly']), ('answer', ['text + link', 'Cache Storage keeps', 'the model file'])]
for i, s in enumerate(A_RUN): box(d, i, 274, s, 'done')
d.text((25, 410), 'No server code. No API key. The only network traffic is loading the static files once.', font=body, fill=GREY)
d.text((25, 432), 'Rules and calculator run before the model on purpose: they handle what a 135M model cannot do reliably.', font=body, fill=GREY)
a.save(HERE / 'architecture.png'); print('saved architecture.png')

# ── runtime.gif: replay recorded answers ──
ex = HERE / 'runtime_examples.json'
if ex.exists():
    scenes = json.load(open(ex, encoding='utf-8')); frames, durs = [], []
    def run_frame(sc, upto, show_ans, typed=None):
        im = Image.new('RGB', (W, 470), PAPER); d = ImageDraw.Draw(im)
        d.text((25, 18), 'What happens to one question', font=title, fill=INK)
        d.text((25, 56), 'you type:', font=body, fill=GREY); d.text((110, 52), (sc['q'] if typed is None else sc['q'][:typed]) + ('|' if typed is not None else ''), font=big, fill=INK)
        for i, s in enumerate(RUN): box(d, i, 92, s, 'idle' if upto < 0 else ('on' if i == upto else ('done' if i < upto else 'idle')))
        d.line([25, 232, W - 25, 232], fill=LINE)
        if upto >= 0: d.text((25, 244), sc['detail'][min(upto, len(sc['detail']) - 1)], font=body, fill=BLUE)
        if show_ans:
            lines = sc['answer'].split('\n')
            d.rectangle([25, 290, W - 25, 322 + 22 * len(lines)], outline=INK, width=2, fill=TINT)
            d.text((37, 298), 'answer', font=body, fill=BLUE)
            for k, line in enumerate(lines): d.text((37, 320 + k * 22), line, font=big, fill=INK)
            d.text((25, 440), sc.get('footer', ''), font=body, fill=GREY)
        return im
    for sc in scenes:
        steps = {'rules': [0], 'calculator': [0, 1], 'model': [0, 1, 2, 3, 4]}[sc['via']]
        frames.append(run_frame(sc, -1, False, 0)); durs.append(400)
        frames.append(run_frame(sc, -1, False, len(sc['q']))); durs.append(900)
        for u in steps[:-1]: frames.append(run_frame(sc, u, False)); durs.append(1000)
        frames.append(run_frame(sc, steps[-1], True)); durs.append(3600)
    save_gif(frames, durs, HERE / 'runtime.gif')
