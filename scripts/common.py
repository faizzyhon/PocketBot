"""Shared helpers: config, knowledge loading, chunking. Used by every pipeline script."""
import json, re, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'          # intermediate files (datasets, checkpoints, ONNX). git-ignored.
DIST = ROOT / 'dist'            # the static folder you deploy. git-ignored.


def config():
    return tomllib.loads((ROOT / 'pocketbot.toml').read_text('utf-8'))


def load_faq(path=None):
    """knowledge/faq.txt. One entry per block:
        @@ category | weight        (weight optional, default 1.0)
        Q: first way of asking
        Q: another way of asking   (any number of Q: lines)
        A: the answer              (may span lines)
    """
    path = Path(path or ROOT / 'knowledge' / 'faq.txt')
    entries = []
    for blk in re.split(r'^@@ ?', path.read_text('utf-8'), flags=re.M)[1:]:
        head, _, rest = blk.partition('\n')
        cat, _, w = head.partition('|')
        qs, ans, mode = [], [], None
        for line in rest.split('\n'):
            if line.startswith('Q:'): qs.append(line[2:].strip()); mode = 'q'
            elif line.startswith('A:'): ans.append(line[2:].strip()); mode = 'a'
            elif mode == 'a': ans.append(line)
        a = '\n'.join(ans).strip()
        if not qs or not a: raise SystemExit(f'faq.txt: entry "{head.strip()}" needs at least one Q: and an A:')
        entries.append({'cat': cat.strip(), 'w': float(w) if w.strip() else 1.0, 'qs': qs, 'a': a})
    return entries


def load_docs(folder=None, max_words=190, overlap=30):
    """knowledge/docs/*.md|txt -> passages. A document may start with <!-- url: /menu --> so answers can link to it."""
    folder = Path(folder or ROOT / 'knowledge' / 'docs'); out = []
    for f in sorted(folder.glob('*')) if folder.exists() else []:
        if f.suffix not in ('.md', '.txt'): continue
        text = f.read_text('utf-8'); m = re.match(r'\s*<!--\s*url:\s*(\S+)\s*-->', text); url = m.group(1) if m else ''
        text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
        title = (re.search(r'^#\s+(.+)$', text, re.M) or [None, f.stem.replace('-', ' ').title()])[1].strip()
        sections, head, buf = [], '', []
        for line in text.split('\n'):
            h = re.match(r'^#{2,3}\s+(.+)$', line)
            if h: sections.append((head, ' '.join(buf))); head, buf = h.group(1).strip(), []
            elif not line.startswith('# '): buf.append(line.strip())
        sections.append((head, ' '.join(buf)))
        for head, body in sections:
            words = body.split(); i = 0
            while i < len(words):
                piece = words[i:i + max_words]
                if len(piece) >= 8: out.append({'u': url, 't': title, 'h': head, 'x': ' '.join(piece)})
                if i + max_words >= len(words): break
                i += max_words - overlap
    return out


def faq_passage(e, q=None):
    return f"Q: {q or e['qs'][0]}\nA: {e['a']}"
