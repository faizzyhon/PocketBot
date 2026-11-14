"""Run before every push:  python scripts/check_repo.py
Looks at exactly the files git would commit (tracked + untracked, minus .gitignore) and fails on: likely secrets, large files,
model weights, and email addresses other than example domains. Add the comment `allow-secret` on a line to exempt an obvious test constant. It is a safety net, not a guarantee: read `git status` and `git diff --cached` too."""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_MB = 3
SECRET = [(r'AIza[0-9A-Za-z_\-]{30,}', 'Google API key'), (r'sk-[A-Za-z0-9_\-]{20,}', 'OpenAI/Anthropic-style key'), (r'gh[pousr]_[A-Za-z0-9]{30,}', 'GitHub token'),
          (r'hf_[A-Za-z0-9]{30,}', 'Hugging Face token'), (r'AKIA[0-9A-Z]{16}', 'AWS access key'), (r'-----BEGIN [A-Z ]*PRIVATE KEY-----', 'private key'),
          (r'(?i)(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*["\'][^"\'\s]{8,}["\']', 'hard-coded credential')]
EMAIL = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
OK_EMAIL = ('.example', 'example.com', 'users.noreply.github.com')
BAD_EXT = {'.onnx', '.safetensors', '.gguf', '.bin', '.pt', '.pth', '.ckpt', '.zip', '.tar', '.gz', '.env', '.pem', '.key'}

def files():
    r = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, capture_output=True, text=True)
    if r.returncode: sys.exit('Run this inside a git repository (git init first).')
    return [ROOT / f for f in r.stdout.split('\n') if f and (ROOT / f).is_file()]

problems = []
for f in files():
    rel = f.relative_to(ROOT).as_posix(); size = f.stat().st_size
    if f.suffix.lower() in BAD_EXT: problems.append(f'{rel}: {f.suffix} files must not be committed')
    if size > MAX_MB * 1e6: problems.append(f'{rel}: {size / 1e6:.1f} MB is larger than {MAX_MB} MB')
    if f.suffix.lower() in {'.gif', '.png', '.wasm', '.jpg', '.ico'} or size > 2e6: continue
    try: text = f.read_text('utf-8')
    except UnicodeDecodeError: continue
    lines = text.split('\n')
    for pat, what in SECRET:
        for m in re.finditer(pat, text):
            line = lines[text.count('\n', 0, m.start())]
            if 'allow-secret' in line: continue   # an obvious test constant, marked on its own line
            problems.append(f'{rel}: looks like a {what}: {m.group(0)[:12]}...')
    for m in EMAIL.finditer(text):
        if not m.group(0).lower().endswith(OK_EMAIL): problems.append(f'{rel}: email address {m.group(0)} (is it meant to be public?)')

if problems:
    print('PROBLEMS:'); [print('  -', p) for p in sorted(set(problems))]; sys.exit(1)
print(f'ok: {len(files())} files would be committed, none flagged.')
