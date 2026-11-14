"""Downloads the two JavaScript libraries the browser needs and lays them out in web/vendor/lib (needs Node.js + npm).
  - @huggingface/transformers: used only for its tokenizer (Apache-2.0)
  - onnxruntime-web: runs the model on WebGPU or WebAssembly (MIT)
Versions are pinned to the ones this repo was tested with. Run once: python scripts/fetch_runtime.py"""
import shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path
from common import ROOT

TRANSFORMERS = '@huggingface/transformers@4.3.2'
ORT = 'onnxruntime-web@1.31.0-dev.20260914-8d85527a0'
LIB = ROOT / 'web' / 'vendor' / 'lib'

def pack(spec, tmp):
    r = subprocess.run(f'npm pack {spec} --silent', cwd=tmp, shell=True, capture_output=True, text=True)
    if r.returncode: raise SystemExit(f'npm pack {spec} failed:\n{r.stderr}\n(is Node.js installed?)')
    tgz = tmp / r.stdout.strip().splitlines()[-1]
    with tarfile.open(tgz) as t: t.extractall(tmp / tgz.stem)
    return tmp / tgz.stem / 'package' / 'dist'

def main():
    LIB.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t); tf, ort = pack(TRANSFORMERS, tmp), pack(ORT, tmp)
        # transformers.js imports two bare module names; point them at the local bundle so no import map or CDN is needed
        src = (tf / 'transformers.web.min.js').read_text('utf-8').replace('from"onnxruntime-web/webgpu"', 'from"./ort.wasm.bundle.min.js"').replace('from"onnxruntime-common"', 'from"./ort.wasm.bundle.min.js"')
        (LIB / 'transformers.web.min.js').write_text(src, 'utf-8')
        # .mjs -> .js: many static hosts serve .mjs as text/plain, which browsers refuse to run as a module
        for a, b in [('ort.wasm.bundle.min.mjs', 'ort.wasm.bundle.min.js'), ('ort.webgpu.bundle.min.mjs', 'ort.webgpu.bundle.min.js'),
                     ('ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.js'), ('ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.js')]:
            shutil.copy(ort / a, LIB / b)
        for f in ('ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.asyncify.wasm'): shutil.copy(ort / f, LIB / f)
    print('runtime libraries ready in', LIB, f'({sum(f.stat().st_size for f in LIB.iterdir()) / 1e6:.0f} MB)')

if __name__ == '__main__': main()
