"""Actual wrapper arguments and JIT binder, offline SM80 compilation only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'kernel-evidence'
os.environ['TRITON_CACHE_DIR'] = str(OUT / 'cache')
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
from triton.runtime.jit import create_function_from_signature

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
SOURCE = HERE / 'flash_decode2_llama_repaired.py'
assert sha(SOURCE) == '2d8a58c053404bcc7b8b58117aaea53cb4eed4742ddcadc90c8eaa6075e34c85'
actual = load('llama_profile_source', SOURCE)
fixtures = load('llama_profile_fixtures', HERE.parent / 'flash_decode2_phi/check_contract.py')
call, _, _ = fixtures.harness(SOURCE)
fn = actual._fwd_kernel_flash_decode_stage2

class Capture:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args = [x.data if isinstance(x, fixtures.Tensor) else x for x in args]
            self.kwargs, self.grid = kwargs, grid
        return launch

capture = Capture()
call.__globals__[fn.__name__] = capture
records = []
for d in (16, 32, 64, 128):
    for b, h, s, q in ((2, 3, 4, 4), (1, 1, 1, 2**31-1),
                        (1, 1, 2, 1), (1, 1, 2, 2**30), (1, 1, 2, 2**31-2)):
        args, _ = fixtures.sample(b, h, s, d, [q] * b, q)
        call(*args)
        target = GPUTarget('cuda', 80, 32)
        backend = CUDABackend(target)
        binder = create_function_from_signature(fn.signature, fn.params, backend)
        bound, specialization, options = binder(*capture.args, **capture.kwargs)
        parsed, signature, constants, attrs = fn._pack_args(backend, options, bound, specialization, options)
        compiled = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                                  target=target, options=options)
        name = f'd{d}_q{q}'
        artifacts = []
        for kind in ('ttir', 'ptx'):
            path = OUT / (name + '.' + kind)
            path.write_text(compiled.asm[kind])
            artifacts.append(dict(path=str(path), sha256=sha(path)))
        records.append(dict(name=name, status='compiled', source_sha256=sha(SOURCE),
            grid=capture.grid, signature=signature, options=repr(parsed),
            constants={fn.arg_names[k[0]]:str(v) for k,v in constants.items()}, artifacts=artifacts))
        print(name, 'compiled', flush=True)
        (OUT/'profiles.json').write_text(json.dumps(dict(triton=triton.__version__,
            gpu_execution=False, actual_binder=True, actual_wrapper_arguments=True,
            results=records), indent=2)+'\n')
print('PASS: 20 source profiles across all admitted D values and sequence-block boundaries.')
