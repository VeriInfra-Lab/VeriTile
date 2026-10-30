"""Actual wrapper/JIT binder; offline SM80 compilation, with no GPU execution."""
import hashlib
import importlib.util
import itertools
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
SOURCE = HERE / 'fused_recurrent_hgrn_repaired.py'
assert sha(SOURCE) == 'f920993bd8b61bc71450ab7a1775ab8dc4c350435e81b83728834df1af922cde'
actual = load('hgrn_profile_source', SOURCE)
fixtures = load('hgrn_profile_fixtures', HERE / 'check_contract.py')

class Capture:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.args = [x.data if isinstance(x, fixtures.Tensor) else x for x in args]
            self.kwargs, self.grid = kwargs, grid
        return launch

target = GPUTarget('cuda', 80, 32)
backend = CUDABackend(target)
records = []
# All twelve actual autotune candidates; then branch, empty/singleton/time-loop controls.
profiles = [(3, 65, True, True, bd, warps)
            for bd, warps in itertools.product((32, 64, 128), (1, 2, 4, 8))]
profiles += [(t, 33, u, v, 32, 4)
             for t, u, v in itertools.product((0, 1, 2, 4), (False, True), (False, True))]
summary = OUT / 'profiles.json'
if sys.argv[1:] == ['--loop-controls']:
    profiles = [(t, 33, flag, flag, 32, 4) for t, flag in itertools.product((17, 1024), (False, True))]
    summary = OUT / 'loop-profiles.json'
for t, d, seed, final, bd, warps in profiles:
    program, context, _, _ = fixtures.host(bd=bd)
    captures = {phase: Capture() for phase in ('fwd', 'bwd')}
    for phase in captures:
        program.forward.__globals__[f'fused_recurrent_hgrn_{phase}_kernel'] = captures[phase]
    shape = (1, 2, t, d)
    x = fixtures.Tensor(torch.zeros(shape))
    g = fixtures.Tensor(torch.zeros(shape))
    h0 = fixtures.Tensor(torch.zeros((1, 2, d))) if seed else None
    dht = fixtures.Tensor(torch.zeros((1, 2, d))) if final else None
    ctx = context()
    program.forward(ctx, x, g, h0, final)
    program.backward(ctx, fixtures.Tensor(torch.zeros(shape)), dht)
    for phase, capture in captures.items():
        autotuner = getattr(actual, f'fused_recurrent_hgrn_{phase}_kernel')
        assert any(c.kwargs == {'BD': bd} and c.num_warps == warps for c in autotuner.configs)
        fn = autotuner.fn
        kwargs = dict(capture.kwargs, BD=bd, num_warps=warps)
        binder = create_function_from_signature(fn.signature, fn.params, backend)
        bound, specialization, options = binder(*capture.args, **kwargs)
        parsed, signature, constants, attrs = fn._pack_args(backend, options, bound, specialization, options)
        compiled = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                                  target=target, options=options)
        name = f'{phase}_t{t}_d{d}_u{int(seed)}_v{int(final)}_bd{bd}_w{warps}'
        artifacts = []
        for kind in ('ttir', 'ptx'):
            path = OUT / (name + '.' + kind)
            path.write_text(compiled.asm[kind])
            artifacts.append(dict(path=str(path), sha256=sha(path)))
        records.append(dict(name=name, status='compiled', source_sha256=sha(SOURCE),
            grid=capture.grid({'BD': bd}), signature=signature, options=repr(parsed),
            constants={fn.arg_names[k[0]]:str(v) for k,v in constants.items()}, artifacts=artifacts))
        print(name, 'compiled', flush=True)
        summary.write_text(json.dumps(dict(triton=triton.__version__,
            gpu_execution=False, numerical_experiments=False, actual_binder=True,
            actual_wrapper_arguments=True, results=records), indent=2)+'\n')
print(f'PASS: {len(records)} actual-wrapper/JIT-binder source profiles.')
