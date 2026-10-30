"""Offline CUDA compilation through the actual JIT binder and all 12 configs."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import traceback

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
options = parser.parse_args()
OUT = options.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ['TRITON_ALWAYS_COMPILE'] = '1'
import torch
import triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend

SOURCE = Path(__file__).resolve().parent / 'bmm_optimized_grid_axes_checked.py'
spec = importlib.util.spec_from_file_location('bmm_grid_axes_candidate', SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
configs = module.bmm_kernel.configs
assert len(configs) == 12
heuristics = module.bmm_kernel.fn
fn = heuristics.fn
backend = CUDABackend(GPUTarget('cuda', 80, 32))
binder = create_function_from_signature(fn.signature, fn.params, backend)
profiles = []
for occurrence, config in enumerate(configs):
    profiles.append((f'config_{occurrence:02d}_tails', occurrence,
                     config.kwargs['TILE_M'] + 1, config.kwargs['TILE_N'] + 1, 33))
for dm in (False, True):
    for dn in (False, True):
        for dk in (False, True):
            if dm or dn or dk:
                profiles.append((f'masks_{int(dm)}{int(dn)}{int(dk)}', 0,
                                 32 if dm else 33, 32 if dn else 33, 32 if dk else 33))
profiles.append(('zero_k_live', 0, 3, 5, 0))
results = []
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name, occurrence, M, N, K in profiles:
    config = configs[occurrence]
    metadata = {'M': M, 'N': N, 'K': K, **config.kwargs}
    flags = {key: bool(value(metadata)) for key, value in heuristics.values.items()}
    a = torch.empty((2, M, K), dtype=torch.float32)
    b = torch.empty((2, K, N), dtype=torch.float32)
    out = torch.empty((2, M, N), dtype=torch.float32)
    launch_options = {'num_warps': config.num_warps, 'num_stages': config.num_stages,
                      'num_ctas': config.num_ctas, 'enable_fp_fusion': True}
    bound, specialization, binder_options = binder(a, b, out, M, N, K, **config.kwargs, **flags, **launch_options)
    parsed, signature, constants, attrs = fn._pack_args(backend, binder_options, bound, specialization, binder_options)
    row = {'name': name, 'occurrence': occurrence, 'source_sha256': sha(SOURCE),
           'shape': [2, M, N, K], 'config': config.kwargs, 'heuristics': flags,
           'requested_options': launch_options, 'signature': signature,
           'constants': {fn.arg_names[key[0]]: str(value) for key, value in constants.items()},
           'attrs': {str(key): value for key, value in attrs.items()},
           'actual_JIT_signature_binder': True, 'CPU_descriptors_for_offline_CUDA_target': True}
    os.environ['TRITON_CACHE_DIR'] = str(OUT / 'cache' / name)
    try:
        code = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                              target=backend.target, options=launch_options)
        artifacts = {}
        for kind in ('ttir', 'ttgir', 'llir', 'ptx'):
            artifact = OUT / (name + '.' + kind)
            artifact.write_text(code.asm[kind])
            artifacts[kind] = {'path': str(artifact), 'sha256': sha(artifact)}
        assert code.metadata.num_warps == 4 and code.metadata.num_stages == config.num_stages
        assert code.metadata.num_ctas == 1 and code.metadata.enable_fp_fusion
        row.update(status='compiled', artifacts=artifacts, metadata=code.metadata._asdict(),
                   scalar_fma='fma.rn.f32' in code.asm['ptx'], tensor_core_mma='mma.sync' in code.asm['ptx'])
        assert row['scalar_fma'] and not row['tensor_core_mma']
    except Exception as error:
        row.update(status='failed', error=str(error))
        (OUT / (name + '.error.txt')).write_text(traceback.format_exc())
    results.append(row)
    print(name, row['status'], row.get('error', ''), flush=True)
    report = {'status': 'PASS' if all(r['status'] == 'compiled' for r in results) else 'FAIL',
              'source': {'path': str(SOURCE), 'sha256': sha(SOURCE)},
              'triton': triton.__version__, 'target': dataclasses.asdict(backend.target),
              'actual_JIT_signature_binder': True, 'GPU_execution': False,
              'universal_compiler_conformance_claim': False, 'profiles': results}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
assert len(results) == 20 and all(r['status'] == 'compiled' for r in results)
assert {r['occurrence'] for r in results} == set(range(12))
assert {tuple(r['heuristics'][key] for key in ('DIVISIBLE_M', 'DIVISIBLE_N', 'DIVISIBLE_K')) for r in results} == set(__import__('itertools').product((False, True), repeat=3))
print('PASS 20 binder profiles; 12 config occurrences; 8 mask profiles; live K=0. No GPU execution.')
