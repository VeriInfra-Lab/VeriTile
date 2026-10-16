#!/usr/bin/env python3
"""Offline compiler evidence for the unchanged two-loop RMSNorm source.

No CUDA tests, benchmark import or GPU execution. The probed native profiles
are separate from Lean's symbolic exact-real/structural theorems.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1] / 'rmsnorm_implementation.py'
FLOATS = ('fp16', 'bf16', 'fp32', 'fp64', 'fp8e4nv', 'fp8e5')


def compile_profile(input_type: str, weight: str, output: str) -> dict:
    os.environ.setdefault('TRITON_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'veritile-rmsnorm-native-cache'))
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'rmsnorm_triton')
    with tempfile.TemporaryDirectory(prefix='veritile-rmsnorm-source-') as directory:
        path = Path(directory) / 'rmsnorm_source.py'
        path.write_text('import triton\nimport triton.language as tl\n\n' + ast.unparse(node) + '\n')
        spec = importlib.util.spec_from_file_location('rmsnorm_source', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        function = module.rmsnorm_triton
        constants = {'N_SIZE': 4097, 'BLOCK_N_SIZE': 4096, 'eps': 1e-6}
        signature = {name: 'i32' for name in function.arg_names if name not in constants}
        signature.update(x_ptr='*' + input_type, rms_w_ptr='*' + weight, out_ptr='*' + output)
        try:
            compiled = triton.compile(ASTSource(function, signature, constexprs=constants),
                                     target=GPUTarget('cuda', 90, 32),
                                     options={'num_warps': 16, 'num_stages': 3})
            ttir = compiled.asm['ttir']
            return {'input': input_type, 'weight': weight, 'output': output, 'status': 'compiled',
                    'ttir_sha256': hashlib.sha256(ttir.encode()).hexdigest(),
                    'ptx_bytes': len(compiled.asm['ptx']),
                    'fp64_division': any('arith.divf' in line and 'f64' in line for line in ttir.splitlines()),
                    'fp64_multiply': any('arith.mulf' in line and 'f64' in line for line in ttir.splitlines()),
                    'fp32_sqrt': any(('sqrt' in line) and 'f32' in line for line in ttir.splitlines())}
        except Exception as error:
            return {'input': input_type, 'weight': weight, 'output': output,
                    'status': 'compile_failure', 'error_type': type(error).__name__, 'error': str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--child', nargs=3, metavar=('INPUT', 'WEIGHT', 'OUTPUT'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(compile_profile(*args.child)))
        return 0
    if args.output is None:
        parser.error('--output is required')
    profiles = [(x, w, x) for x in FLOATS for w in FLOATS]
    profiles += [('fp64', 'fp16', 'bf16'), ('fp16', 'fp64', 'bf16'),
                 ('fp64', 'fp64', 'fp32'), ('fp32', 'fp32', 'fp64')]
    outside = [('i32', 'fp32', 'i32'), ('fp32', 'i32', 'fp32'),
               ('i1', 'fp32', 'i1'), ('i32', 'i32', 'i32')]
    results = []
    for profile in profiles + outside:
        child = subprocess.run([sys.executable, __file__, '--child', *profile],
                               capture_output=True, text=True)
        if child.returncode:
            raise RuntimeError(f'Probe process failed for {profile}: {child.stderr}')
        result = json.loads(child.stdout)
        result['in_formal_float_scope'] = profile in profiles
        diagnostics = [line for line in child.stderr.splitlines()
                       if line.startswith(('Unsupported conversion', 'LLVM ERROR:'))]
        if diagnostics:
            result['compiler_diagnostics'] = diagnostics
        if profile in profiles:
            unsupported = (profile[0] == 'fp64' and profile[1] in ('fp8e4nv', 'fp8e5') or
                           profile[1] == 'fp64' and profile[2] in ('fp8e4nv', 'fp8e5'))
            result['expected'] = 'unsupported_fp8_fp64_conversion' if unsupported else 'compiled'
            result['precision_matches_expected'] = (
                result['status'] == 'compiled' and
                result['fp64_division'] == (profile[0] == 'fp64') and
                result['fp64_multiply'] == ('fp64' in profile[:2]) and result['fp32_sqrt'])
            result['matches_expected'] = (
                result['status'] == 'compile_failure' and bool(diagnostics)
                if unsupported else result['precision_matches_expected'])
        else:
            result['expected'] = 'compiled_outside_formal_scope'
            result['matches_expected'] = result['status'] == 'compiled'
        results.append(result)
        print(f'{profile}: {result["status"]}', flush=True)
    import triton.language.semantic as semantic
    receipt = {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'triton_version': importlib.metadata.version('triton'),
               'semantic_sha256': hashlib.sha256(Path(semantic.__file__).read_bytes()).hexdigest(),
               'target': {'backend': 'cuda', 'arch': 90, 'warp_size': 32},
               'constexprs': {'N_SIZE': 4097, 'BLOCK_N_SIZE': 4096, 'eps': 1e-6},
               'options': {'num_warps': 16, 'num_stages': 3},
               'scope': '36 floating wrapper profiles, 4 other floating outputs, 4 nonfloating source-domain probes. All runtime strides signed i32. Unchanged AST, offline compilation only; no GPU execution, global race proof, native equivalence or IEEE result.',
               'results': results}
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    return int(any(not row['matches_expected'] for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
