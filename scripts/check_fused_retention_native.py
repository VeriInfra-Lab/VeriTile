#!/usr/bin/env python3
"""Offline admission of the unchanged pinned retention JITs; never launch them.

This runner is a draft for root review. Every rejection is retained. Runtime
B/H/T/strides and fp32 scale stay runtime arguments, so example T values below
are NOT separately compiled or executed tests. Only original constexprs and
the wrapper's None arguments are specialized. No numerical/refinement proof.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

PIN = '369709736fce34c980e29fbf8db9b2117eaa5b22d65acad20ea3a898956426f5'
FLOATS = ('fp16', 'bf16', 'fp32', 'fp64', 'fp8e4nv', 'fp8e5')
JITS = {
    'forward': 'fused_recurrent_retention_fwd_kernel',
    'backward': 'fused_recurrent_retention_bwd_kernel',
}
NAMES = {*JITS.values(), 'FusedRecurrentRetentionFunction', 'fused_recurrent_retention'}
RUNTIME_INTS = ('s_qk_h', 's_qk_t', 's_qk_d', 's_vo_h', 's_vo_t', 's_vo_d', 'B', 'H', 'T')
# Both arange sizes must be powers of two. Keep the actual wrapper rule even
# for (3,5): its expected rejection is evidence, not a reason to round it up.
SHAPES = ((1, 1), (16, 16), (33, 35), (3, 5))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ast_digest(node):
    return digest(ast.dump(node, include_attributes=False).encode())


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size}


def write_receipt(path, receipt):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def extract_original(source, destination):
    raw = source.read_bytes()
    if digest(raw) != PIN:
        raise ValueError('Pinned source changed: ' + str(source))
    text = raw.decode()
    tree = ast.parse(text)
    nodes = [n for n in tree.body
             if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno < 10
             or isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES]
    if {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))} != NAMES:
        raise ValueError('Missing or changed original JIT/wrapper definitions')
    lines = text.splitlines(keepends=True)
    # Copy each full definition verbatim, including its decorators. Exclude only
    # the original CUDA test harness and its import-time invocation.
    chunks = []
    for node in nodes:
        begin = min([node.lineno, *(d.lineno for d in getattr(node, 'decorator_list', []))])
        chunks.append(''.join(lines[begin - 1:node.end_lineno]))
    destination.write_text('\n'.join(chunks) + '\n')
    emitted = ast.parse(destination.read_text())
    if [ast_digest(n) for n in emitted.body] != [ast_digest(n) for n in nodes]:
        raise ValueError('Extraction modified a definition or decorator')
    return {
        'source': file_record(source), 'pinned_sha256': PIN,
        'extracted': file_record(destination),
        'definition_ast_sha256': {n.name: ast_digest(n) for n in nodes
                                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))},
        'original_statement_spelling_preserved': True,
        'test_harness_executed': False, 'wrapper_executed': False,
    }


def runtime_examples(dk, dv):
    examples = []
    for t in (0, 1, 3, 8):
        b, h = 2, 3
        # Values are documentary contiguous-layout examples. They are never
        # passed to ASTSource.constexprs, a driver, or a compiled kernel.
        examples.append({
            'B': b, 'H': h, 'T': t, 'scale_python_float': dk ** -0.5,
            's_qk_h': t * dk, 's_qk_t': dk, 's_qk_d': 1,
            's_vo_h': t * dv, 's_vo_t': dv, 's_vo_d': 1,
            'grid': [(dv + min(dv, 32) - 1) // min(dv, 32),
                     (dk + min(dk, 32) - 1) // min(dk, 32), b * h],
            'compiled_with_these_values': False, 'executed': False,
        })
    return examples


def make_case(kind, src, dst, dk, dv, seed, final, int_abi):
    inputs = ('q', 'k', 'v') if kind == 'forward' else ('q', 'k', 'v', 'do')
    outputs = ('o',) if kind == 'forward' else ('dq', 'dk', 'dv')
    signature = {name: '*' + src for name in inputs}
    signature.update({name: '*' + dst for name in outputs})
    signature.update({name: int_abi for name in RUNTIME_INTS})
    signature['scale'] = 'fp32'
    constexprs = {'BK': min(dk, 32), 'BV': min(dv, 32), 'DK': dk, 'DV': dv,
                 'USE_INITIAL_STATE': seed}
    if seed:
        signature['initial_state'] = '*' + src
    else:
        signature['initial_state'] = 'constexpr'
        constexprs['initial_state'] = None  # Exactly the original wrapper ABI.
    if kind == 'forward':
        constexprs['STORE_FINAL_STATE'] = final
        if final:
            signature['final_state'] = '*' + dst
        else:
            signature['final_state'] = 'constexpr'
            constexprs['final_state'] = None
    name = f'{kind}-{src}-{dst}-dk{dk}-dv{dv}-seed{int(seed)}-final{int(final)}-{int_abi}'
    return {'name': name, 'kind': kind, 'input_dtype': src, 'output_dtype': dst,
            'signature': signature, 'constexprs': constexprs,
            'options': {'num_warps': 1, 'num_stages': 1},
            'wrapper_output_dtype_rule_satisfied': src == dst,
            'runtime_examples_not_executed': runtime_examples(dk, dv)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='A new directory for receipt and all artifacts')
    parser.add_argument('--arch', type=int, default=90)
    parser.add_argument('--integer-abi', choices=('i32', 'i64'), default='i32')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = args.root.resolve() / 'bench/tritonbench_g/fused_recurrent_retention/fused_recurrent_retention.py'
    source_info = extract_original(source, output / 'original.py')
    # A fresh cache prevents unrelated existing cache entries from being taken
    # as newly compiled evidence. No driver is initialized and no grid launched.
    os.environ['TRITON_CACHE_DIR'] = str(output / 'cache')
    import torch
    import triton
    from triton import knobs
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDABackend, get_ptxas
    from triton.compiler import ASTSource
    from triton.runtime.jit import JITFunction, mangle_type

    # Check actual installed runtime scalar/None mangling, not a hard-coded
    # assumption about how Python floats or absent optional buffers lower.
    if mangle_type(0.25) != 'fp32' or mangle_type(None) != 'constexpr':
        raise RuntimeError('Runtime scalar/None ABI differs; do not substitute another ABI')
    target = GPUTarget('cuda', args.arch, 32)
    backend = CUDABackend(target)
    options = {'num_warps': 1, 'num_stages': 1}
    resolved_options = backend.parse_options(dict(options))
    if (resolved_options.arch != f'sm{args.arch}' or resolved_options.ir_override
            or knobs.compilation.override or knobs.runtime.add_stages_inspection_hook is not None):
        raise RuntimeError('Compiler target/IR override prevents unchanged-source observation')
    ptxas = get_ptxas(args.arch)
    ptxas_version = subprocess.run([ptxas.path, '--version'], check=True,
                                  capture_output=True, text=True).stdout
    package = Path(triton.__file__).resolve().parent
    compiler_paths = [package / relative for relative in (
        '__init__.py', 'compiler/compiler.py', 'compiler/code_generator.py',
        'runtime/jit.py', 'language/core.py', 'language/math.py',
        'backends/compiler.py', 'backends/nvidia/compiler.py', 'knobs.py')]
    compiler_paths += sorted((package / '_C').glob('libtriton*.so'))
    if not any(p.name.startswith('libtriton') for p in compiler_paths):
        raise RuntimeError('Could not identify the loaded compiler binary')
    captured = [source, Path(__file__).resolve(), output / 'original.py',
                Path(ptxas.path), *compiler_paths,
                *(Path(p) for _, p in resolved_options.extern_libs)]
    before = {str(p.resolve()): file_record(p) for p in captured}
    spec = importlib.util.spec_from_file_location('veritile_retention_native_original', output / 'original.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    functions = {kind: getattr(module, name) for kind, name in JITS.items()}
    if not all(isinstance(fn, JITFunction) for fn in functions.values()):
        raise RuntimeError('Original decorators no longer yield JITFunction objects')
    receipt = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'source': source_info, 'checker': file_record(__file__),
        'target': {'backend': 'cuda', 'arch': args.arch, 'warp_size': 32},
        'environment': {
            'python': sys.version, 'platform': platform.platform(),
            'triton_version': triton.__version__, 'torch_version': torch.__version__,
            'torch_cuda_build_version': torch.version.cuda,
            'ptxas': {**file_record(ptxas.path), 'cuda_version': ptxas.version,
                      'version_output': ptxas_version},
            'compiler_files': [file_record(p) for p in compiler_paths],
            'extern_libs': {name: file_record(path) for name, path in resolved_options.extern_libs},
            'resolved_options': asdict(resolved_options),
            'cache_invalidating_environment': get_cache_invalidating_env_vars(),
            'scalar_mangling': {'python_float': mangle_type(0.25),
                               'small_python_int': mangle_type(2), 'None': mangle_type(None)},
            'runtime_integer_abi_requested': args.integer_abi,
        },
        'scope': ('Unchanged two JIT definitions, decorators, autograd class and wrapper ASTs. '
                  'All 36 common-input/independent-output floating storage pairs, four original '
                  'wrapper block shapes, all seed/final flags, one warp/one stage. B/H/T and '
                  'strides remain runtime signed integers; scale remains runtime fp32. Absent '
                  'optional buffers are original None constexprs. T examples are not compiled '
                  'specializations or executed tests. Mixed output dtypes are direct-JIT profiles, '
                  'not emitted by the q.new_empty wrapper. Every source rejection is retained. '
                  'No GPU execution, host reduction/dtype admission, universal shape admission, '
                  'integer-overflow proof, native/IEEE correctness or independent Lean replay.'),
        'kernel_launched': False, 'complete_task': False, 'independent_replay': False,
        'expected_cases': len(FLOATS) ** 2 * len(SHAPES) * 6,
        'results': [], 'inputs_before': before,
    }
    receipt_path = output / 'receipt.json'
    write_receipt(receipt_path, receipt)
    for kind in ('forward', 'backward'):
        flags = itertools.product((False, True), repeat=2) if kind == 'forward' else ((False, False), (True, False))
        for seed, final in flags:
            for (dk, dv), src, dst in itertools.product(SHAPES, FLOATS, FLOATS):
                row = make_case(kind, src, dst, dk, dv, seed, final, args.integer_abi)
                path = output / row['name']
                path.mkdir()
                try:
                    fn = functions[kind]
                    if set(row['signature']) | set(row['constexprs']) != set(fn.arg_names):
                        raise ValueError('Incomplete source ABI in case ' + row['name'])
                    compiled = triton.compile(ASTSource(fn, row['signature'], constexprs=row['constexprs']),
                                              target=target, options=dict(row['options']))
                    row['status'] = 'compiled'
                    row['artifacts'] = {}
                    for extension in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
                        payload = compiled.asm[extension]
                        raw = payload.encode() if isinstance(payload, str) else payload
                        artifact = path / ('kernel.' + extension)
                        artifact.write_bytes(raw)
                        row['artifacts'][extension] = file_record(artifact)
                    row['source_ir_observations'] = [line.strip() for line in compiled.asm['ttir'].splitlines()
                        if any(anchor in line for anchor in ('tt.load', 'tt.store', 'tt.reduce', 'barrier',
                           'arith.extf', 'arith.truncf', 'arith.mulf', 'arith.addf', 'math.exp2',
                           'scf.for', 'tt.addptr'))]
                except Exception as error:
                    row.update(status='rejected', exception_type=type(error).__name__, message=str(error))
                    (path / 'error.txt').write_text(str(error) + '\n')
                receipt['results'].append(row)
                receipt['counts'] = dict(Counter(r['status'] for r in receipt['results']))
                write_receipt(receipt_path, receipt)
                print(row['name'], row['status'], flush=True)
    receipt['inputs_after'] = {str(p.resolve()): file_record(p) for p in captured}
    receipt['inputs_unchanged'] = receipt['inputs_before'] == receipt['inputs_after']
    receipt['matrix_complete'] = len(receipt['results']) == receipt['expected_cases']
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_receipt(receipt_path, receipt)
    print(json.dumps(receipt['counts']), flush=True)
    if not receipt['inputs_unchanged'] or not receipt['matrix_complete']:
        return 2
    # Expected source-inadmissible profiles still make this an admission failure;
    # completion of the observation matrix is reported separately in the receipt.
    return int(any(row['status'] == 'rejected' for row in receipt['results']))


if __name__ == '__main__':
    raise SystemExit(main())
