#!/usr/bin/env python3
"""Compare all 216 backward storage quotations with the complete original AST.

Keep every LOG, CAUSAL and IS_FP16 branch. Normalize only direct parameter
antiquotations, block delimiters and the checked destination store conversion.
This checks source syntax; native compilation, numerical refinement, proof
replay and public-port acceptance remain separate obligations.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source
from bench.source_control_materializations import _brace_body

PORT = ROOT / 'bench/tritonbench_g/ksoftmax_triton'
ENTRY = PORT / 'Proofs/BackwardSource.lean'
PYTHON = PORT / 'ksoftmax_triton.py'
PYTHON_SHA256 = '8f392086c34ec0cd665c31e614d1534fa2899fba0beecc6435cc06e2095d0e66'
DTYPES = ('float32', 'float64', 'float16', 'bfloat16', 'float8e4nv', 'float8e5')
PARAMETERS = ('GradIn', 'GradOut', 'Out', 'stride_bm', 'stride_bn',
              'stride_gm', 'stride_gn', 'stride_om', 'stride_on', 'K',
              'LOG', 'CAUSAL', 'DEPTH', 'IS_FP16')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def original_body(text):
    kernels = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
               and n.name == '_softmax_backward']
    if len(kernels) != 1:
        raise ValueError('Expected exactly one original backward JIT')
    kernel = kernels[0]
    if tuple(a.arg for a in kernel.args.args) != PARAMETERS:
        raise ValueError('Original signature differs')
    body = list(kernel.body)
    if not (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        raise ValueError('Expected the original documentation string')
    return ast.unparse(ast.Module(body=body[1:], type_ignores=[]))


def check_body(original, lean):
    stores = source.store_calls_with_explicit_destination_casts(original, lean)
    if len(stores) != 1:
        raise ValueError('Expected one destination store')
    expected = ast.parse(original)
    final = expected.body[-1]
    if not (isinstance(final, ast.Expr) and isinstance(final.value, ast.Call)
            and ast.unparse(final.value.func) == 'tl.store'):
        raise ValueError('Original final store moved')
    expected.body[-1] = ast.parse(stores[0]).body[0]

    def parameter(match):
        if match[1] not in PARAMETERS:
            raise ValueError('Non-parameter antiquotation: ' + match[1])
        return match[1]

    actual = source.strip_region_declarations(lean)
    actual = re.sub(r'\$\((\w+)\)', parameter, actual)
    if '$' in actual:
        raise ValueError('Unresolved source antiquotation')
    actual = _brace_body(actual)
    if ast.dump(actual, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Complete original expression/control AST differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {ENTRY, PYTHON, Path(__file__).resolve(), ROOT / 'bench/audit_source.py',
              ROOT / 'bench/source_control_materializations.py'}
    captured = {p: p.read_bytes() for p in sorted(inputs)}
    for path, data in captured.items():
        target = output / 'sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    original = captured[PYTHON].decode()
    if digest(original.encode()) != PYTHON_SHA256:
        raise ValueError('Original Python pin changed; review before updating')
    original = original_body(original)
    rows = []
    for gradient, saved, destination in itertools.product(DTYPES, repeat=3):
        selection = (f'gradientType=tl.{gradient}, savedType=tl.{saved}, '
                     f'destinationType=tl.{destination}, version=original')
        view, count = re.subn(r'^Source-audit args: .*$', 'Source-audit args: ' + selection,
                             captured[ENTRY].decode(), flags=re.M)
        if count != 1:
            raise ValueError('Expected exactly one source selector')
        selected = source.select_lean_audit_source(view)
        lean = source.lean_first_triton_body(selected)
        regions = source.declared_region_dtypes(lean)
        if regions != {'GradOut': {'tl.' + gradient}, 'Out': {'tl.' + saved},
                       'GradIn': {'tl.' + destination}}:
            raise ValueError('Selected pointer storage differs')
        check_body(original, lean)
        wrong_destination = 'float16' if destination == 'float32' else 'float32'
        mutations = [
            ('m * $(stride_gm)', 'm + $(stride_gm)'),
            ('(zero).to(g.dtype)', '(zero).to(tl.float32)'),
            ('k > n', 'k >= n'),
            ('if IS_FP16', 'if CAUSAL'),
            ('g - tl.exp(o) * s', 'g + tl.exp(o) * s'),
            ('tl.sum(g * o, 0)', 'tl.sum(g + o, 0)'),
            ('mask=k < $(K)', 'mask=io_mask'),
            (f'(grad_in).to(tl.{destination})', f'(grad_in).to(tl.{wrong_destination})'),
        ]
        for before, after in mutations:
            if before not in lean:
                raise ValueError('Negative-control anchor disappeared: ' + before)
            try:
                check_body(original, lean.replace(before, after, 1))
            except ValueError:
                pass
            else:
                raise ValueError('Source mutation was accepted: ' + before)
        name = '-'.join((gradient, saved, destination))
        target = output / 'profiles' / (name + '.txt')
        target.parent.mkdir(exist_ok=True)
        target.write_text(lean)
        rows.append({'gradient': gradient, 'saved': saved, 'destination': destination,
                     'full_ast_match': True, 'negative_controls_rejected': len(mutations),
                     'source_body_sha256': digest(lean.encode())})
    changed = [str(p.relative_to(ROOT)) for p, data in captured.items() if p.read_bytes() != data]
    receipt = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': __doc__, 'success': not changed, 'profiles': rows,
               'changed_inputs': changed, 'complete_task': False,
               'native_admission': False, 'independent_replay': False,
               'input_sha256': {str(p.relative_to(ROOT)): digest(data) for p, data in captured.items()}}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'success': not changed, 'profiles': len(rows),
                      'negative_controls': sum(r['negative_controls_rejected'] for r in rows),
                      'receipt': str(output / 'receipt.json')}))
    return 0 if not changed else 1


if __name__ == '__main__':
    raise SystemExit(main())
