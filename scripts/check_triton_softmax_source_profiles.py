#!/usr/bin/env python3
"""Check all 36 floating source views against the complete original softmax AST.

Only direct parameter antiquotations and explicit destination store conversions
are normalized. This is source correspondence, not native profile admission,
Lean proof replay, or a claim about the compiler's numerical behavior.
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
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import audit_source as source
from bench.port_layout import source_text

PORT = ROOT / 'bench/tritonbench_g/triton_softmax'
DTYPES = ('float32', 'float64', 'float16', 'bfloat16', 'float8e4nv', 'float8e5')
PYTHON_SHA256 = 'c49fa6d8ad7c0f7fa54c2a389ab7506e5c90101e72e178cd0eaca50f890812b8'
PARAMETERS = ('output_ptr', 'input_ptr', 'input_row_stride', 'output_row_stride',
              'n_cols', 'BLOCK_SIZE')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_body(python_text, selected):
    tree = ast.parse(python_text)
    kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == 'softmax_kernel']
    if len(kernels) != 1:
        raise ValueError('Expected the one original softmax JIT')
    kernel = kernels[0]
    if tuple(a.arg for a in kernel.args.args) != PARAMETERS:
        raise ValueError('Kernel signature changed')
    lean = source.lean_first_triton_body(selected)
    python_body = ast.unparse(ast.Module(body=kernel.body, type_ignores=[]))
    stores = source.store_calls_with_explicit_destination_casts(python_body, lean)
    if len(stores) != 1:
        raise ValueError('Expected exactly one original output store')
    expected = ast.parse(python_body)
    if not (isinstance(expected.body[-1], ast.Expr)
            and isinstance(expected.body[-1].value, ast.Call)
            and ast.unparse(expected.body[-1].value.func) == 'tl.store'):
        raise ValueError('Store must remain the final source statement')
    expected.body[-1] = ast.parse(stores[0]).body[0]

    def parameter(match):
        if match[1] not in PARAMETERS:
            raise ValueError('Non-parameter source antiquotation: ' + match[1])
        return match[1]

    lean = source.strip_region_declarations(lean)
    lean = re.sub(r'\$\((\w+)\)', parameter, lean)
    if '$' in lean:
        raise ValueError('Unresolved source antiquotation')
    # Macro statement splices can have different indentation. This kernel is
    # straight-line; flattening indentation cannot admit a new guarded branch.
    actual = ast.parse('\n'.join(line.lstrip() for line in lean.splitlines()))
    if ast.dump(expected, include_attributes=False) != ast.dump(actual, include_attributes=False):
        raise ValueError('Complete original statement/expression AST differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory retaining frozen inputs, profiles and logs')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {PORT / 'Kernels.lean', PORT / 'triton_softmax.py', Path(__file__).resolve(),
              ROOT / 'bench/audit_tritonbench_g.sh'}
    inputs |= set((ROOT / 'bench').glob('*.py'))
    inputs |= set(PORT.rglob('*.lean'))
    inputs |= set((ROOT / 'bench/tritonbench_g').glob('source_*profiles.json'))
    captured = {p: p.read_bytes() for p in sorted(inputs)}
    original = captured[PORT / 'triton_softmax.py'].decode()
    if digest(original.encode()) != PYTHON_SHA256:
        raise ValueError('Original Python source changed; review it before updating its pin')
    frozen = output / 'repo'
    for path, data in captured.items():
        target = frozen / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    profiles = output / 'profiles'
    results = []
    for input_type, output_type in itertools.product(DTYPES, repeat=2):
        name = input_type + '-' + output_type
        directory = profiles / name
        directory.mkdir(parents=True)
        text = captured[PORT / 'Kernels.lean'].decode()
        selector = f'inputType=tl.{input_type}, outputType=tl.{output_type}, version=original'
        text, changes = re.subn(r'^Source-audit args: .*$', 'Source-audit args: ' + selector,
                               text, flags=re.M)
        if changes != 1:
            raise ValueError('Expected one source selector')
        entry = directory / 'Kernels.lean'
        for path, data in captured.items():
            if path.is_relative_to(PORT) and path.suffix == '.lean':
                target = directory / path.relative_to(PORT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        entry.write_text(text)
        (directory / 'triton_softmax.py').write_text(original)
        selected = source_text(entry)
        regions = source.declared_region_dtypes(source.lean_first_triton_body(selected))
        if regions != {'input_ptr': {'tl.' + input_type}, 'output_ptr': {'tl.' + output_type}}:
            raise ValueError('Source pointer storage differs from selected profile')
        check_body(original, selected)
        # Full-expression checks must catch mutations invisible to call inventories.
        for before, after in [('numerator / denominator', 'numerator * denominator'),
                              ('row - row_max', 'row + row_max'),
                              ('row_idx * $(output_row_stride)', 'row_idx + $(output_row_stride)')]:
            if before not in selected:
                raise ValueError('Negative-control anchor disappeared: ' + before)
            try:
                check_body(original, selected.replace(before, after))
            except ValueError:
                pass
            else:
                raise ValueError('Source mutation was accepted: ' + before)
        results.append({'input': input_type, 'output': output_type, 'full_ast_match': True,
                        'negative_controls_rejected': 3, 'selected_sha256': digest(selected.encode())})

    script = captured[ROOT / 'bench/audit_tritonbench_g.sh'].decode()
    section = script[script.index('# A source view must be extractable'):
                     script.index('# ---- Machine-checkable trust audit')]
    command = 'failures=0\nPORTS_ROOT="$1"\n' + section + '\nexit "$failures"\n'
    (output / 'production-source-gates.sh').write_text(command)
    run = subprocess.run(['bash', '-c', command, 'softmax-profiles', str(profiles)],
                         cwd=frozen, capture_output=True, text=True)
    (output / 'source-gates.log').write_text(run.stdout + run.stderr)
    gates = sum(line.startswith('ok ') for line in run.stdout.splitlines())
    changed = [str(p.relative_to(ROOT)) for p, data in captured.items() if p.read_bytes() != data]
    success = run.returncode == 0 and gates == 24 and not changed
    receipt = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': '36 complete original floating source ASTs and unmodified production source gates. Includes two native-rejected fp64-to-fp8 pairs as source syntax only. No native admission, proof replay or compiler correctness claim.',
        'success': success, 'profiles': results, 'mechanical_gates': gates,
        'production_exit_code': run.returncode, 'changed_inputs': changed,
        'input_sha256': {str(p.relative_to(ROOT)): digest(data) for p, data in captured.items()}}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'success': success, 'profiles': len(results), 'gates': gates,
                      'receipt': str(output / 'receipt.json')}))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
