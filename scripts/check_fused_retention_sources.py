#!/usr/bin/env python3
"""Compare both complete retention quotations with the pinned Python AST.

Normalize direct parameter/literal antiquotes, source block notation, the
independent program-ID tuple, range defaults, the unused loop binder, and
the explicit `do`/`do_` parameter renaming. Preserve every expression, flag,
cast, memory access, loop, pointer update and barrier. This is source evidence,
not execution, native admission, numerical refinement or independent replay.
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
from scripts.check_reversed_cumsum_sources import Canonical, braces

PORT = ROOT / 'bench/tritonbench_g/fused_recurrent_retention'
PYTHON = PORT / 'fused_recurrent_retention.py'
LEAN = PORT / 'Kernels.lean'
PIN = '369709736fce34c980e29fbf8db9b2117eaa5b22d65acad20ea3a898956426f5'
DTYPES = ('float32', 'float64', 'float16', 'bfloat16', 'float8e4nv', 'float8e5')
COMMON = ('s_qk_h', 's_qk_t', 's_qk_d', 's_vo_h', 's_vo_t', 's_vo_d',
          'B', 'H', 'T', 'scale', 'BK', 'BV', 'DK', 'DV', 'USE_INITIAL_STATE')
PROFILES = {
    'forward': ('fused_recurrent_retention_fwd_kernel', 'retentionForwardSource%',
                ('q', 'k', 'v', 'o', 'initial_state', 'final_state'),
                ('qTy', 'kTy', 'vTy', 'oTy', 'seedTy', 'finalTy'),
                {'o', 'final_state'}),
    'backward': ('fused_recurrent_retention_bwd_kernel', 'retentionBackwardSource%',
                 ('q', 'k', 'v', 'do', 'dq', 'dk', 'dv', 'initial_state'),
                 ('qTy', 'kTy', 'vTy', 'doTy', 'dqTy', 'dkTy', 'dvTy', 'seedTy'),
                 {'dq', 'dk', 'dv'}),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class RetentionCanonical(Canonical):
    def visit_Name(self, node):
        if node.id == 'do_':
            node.id = 'do'
        return super().visit_Name(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == '_':
            if any(isinstance(n, ast.Name) and n.id in ('_', '_t')
                   for statement in node.body for n in ast.walk(statement)):
                raise ValueError('The original unused binder is read or captured')
            node.target.id = '_t'
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = super().visit_Call(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'range':
            if node.keywords or not 1 <= len(node.args) <= 3:
                raise ValueError('Unsupported range call')
            if len(node.args) == 1:
                node.args.insert(0, ast.Constant(0))
            if len(node.args) == 2:
                node.args.append(ast.Constant(1))
        return node


def check(original, lean, parameters):
    # Resolve both source spellings through checked pointer provenance.
    expected = source.resolve_source_dtype_attributes(original, lean)
    actual = source.resolve_source_dtype_attributes(lean, lean)
    actual = source.strip_region_declarations(actual)

    def direct(match):
        value = match[1]
        if value not in parameters and value not in ('0', '1'):
            raise ValueError('Non-direct parameter/literal antiquotation: ' + value)
        return value

    actual = re.sub(r'\$\((\w+)\)', direct, actual)
    if '$' in actual:
        raise ValueError('Unresolved source splice')
    actual = re.sub(r'\+\s*\n\s*', '+ ', actual)
    left = RetentionCanonical().visit(ast.parse(expected))
    right = RetentionCanonical().visit(braces(actual))
    if ast.dump(left, include_attributes=False) != ast.dump(right, include_attributes=False):
        raise ValueError('Complete expression/control AST differs\nEXPECTED ' +
                         ast.unparse(ast.fix_missing_locations(left)) + '\nACTUAL ' +
                         ast.unparse(ast.fix_missing_locations(right)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [PYTHON, LEAN, Path(__file__).resolve(), ROOT / 'bench/audit_source.py',
             ROOT / 'scripts/check_reversed_cumsum_sources.py',
             ROOT / 'scripts/check_reversed_cumsum_native.py']
    captured = {p: p.read_bytes() for p in paths}
    if digest(captured[PYTHON]) != PIN:
        raise ValueError('Pinned original Python changed')
    tree = ast.parse(captured[PYTHON])
    rows = []
    for kind, (function, macro, buffers, arguments, outputs) in PROFILES.items():
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
        if len(functions) != 1:
            raise ValueError('Expected one original JIT: ' + function)
        fn = functions[0]
        parameters = buffers + COMMON + (('STORE_FINAL_STATE',) if kind == 'forward' else ())
        if tuple(a.arg for a in fn.args.args) != parameters:
            raise ValueError('Original signature changed')
        original = ast.unparse(ast.Module(body=fn.body, type_ignores=[]))
        for input_type, output_type in itertools.product(DTYPES, repeat=2):
            regions = {('do_' if b == 'do' else b): {'tl.' + (output_type if b in outputs else input_type)}
                       for b in buffers}
            values = [arg + '=tl.' + (output_type if b in outputs else input_type)
                      for b, arg in zip(buffers, arguments)]
            metadata = ('/-\nSource-audit Lean: ' + macro + '\nSource-audit args: ' +
                        ', '.join(values + ['version=original']) + '\n-/\n')
            selected = source.select_lean_audit_source(captured[LEAN].decode(), metadata)
            lean = source.lean_first_triton_body(selected)
            if source.declared_region_dtypes(lean) != regions:
                raise ValueError('Selected storage declarations differ')
            params = {*parameters, 'do_'}
            check(original, lean, params)
            mutations = [
                ('i_bh % $(H)', 'i_bh + $(H)'),
                ('if USE_INITIAL_STATE', 'if STORE_FINAL_STATE'),
                ('range($(0), $(T), $(1))', 'range($(1), $(T), $(1))'),
                ('mask=mask_bk, other=0', 'mask=mask_bv, other=0'),
                ('b_b * h + _k', 'b_b + h + _k'),
                ('p_k += $(DK)', 'p_k += $(DV)'),
                ('tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)',
                 'tl.load(p_k, mask=mask_bk, other=0).to(tl.float16)'),
            ]
            if kind == 'forward':
                mutations += [('if STORE_FINAL_STATE', 'if USE_INITIAL_STATE'),
                              ('(h).to(p_final_s.dtype.element_ty)', 'h'),
                              ('mask=mask_kv)', 'mask=mask_bk)')]
            else:
                mutations += [('tl.debug_barrier()', ''),
                              ('($(T) - 1) * $(DK)', '($(T) + 1) * $(DK)'),
                              ('p_dk -= $(DK)', 'p_dk += $(DK)'),
                              ('d_h *= b_b', 'd_h += b_b'),
                              ('(d_v).to(p_dv.dtype.element_ty)', 'd_v')]
            for before, after in mutations:
                if before not in lean:
                    raise ValueError('Missing negative-control anchor: ' + before)
                try:
                    check(original, lean.replace(before, after, 1), params)
                except (ValueError, SyntaxError):
                    pass
                else:
                    raise ValueError('Source mutation accepted: ' + before)
            target = args.output / 'profiles' / f'{kind}-{input_type}-{output_type}.txt'
            target.parent.mkdir(exist_ok=True)
            target.write_text(lean)
            rows.append({'kernel': kind, 'input_type': input_type, 'output_type': output_type,
                         'full_ast_match': True, 'negative_controls_rejected': len(mutations),
                         'source_body_sha256': digest(lean.encode())})
    for path, raw in captured.items():
        target = args.output / 'sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    changed = [str(p.relative_to(ROOT)) for p, raw in captured.items() if p.read_bytes() != raw]
    receipt = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
               'success': not changed, 'profiles': rows, 'changed_inputs': changed,
               'complete_task': False, 'native_admission': False, 'independent_replay': False,
               'input_sha256': {str(p.relative_to(ROOT)): digest(b) for p, b in captured.items()}}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'success': not changed, 'profiles': len(rows),
                      'negative_controls': sum(r['negative_controls_rejected'] for r in rows),
                      'receipt': str(args.output / 'receipt.json')}))
    return int(bool(changed))


if __name__ == '__main__':
    raise SystemExit(main())
