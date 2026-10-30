"""Complete decay phases, exact immutable globals, casts and existing candidates."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest, check_source_constants
ROOT = Path(__file__).resolve().parents[2]
BASE = Path('bench/repairs/tritonbench_g/decay_cumsum')

class DecayCumsumFunctionBodyTests(unittest.TestCase):
    def records(self):
        return json.loads((ROOT / BASE / 'jit-correspondence.json').read_text())
    def test_six_complete_bodies(self):
        rows = check_manifest(ROOT, ROOT / BASE / 'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in rows], [8, 8, 11, 11, 15, 15])
        self.assertTrue(all(r['mode'] == 'pasted_body' for r in rows))
    def test_casts_masks_loop_bounds_and_unconverted_stores_are_required(self):
        record = self.records()[4]
        python = (ROOT / record['python']).read_text(); lean = (ROOT / record['lean']).read_text()
        for old, new in [('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
                         ('tl.cast(BT - 1, tl.int64)', 'BT - 1'),
                         ('tl.cast(-1, tl.int64)', '0'),
                         ('tl.store(p_dq_inter, _dq,', 'tl.store(p_dq_inter, _dq.to(tl.float32),'),
                         ('if j == (BT-1):', 'if j == 0:'),
                         ('cum_grad_dg.to(p_dg.dtype.element_ty)', 'cum_grad_dg')]:
            self.assertIn(old, lean)
            with self.subTest(change=old), self.assertRaises(ValueError):
                check(python, lean.replace(old, new), **{k:v for k,v in record.items() if k not in ('python','lean')})
    def test_candidate_changes_only_the_mask_integer_product(self):
        original = ast.parse((ROOT / BASE / 'decay_cumsum_repaired.py').read_text())
        candidate = ast.parse((ROOT / BASE / 'candidate_kernels.py').read_text())
        for record in self.records()[::2]:
            expected = copy.deepcopy(next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == record['python_function']))
            expected.name += '_index_commuted'
            mask = next(n for n in expected.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'mask')
            self.assertEqual(ast.unparse(mask.value.left.left), 'i_k * BK')
            mask.value.left.left = ast.parse('BK * i_k', mode='eval').body
            actual = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == expected.name)
            self.assertEqual(ast.dump(expected), ast.dump(actual))
    def test_source_constant_metadata_is_checked(self):
        record = self.records()[0]; python=(ROOT / record['python']).read_text(); lean=(ROOT / record['lean']).read_text()
        kwargs={k:v for k,v in record.items() if k not in ('python','lean')}
        with self.assertRaises(ValueError):check(python, lean.replace('"1.44269504"', '"1.0"'), **kwargs)
        with self.assertRaises(ValueError):check(python.replace('tl.constexpr(1.44269504)','tl.constexpr(1.0)'), lean, **kwargs)
    def check_global(self, source, bindings=None):
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
        check_source_constants(source,function,{'C':'1.44269504'} if bindings is None else bindings)
    def test_unique_literal_global(self):
        self.check_global('C = tl.constexpr(1.44269504)\ndef kernel(x):\n    return x * C\n')
    def test_missing_global_binding_is_rejected(self):
        with self.assertRaises(ValueError):self.check_global('C = tl.constexpr(1.44269504)\ndef kernel(x):\n    return x * C\n',{})
    def test_computed_global_is_rejected(self):
        for value in ['f()', '1.0 + 0.44269504', 'OTHER', 'True']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check_global('C = tl.constexpr('+value+')\ndef kernel(x):\n    return x * C\n')
    def test_global_rebinding_and_shadowing_are_rejected(self):
        prefix='C = tl.constexpr(1.44269504)\n'
        for suffix in ['C = 0\ndef kernel(x):\n    return x * C\n',
                       'if flag:\n    C = 0\ndef kernel(x):\n    return x * C\n',
                       'from other import C\ndef kernel(x):\n    return x * C\n',
                       'def kernel(C):\n    return C\n',
                       'def kernel(x):\n    C = 0\n    return x * C\n',
                       'def kernel(x):\n    for C in range(2):\n        x *= C\n    return x\n',
                       'del C\ndef kernel(x):\n    return x * C\n']:
            with self.subTest(source=suffix), self.assertRaises(ValueError):self.check_global(prefix+suffix)
