"""Actual delegated FastCE bodies and the existing integer-only candidates."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('bench/repairs/tritonbench_g/fast_ce_loss')

class FastCeFunctionBodyTests(unittest.TestCase):
    def records(self):
        return json.loads((ROOT / BASE / 'jit-correspondence.json').read_text())

    def test_all_six_complete_bodies(self):
        pairs = check_manifest(ROOT, ROOT / BASE / 'jit-correspondence.json')
        self.assertEqual([p['statements'] for p in pairs], [17] * 6)
        self.assertTrue(all(p['mode'] == 'pasted_body' for p in pairs))

    def test_extra_casts_or_removed_original_casts_are_rejected(self):
        record = self.records()[0]
        python = (ROOT / record['python']).read_text()
        lean = (ROOT / record['lean']).read_text()
        for before, after in [('tl.store(loss_ptr, loss)', 'tl.store(loss_ptr, loss.to(tl.float32))'),
                              ('logits_row_stride.to(tl.int64)', 'logits_row_stride'),
                              ('if DO_SOFTCAPPING:', 'if True:')]:
            self.assertIn(before, lean)
            with self.subTest(change=before), self.assertRaises(ValueError):
                check(python, lean.replace(before, after), **{
                    k: v for k, v in record.items() if k not in ['python', 'lean']})

    def test_alias_target_and_runtime_stride_are_checked(self):
        record = self.records()[4]
        python = (ROOT / record['python']).read_text()
        lean = (ROOT / record['lean']).read_text()
        kwargs = {k: v for k, v in record.items() if k not in ['python', 'lean']}
        with self.assertRaises(ValueError):
            check(python.replace('triton_tanh = libdevice.tanh', 'triton_tanh = libdevice.cos'), lean, **kwargs)
        with self.assertRaises(ValueError):
            check(python, lean.replace('dloss_row_stride := dloss_row_stride',
                                      'dloss_row_stride := logits_row_stride'), **kwargs)

    def test_candidates_only_commute_the_existing_row_product(self):
        source = ast.parse((ROOT / BASE / 'fast_ce_loss_body.py').read_text())
        candidate = ast.parse((ROOT / BASE / 'candidate_kernels.py').read_text())
        for record in self.records()[::2]:
            expected = copy.deepcopy(next(n for n in source.body if isinstance(n, ast.FunctionDef)
                                          and n.name == record['python_function']))
            expected.name += '_index_commuted'
            pointer = next(n for n in expected.body if isinstance(n, ast.AugAssign)
                           and isinstance(n.target, ast.Name) and n.target.id == 'logits_ptr')
            self.assertEqual(ast.unparse(pointer.value), 'row_idx * logits_row_stride.to(tl.int64)')
            pointer.value = ast.parse('logits_row_stride.to(tl.int64) * row_idx', mode='eval').body
            actual = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == expected.name)
            self.assertEqual(ast.dump(expected, include_attributes=False), ast.dump(actual, include_attributes=False))
