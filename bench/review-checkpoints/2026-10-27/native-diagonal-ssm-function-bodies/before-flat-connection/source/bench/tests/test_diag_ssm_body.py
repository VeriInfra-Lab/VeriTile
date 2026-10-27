"""Full source correspondence for the four real/complex SSM functions."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest, quotation

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('bench/repairs/tritonbench_g/diag_ssm_triton')

class DiagonalSsmFunctionBodyTests(unittest.TestCase):
    def records(self):
        return json.loads((ROOT / BASE / 'jit-correspondence.json').read_text())

    def test_all_eight_complete_bodies(self):
        pairs = check_manifest(ROOT, ROOT / BASE / 'jit-correspondence.json')
        self.assertEqual([p['statements'] for p in pairs], [10, 10, 13, 13, 12, 12, 18, 18])
        self.assertTrue(all(p['mode'] == 'pasted_body' for p in pairs))

    def test_casts_masks_negation_and_updates_cannot_drift(self):
        seen = set()
        for record in self.records():
            python = (ROOT / record['python']).read_text()
            lean = (ROOT / record['lean']).read_text()
            body = quotation(lean, record['lean_definition'])[1]
            for before, after in [('tl.cast(length, tl.int64)', 'length'),
                                  ('tl.cast(batch_size, tl.int64)', 'batch_size'),
                                  ('tl.cast(dim, tl.int64)', 'dim'),
                                  ('mask=mask', 'mask=True'),
                                  ('-tl.load(', 'tl.load('),
                                  ('+=', '=')]:
                if before not in body:
                    continue
                seen.add(before)
                with self.subTest(function=record['python_function'], change=before), self.assertRaises(ValueError):
                    check(python, lean.replace(before, after), **{
                        k: v for k, v in record.items() if k not in ['python', 'lean']})
        self.assertEqual(len(seen), 6)

    def test_runtime_bindings_and_complex_component_storage_are_checked(self):
        record = self.records()[6]
        python = (ROOT / record['python']).read_text()
        lean = (ROOT / record['lean']).read_text()
        for before, after in [('dim := dim', 'dim := batch_size'),
                              ('("dim", .integer)', '("dim", .i32)'),
                              ('("grad_y_ptr", .fp32)', '("grad_y_ptr", .fp16)')]:
            self.assertIn(before, lean)
            with self.subTest(change=before), self.assertRaises(ValueError):
                check(python, lean.replace(before, after), **{
                    k: v for k, v in record.items() if k not in ['python', 'lean']})

    def test_candidates_only_commute_the_existing_integer_index_product(self):
        source = ast.parse((ROOT / BASE / 'diag_ssm_triton_repaired.py').read_text())
        candidate = ast.parse((ROOT / BASE / 'candidate_kernels.py').read_text())
        for record in self.records()[::2]:
            expected = copy.deepcopy(next(n for n in source.body if isinstance(n, ast.FunctionDef)
                                          and n.name == record['python_function']))
            expected.name += '_index_commuted'
            indices = [n for n in expected.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'col_idx' for t in n.targets)]
            self.assertEqual(len(indices), 1)
            self.assertEqual(ast.unparse(indices[0].value), 'tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE')
            indices[0].value = ast.parse('BLOCK_SIZE * tl.cast(tl.program_id(0), tl.int64)', mode='eval').body
            actual = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == expected.name)
            self.assertEqual(ast.dump(expected, include_attributes=False), ast.dump(actual, include_attributes=False))
