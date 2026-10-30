"""Full delegated CrossEntropyOps sources and integer-only candidates."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest, scalar_binding_term

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('bench/repairs/tritonbench_g/cross_entropy_ops')

class CrossEntropyOpsFunctionBodyTests(unittest.TestCase):
    def records(self):
        return json.loads((ROOT / BASE / 'jit-correspondence.json').read_text())

    def test_four_complete_bodies(self):
        pairs = check_manifest(ROOT, ROOT / BASE / 'jit-correspondence.json')
        self.assertEqual([p['statements'] for p in pairs], [14, 14, 16, 16])
        self.assertTrue(all(p['mode'] == 'pasted_body' for p in pairs))

    def test_original_casts_branches_and_unused_assignment_are_required(self):
        record = self.records()[2]
        python = (ROOT / record['python']).read_text()
        lean = (ROOT / record['lean']).read_text()
        for before, after in [('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
                              ('logits_row_stride.to(tl.int64)', 'logits_row_stride'),
                              ('smooth_positive = 1.0 - smoothing', 'smooth_positive = 0.0'),
                              ('if not is_ignored:', 'if True:')]:
            self.assertIn(before, lean)
            with self.subTest(change=before), self.assertRaises(ValueError):
                check(python, lean.replace(before, after), **{
                    k: v for k, v in record.items() if k not in ['python', 'lean']})

    def test_call_and_offset_bindings_are_checked(self):
        record = self.records()[0]
        python = (ROOT / record['python']).read_text()
        lean = (ROOT / record['lean']).read_text()
        kwargs = {k: v for k, v in record.items() if k not in ['python', 'lean']}
        for before, after in [('logits_ptr := p.logitsBase', 'logits_ptr := p.lseBase'),
                              ('p.totalClasses', 'p.cols')]:
            with self.subTest(binding=before), self.assertRaises(ValueError):
                check(python, lean.replace(before, after), **kwargs)

    def test_integer_call_annotation_is_retained_and_bounded(self):
        self.assertEqual(scalar_binding_term('( p.totalClasses : Int )'), '(p.totalClasses : Int)')
        self.assertNotEqual(scalar_binding_term('(p.totalClasses : Int)'),
                            scalar_binding_term('(p.totalClasses : Nat)'))
        self.assertNotEqual(scalar_binding_term('(p.totalClasses : Int)'),
                            scalar_binding_term('p.totalClasses'))
        for value in ['(f(p.cols) : Int)', '(p.cols : Real)', '(p.cols; bad : Int)',
                      '(p.cols : Int); bad', '(p.cols[0] : Int)']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scalar_binding_term(value)

    def test_candidates_only_commute_column_product(self):
        source = ast.parse((ROOT / BASE / 'cross_entropy_ops_conditional.py').read_text())
        candidate = ast.parse((ROOT / BASE / 'candidate_kernels.py').read_text())
        for record in self.records()[::2]:
            expected = copy.deepcopy(next(n for n in source.body if isinstance(n, ast.FunctionDef)
                                          and n.name == record['python_function']))
            expected.name += '_index_commuted'
            offset = next(n for n in expected.body if isinstance(n, ast.Assign)
                          and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'col_offsets')
            self.assertEqual(ast.unparse(offset.value.left), 'col_block_idx * BLOCK_SIZE')
            offset.value.left = ast.parse('BLOCK_SIZE * col_block_idx', mode='eval').body
            actual = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == expected.name)
            self.assertEqual(ast.dump(expected, include_attributes=False), ast.dump(actual, include_attributes=False))
