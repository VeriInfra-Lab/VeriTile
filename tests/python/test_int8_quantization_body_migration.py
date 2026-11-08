import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check, check_statement_function

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/int8_quantization'


class Int8QuantizationBodyMigration(unittest.TestCase):
    def setUp(self):
        self.rows = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_two_complete_bodies(self):
        results = check_pinned_pairs(ROOT, CASE)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body'
                            and r['statements'] == 23 for r in results))

    def test_lost_cast_constant_or_safety_operation_is_rejected(self):
        row = self.rows[0]
        python = (ROOT / row['python']).read_text()
        mutations = [
            ('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
            ('tl.arange(0, WIDTH).to(tl.int64)', 'tl.arange(0, WIDTH)'),
            ('C**-0.5 * 1.44269504', 'factor'),
            ('tl.where(finite, x, 0.0)', 'x'),
            ('tl.minimum(tl.maximum(normalized, -127.0), 127.0)', 'normalized'),
            ('biased.to(tl.int8)', 'biased'),
            ('mask=active', 'mask=True'),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    check_statement_function(python, self.lean.replace(before, after),
                                             row['python_function'], row['lean_definition'])

    def test_changed_storage_type_is_rejected(self):
        row = self.rows[0]
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), self.lean.replace('("X_int8",.i8)', '("X_int8",.i32)'),
                  row['python_function'], row['lean_definition'], row['signature_definition'],
                  row['expected_signature'], row['expected_call_bindings'])

    def test_candidate_changes_only_half_sign_selection(self):
        meta = variant_provenance(CASE)
        sources = {}
        for key in ['original', 'candidate']:
            sources[key] = variant_source(CASE / meta[key])
            self.assertEqual(hashlib.sha256(sources[key].encode()).hexdigest(), meta[key + '_sha256'])
        trees = {key: ast.parse(source) for key, source in sources.items()}
        candidate = next(n for n in trees['candidate'].body if isinstance(n, ast.FunctionDef)
                         and n.name == '_per_block_int8_repaired')
        biased = next(n for n in candidate.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'biased' for t in n.targets))
        expected = ast.parse('biased = normalized + tl.where(normalized >= 0.0, 0.5, -0.5)').body[0]
        self.assertEqual(ast.dump(biased), ast.dump(expected))
        replacement = ast.parse('biased = normalized + 0.5 * tl.where(normalized >= 0.0, 1.0, -1.0)').body[0]
        candidate.body[candidate.body.index(biased)] = replacement
        self.assertEqual(ast.dump(ast.Module(body=trees['candidate'].body[1:], type_ignores=[])),
                         ast.dump(ast.Module(body=trees['original'].body[1:], type_ignores=[])))
