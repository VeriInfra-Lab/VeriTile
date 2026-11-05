import ast
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/decay_cumsum'


class DecayCumsumBodyMigrationTests(unittest.TestCase):
    def records(self):
        return pinned_pairs(CASE)

    def test_all_originals_and_existing_candidates(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in rows], [8, 8, 11, 11, 15, 15])
        for row in rows:
            self.assertEqual(row['runtime_parameters']['scale'], 'fp32')
            self.assertEqual(row['constexpr_parameters'], ['BK', 'BT', 'DK'])

    def test_source_casts_masks_reverse_bounds_and_final_pointer_updates(self):
        for r in self.records():
            python = (ROOT / r['python']).read_text()
            lean = (ROOT / r['lean']).read_text()
            start = lean.index('def ' + r['lean_definition'] + ' ')
            end = lean.find('\ndef ', start + 1)
            end = len(lean) if end == -1 else end
            body = lean[start:end]
            mutants = [('.to(tl.int64)', ''), ('.to(tl.float32)', ''),
                       ('range(tl.cast(BT, tl.int64))', 'range(BT)'),
                       ('tl.cast(-1, tl.int64)', 'tl.cast(0, tl.int64)'),
                       (', -1):', ', 1):'), ('mask=mask', 'mask=True'),
                       ('p_go += DK', 'p_go += 1'), ('p_qg += DK', 'p_qg += 1'),
                       ('p_dg -= DK', 'p_dg -= 1'), ('if j == (BT-1):', 'if j == BT:')]
            checked = 0
            for before, after in mutants:
                if before not in body:
                    continue
                checked += 1
                changed = lean[:start] + body.replace(before, after, 1) + lean[end:]
                with self.subTest(body=r['lean_definition'], mutation=before), self.assertRaises(ValueError):
                    check(python, changed, r['python_function'], r['lean_definition'],
                          r['signature_definition'], r['expected_signature'],
                          expected_call_bindings=r['expected_call_bindings'])
            self.assertGreaterEqual(checked, 5)

    def test_float_module_constant_cannot_drift(self):
        r = self.records()[0]
        python = (ROOT / r['python']).read_text().replace('tl.constexpr(1.44269504)',
                                                         'tl.constexpr(1.44269505)')
        with self.assertRaisesRegex(ValueError, 'Python module literal'):
            check(python, (ROOT / r['lean']).read_text(), r['python_function'], r['lean_definition'],
                  r['signature_definition'], r['expected_signature'],
                  expected_call_bindings=r['expected_call_bindings'])

    def test_candidate_only_commutes_the_three_mask_products(self):
        original = ast.parse(variant_source(CASE / 'decay_cumsum.py'))
        candidate = ast.parse(variant_source(CASE / 'decay_cumsum_index_commuted.py'))
        restored = 0
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name.endswith('_index_commuted'):
                node.name = node.name.removesuffix('_index_commuted')
            if isinstance(node, ast.Name) and node.id.endswith('_index_commuted'):
                node.id = node.id.removesuffix('_index_commuted')
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'mask'
                                                  for t in node.targets):
                product = node.value.left.left
                self.assertEqual(ast.unparse(product), 'BK * i_k')
                product.left, product.right = product.right, product.left
                restored += 1
        self.assertEqual(restored, 3)
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(CASE)
        for field in ['original', 'candidate']:
            self.assertEqual(hashlib.sha256(variant_source(CASE / provenance[field]).encode()).hexdigest(),
                             provenance[field + '_sha256'])
