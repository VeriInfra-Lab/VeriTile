import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/diag_ssm_triton'


class DiagSsmBodyMigrationTests(unittest.TestCase):
    def test_all_four_originals_and_candidates_preserve_typed_calls(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([row['statements'] for row in rows], [10, 10, 13, 13, 12, 12, 18, 18])
        for row in rows:
            self.assertEqual(row['runtime_parameters'],
                             {'length': 'integer', 'batch_size': 'integer', 'dim': 'integer'})
            self.assertEqual(row['constexpr_parameters'], ['BLOCK_SIZE'])

    def test_cast_loop_branch_masks_complex_update_and_signs_are_required(self):
        records = pinned_pairs(CASE)
        for record in records:
            python = (ROOT / record['python']).read_text()
            lean = (ROOT / record['lean']).read_text()
            start = lean.index('def ' + record['lean_definition'] + ' ')
            end = lean.find('\ndef ', start + 1)
            end = len(lean) if end == -1 else end
            quotation = lean[start:end]
            mutants = [('length = tl.cast(length, tl.int64)', 'length = length'),
                       ('batch_size = tl.cast(batch_size, tl.int64)', 'batch_size = batch_size'),
                       ('dim = tl.cast(dim, tl.int64)', 'dim = dim'),
                       ('for t in range(length):', 'for t in range(length - 1):'),
                       ('for i in range(length):', 'for i in range(length - 1):'),
                       ('t = length - 1 - i', 't = length - i'),
                       ('if t > 0:', 'if t >= 0:'),
                       ('mask=mask', 'mask=True'),
                       ('s_real, s_imag = new_s_real, new_s_imag',
                        's_real, s_imag = new_s_imag, new_s_real'),
                       ('-tl.load(', 'tl.load(')]
            tested = 0
            for before, after in mutants:
                if before not in quotation:
                    continue
                tested += 1
                changed = lean[:start] + quotation.replace(before, after, 1) + lean[end:]
                with self.subTest(body=record['lean_definition'], mutant=before), self.assertRaises(ValueError):
                    check(python, changed, record['python_function'], record['lean_definition'],
                          record['signature_definition'], record['expected_signature'],
                          expected_call_bindings=record['expected_call_bindings'])
            self.assertGreaterEqual(tested, 5)

    def test_candidate_changes_only_the_four_integer_index_products(self):
        original = ast.parse(variant_source(CASE / 'diag_ssm_triton.py'))
        candidate = ast.parse(variant_source(CASE / 'diag_ssm_triton_index_commuted.py'))
        restored = 0
        for node in ast.walk(candidate):
            if isinstance(node, ast.FunctionDef) and node.name.endswith('_index_commuted'):
                node.name = node.name.removesuffix('_index_commuted')
            if isinstance(node, ast.Name) and node.id.endswith('_index_commuted'):
                node.id = node.id.removesuffix('_index_commuted')
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'col_idx'
                                                  for t in node.targets):
                self.assertEqual(ast.unparse(node.value), 'BLOCK_SIZE * tl.cast(tl.program_id(0), tl.int64)')
                node.value.left, node.value.right = node.value.right, node.value.left
                restored += 1
        self.assertEqual(restored, 4)
        self.assertEqual(ast.dump(original), ast.dump(candidate))
        provenance = variant_provenance(CASE)
        for field in ['original', 'candidate']:
            self.assertEqual(hashlib.sha256(variant_source(CASE / provenance[field]).encode()).hexdigest(),
                             provenance[field + '_sha256'])
