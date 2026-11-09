import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/chunk_gla_fwd'


class ChunkGlaBodyMigration(unittest.TestCase):
    def setUp(self):
        self.rows = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_all_five_originals_and_candidates_are_complete(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in rows], [8, 8, 15, 15, 17, 17, 7, 7, 12, 12])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in rows))

    def check_mutation(self, index, before, after):
        row = self.rows[index]
        self.assertIn(before, self.lean)
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), self.lean.replace(before, after),
                  row['python_function'], row['lean_definition'], row['signature_definition'],
                  row['expected_signature'], row['expected_call_bindings'])

    def test_source_casts_masks_returns_and_padding_cannot_change(self):
        for index, before, after in [
            (0, 'if i_i <= i_j:', 'if i_i < i_j:'),
            (0, 'b_A.to(A.dtype.element_ty)', 'b_A'),
            (2, 'padding_option="zero"', 'padding_option="nan"'),
            (2, 'min(BC, T-i_t*BT-i_i*BC)', 'BC'),
            (8, '(b_q * tl.exp(b_g)).to(b_q.dtype)', '(b_q * tl.exp(b_g))'),
            (8, 'if i_k >= 0:', 'if i_k > 0:'),
        ]:
            with self.subTest(index=index, before=before):
                self.check_mutation(index, before, after)

    def test_dtype_profiles_and_consumer_ports_are_checked(self):
        self.check_mutation(0, '| .fp16=>.fp16', '| .fp16=>.bf16')
        self.check_mutation(8, 'readPorts := ["q","v","g","h","A"]',
                            'readPorts := ["q","g","h","v","A"]')

    def test_candidate_is_only_the_five_existing_operand_swaps(self):
        meta = variant_provenance(CASE)
        original = variant_source(CASE / meta['original'])
        candidate = variant_source(CASE / meta['candidate'])
        for label, source in [('original', original), ('candidate', candidate)]:
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta[label + '_sha256'])
        changes = [
            ('b_A += tl.dot(b_qg, b_kg)', 'b_A = tl.dot(b_qg, b_kg) + b_A', 1),
            ('b_A = tl.where(o_i >= j, b_A * scale, 0.)',
             'b_A = tl.where(o_i >= j, scale * b_A, 0.)', 2),
            ('b_A += tl.load(p_A, boundary_check=(0, 1), padding_option="zero")',
             'b_A = tl.load(p_A, boundary_check=(0, 1), padding_option="zero") + b_A', 1),
            ('b_o += tl.dot(b_A, b_v, allow_tf32=False)',
             'b_o = tl.dot(b_A, b_v, allow_tf32=False) + b_o', 1),
        ]
        for before, after, count in changes:
            self.assertEqual(candidate.count(after), count)
            candidate = candidate.replace(after, before)
        self.assertEqual(ast.dump(ast.parse(original)), ast.dump(ast.parse(candidate)))
