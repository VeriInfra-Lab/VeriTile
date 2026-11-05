import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/lightning_attention'


class LightningAttentionBodyMigration(unittest.TestCase):
    def setUp(self):
        self.rows = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_complete_forward_intra_inter_and_candidates(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in rows], [15, 15, 33, 33, 24, 24])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in rows))

    def mutation(self, index, before, after):
        row = self.rows[index]
        self.assertIn(before, self.lean)
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), self.lean.replace(before, after),
                  row['python_function'], row['lean_definition'], row['signature_definition'],
                  row['expected_signature'], row['expected_call_bindings'])

    def test_casts_negative_differences_masks_and_reverse_loops_are_preserved(self):
        for index, before, after in [
            (0, 'index = off_block[:, None] - off_block[None, :]',
             'index = (off_block[:, None] - off_block[None, :]).to(tl.int32)'),
            (0, 'for i in range(NUM_BLOCK):', 'for i in range(NUM_BLOCK - 1):'),
            (2, 'array = tl.arange(0, BLOCK).to(tl.float32)', 'array = tl.arange(0, BLOCK)'),
            (2, 'do = tl.load', 'go = tl.load'),
            (4, 'range(NUM_BLOCK - 1, -1, -1)', 'range(NUM_BLOCK)'),
        ]:
            with self.subTest(before=before):
                self.mutation(index, before, after)

    def test_call_storage_and_read_order_are_checked(self):
        self.mutation(0, '("Q",.fp32)', '("Q",.fp16)')
        self.mutation(2, 'readPorts := ["K","V","DO","Q"]', 'readPorts := ["Q","K","V","DO"]')

    def test_candidate_only_reuses_three_existing_offsets(self):
        meta = variant_provenance(CASE)
        source = variant_source(CASE / meta['original'])
        candidate = variant_source(CASE / meta['candidate'])
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate.encode()).hexdigest(), meta['candidate_sha256'])
        before = 'o_offset = off_bh * n * e'
        after = 'o_offset = v_offset'
        self.assertEqual(candidate.count(after), 3)
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(candidate.replace(after, before))))
