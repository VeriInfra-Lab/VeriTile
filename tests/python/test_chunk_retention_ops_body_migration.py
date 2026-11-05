import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/chunk_retention_ops'


class ChunkRetentionOpsBodyMigration(unittest.TestCase):
    def setUp(self):
        self.rows = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_every_original_and_candidate_stage_is_complete(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in rows], [9, 9, 16, 16, 26, 26, 9, 9])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in rows))

    def mutation(self, index, before, after):
        row = self.rows[index]
        self.assertIn(before, self.lean)
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), self.lean.replace(before, after),
                  row['python_function'], row['lean_definition'], row['signature_definition'],
                  row['expected_signature'], row['expected_call_bindings'])

    def test_signed_casts_tail_decay_masks_and_reverse_loop_cannot_change(self):
        for index, before, after in [
            (0, 'tl.program_id(2).to(tl.int64)', 'tl.program_id(2)'),
            (0, 'if i_t == NT - 1 and (T % BT) != 0:', 'if i_t == NT - 1:'),
            (0, '(b_v * d_i[:, None]).to(b_k.dtype)', '(b_v * d_i[:, None])'),
            (2, 'o_i[:, None] >= o_i[None, :]', 'o_i[:, None] > o_i[None, :]'),
            (4, 'tl.num_programs(2).to(tl.int64)', 'tl.num_programs(2)'),
            (4, 'b_dv.to(p_dv.dtype.element_ty)', 'b_dv'),
            (6, 'range(NT - 1, -1, -1)', 'range(NT)'),
            (6, 'if USE_FINAL_GRAD:', 'if True:'),
            (6, 'if STORE_INITIAL_GRAD:', 'if True:'),
            (6, 'tl.minimum(BT, T - i_t * BT)', 'BT'),
        ]:
            with self.subTest(before=before):
                self.mutation(index, before, after)

    def test_caller_storage_precision_and_read_ports_are_checked(self):
        self.mutation(0, '("h0",.fp32)', '("h0",.fp16)')
        self.mutation(6, 'readPorts := ["q","do","d_ht"]', 'readPorts := ["q","d_ht","do"]')

    def test_candidate_only_commutes_the_existing_final_multiply(self):
        meta = variant_provenance(CASE)
        source = variant_source(CASE / meta['original'])
        candidate = variant_source(CASE / meta['candidate'])
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate.encode()).hexdigest(), meta['candidate_sha256'])
        before = 'b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale'
        after = 'b_o = scale * (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False))'
        self.assertEqual(candidate.count(after), 1)
        self.assertEqual(ast.dump(ast.parse(source)), ast.dump(ast.parse(candidate.replace(after, before))))
