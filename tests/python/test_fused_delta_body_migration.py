import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check, check_statement_function

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/fused_recurrent_delta'


class FusedDeltaBodyMigration(unittest.TestCase):
    def setUp(self):
        self.records = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_original_and_candidate_forward_backward_bodies(self):
        result = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in result], [13, 27, 13, 27])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in result))

    def test_cast_store_reload_and_conditional_mutations_are_rejected(self):
        for index, before, after in [
            (0, 'other=0).to(tl.float32)', 'other=0)'),
            (0, 'b_v.to(p_v.dtype.element_ty)', 'b_v'),
            (0, 'p_beta += V if IS_HEADWISE_BETA else 1', 'p_beta += V'),
            (1, 'beta + i_bh * T + T - 1', 'beta + i_bh * T'),
            (1, 'tl.debug_barrier()', 'ignored = 0'),
            (1, 'd_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)', 'd_k = 0.0'),
        ]:
            row = self.records[index]
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    check_statement_function((ROOT / row['python']).read_text(),
                                             self.lean.replace(before, after),
                                             row['python_function'], row['lean_definition'])

    def test_runtime_length_cannot_be_relabelled_constexpr(self):
        row = self.records[0]
        mutated = self.lean.replace('("T",.integer)', '("T",.fp32)')
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), mutated, row['python_function'],
                  row['lean_definition'], row['signature_definition'],
                  row['expected_signature'], row['expected_call_bindings'])

    def test_candidate_changes_only_the_forward_output_grid_term(self):
        meta = variant_provenance(CASE)
        sources = {k: variant_source(CASE / meta[k]) for k in ['original', 'candidate']}
        for k, source in sources.items():
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), meta[k + '_sha256'])
        trees = {k: ast.parse(s) for k, s in sources.items()}
        forward = next(n for n in trees['candidate'].body if isinstance(n, ast.FunctionDef)
                       and n.name == 'fused_recurrent_fwd_kernel')
        stores = [n for n in forward.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'p_o' for t in n.targets)]
        self.assertEqual(len(stores), 1)
        expected = ast.parse('p_o = o + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)').body[0]
        self.assertEqual(ast.dump(stores[0]), ast.dump(expected))
        original = ast.parse('p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)').body[0]
        forward.body[forward.body.index(stores[0])] = original
        self.assertEqual(ast.dump(trees['candidate']), ast.dump(trees['original']))
