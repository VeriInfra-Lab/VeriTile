import ast
import hashlib
from pathlib import Path
import unittest

from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / 'bench/tritonbench_g/chunk_linear_attn'


class ChunkLinearBodyMigration(unittest.TestCase):
    def setUp(self):
        self.rows = pinned_pairs(CASE)
        self.lean = (CASE / 'Kernels.lean').read_text()

    def test_all_four_original_and_candidate_stages_are_complete(self):
        rows = check_pinned_pairs(ROOT, CASE)
        self.assertEqual([r['statements'] for r in rows], [5, 5, 12, 13, 5, 5, 20, 20])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in rows))

    def check_mutation(self, index, before, after):
        row = self.rows[index]
        self.assertIn(before, self.lean)
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), self.lean.replace(before, after),
                  row['python_function'], row['lean_definition'], row['signature_definition'],
                  row['expected_signature'], row['expected_call_bindings'])

    def test_reverse_order_casts_masks_and_endpoint_branches_are_checked(self):
        for index, before, after in [
            (0, 'if USE_INITIAL_STATE:', 'if STORE_FINAL_STATE:'),
            (0, 'b_h.to(p_h.dtype.element_ty)', 'b_h'),
            (2, 'b_o.to(p_o.dtype.element_ty)', 'b_o'),
            (4, 'range(NT - 1, -1, -1)', 'range(NT)'),
            (4, 'if USE_FINAL_GRADIENT:', 'if STORE_INITIAL_GRADIENT:'),
            (4, 'padding_option="zero").to(tl.float32)', 'padding_option="zero")'),
            (6, 'b_dq.to(p_dq.dtype.element_ty)', 'b_dq'),
            (6, 'boundary_check=(0, 1)', 'boundary_check=(0,)'),
        ]:
            with self.subTest(index=index, before=before):
                self.check_mutation(index, before, after)

    def test_source_precision_and_input_ports_are_checked(self):
        self.check_mutation(0, '("k",.fp32)', '("k",.fp16)')
        self.check_mutation(4, 'readPorts := ["q","do","dht"]',
                            'readPorts := ["q","dht","do"]')

    def test_candidate_only_hoists_the_existing_head_address(self):
        meta = variant_provenance(CASE)
        source = variant_source(CASE / meta['source'])
        candidate = variant_source(CASE / meta['candidate'])
        for label, text in [('source', source), ('candidate', candidate)]:
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), meta[label + '_sha256'])
        expected = ast.parse(source)
        actual = ast.parse(candidate)
        original_body = next(n.body for n in expected.body
                             if isinstance(n, ast.FunctionDef) and n.name == 'chunk_linear_attn_fwd_kernel_o')
        new_body = next(n.body for n in actual.body
                       if isinstance(n, ast.FunctionDef) and n.name == 'chunk_linear_attn_fwd_kernel_o')
        hoist = next(n for n in new_body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'qk_head')
        self.assertEqual(ast.dump(hoist), ast.dump(ast.parse('qk_head = i_bh * s_qk_h').body[0]))
        loop = next(n for n in new_body if isinstance(n, ast.For))
        self.assertIs(new_body[new_body.index(loop) - 1], hoist)
        new_body.remove(hoist)

        class RestoreBases(ast.NodeTransformer):
            count = 0

            def visit_Name(self, node):
                if node.id == 'qk_head':
                    self.count += 1
                    return ast.parse('i_bh * s_qk_h', mode='eval').body
                return node

        restore = RestoreBases()
        actual = restore.visit(actual)
        self.assertEqual(restore.count, 2)
        self.assertEqual(len(original_body), 12)
        self.assertEqual(ast.dump(expected), ast.dump(actual))
