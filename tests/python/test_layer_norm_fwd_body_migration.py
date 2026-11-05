import ast
import copy
import hashlib
from pathlib import Path
import unittest
from scripts.bench.jit_correspondence import check

from tests.python.layout_fixtures import check_pinned_pairs, pinned_pairs, variant_provenance, variant_source

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / 'bench/tritonbench_g/layer_norm_fwd'


class LayerNormFwdBodyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.lean = (DIRECTORY / 'Kernels.lean').read_text()
        self.records = pinned_pairs(DIRECTORY)

    def verify(self, text):
        r = self.records[0]
        return check((ROOT / r['python']).read_text(), text, r['python_function'],
                     r['lean_definition'], r['signature_definition'], r['expected_signature'],
                     expected_call_bindings=r['expected_call_bindings'], expected_integer_args=r['expected_integer_args'], expected_arange_args=r['expected_arange_args'])

    def test_both_complete_bodies(self):
        self.assertEqual([r['statements'] for r in check_pinned_pairs(ROOT, DIRECTORY)], [24, 24])

    def test_cast_mask_sentinel_and_final_return_are_checked(self):
        for before, after in [
            ('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)'),
            ('n_rounds=7', 'n_rounds=10'),
            ('x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)',
             'x = tl.load(X + cols, mask=cols < N, other=0.0)'),
            ('x += x1', 'x = x1'),
            ('tl.store(Mean + row, mean)', 'tl.store(Mean + row, var)'),
            ('if HAS_W1:', 'if HAS_B1:'),
            ('x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd', 'x_hat = x * rstd'),
        ]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_call_dtype_and_stride_binding_are_checked(self):
        for before, after in [('("SEEDS",.i64)', '("SEEDS",.i32)'),
                              ('stride_x_row := stride_x_row', 'stride_x_row := 0')]:
            with self.subTest(before=before):
                self.assertIn(before, self.lean)
                with self.assertRaises(ValueError):
                    self.verify(self.lean.replace(before, after))

    def test_existing_candidate_restores_entire_module(self):
        original_text = variant_source(DIRECTORY / 'layer_norm_fwd.py')
        candidate_text = variant_source(DIRECTORY / 'layer_norm_fwd_candidates.py')
        record = variant_provenance(DIRECTORY)
        self.assertEqual(hashlib.sha256(original_text.encode()).hexdigest(), record['original_sha256'])
        self.assertEqual(hashlib.sha256(candidate_text.encode()).hexdigest(), record['candidate_sha256'])
        names = {'_layer_norm_fwd_1pass_kernel_index_commuted': '_layer_norm_fwd_1pass_kernel',
                 '_layer_norm_fwd_index_commuted': '_layer_norm_fwd'}
        class Restore(ast.NodeTransformer):
            def visit_Name(self, node):
                node.id = names.get(node.id, node.id)
                return node
            def visit_FunctionDef(self, node):
                node.name = names.get(node.name, node.name)
                return self.generic_visit(node)
        restored = Restore().visit(ast.parse(candidate_text))
        kernel = next(n for n in restored.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_layer_norm_fwd_1pass_kernel')
        statement = next(n for n in kernel.body if isinstance(n, ast.AugAssign)
                         and ast.unparse(n.target) == 'X')
        self.assertEqual(ast.unparse(statement.value), 'stride_x_row * row')
        statement.value = ast.parse('row * stride_x_row', mode='eval').body
        self.assertEqual(ast.dump(restored, include_attributes=False),
                         ast.dump(ast.parse(original_text), include_attributes=False))
