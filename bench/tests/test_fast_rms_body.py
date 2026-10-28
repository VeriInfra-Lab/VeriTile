"""Raw-body and candidate correspondence, without Torch or GPU imports."""
import ast
import copy
import json
from pathlib import Path
import unittest

from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/fast_rms_layernorm'
MANIFEST = BASE / 'jit-correspondence.json'


class FastRmsBody(unittest.TestCase):
    def setUp(self):
        self.pairs = json.loads(MANIFEST.read_text())

    def assert_rejected(self, definition, before, after):
        row = next(r for r in self.pairs if r['lean_definition'] == definition)
        lean = (ROOT / row['lean']).read_text()
        start = lean.index('def ' + definition + ' ')
        end = lean.find('\ndef ', start + 1)
        if end < 0:
            end = len(lean)
        part = lean[start:end]
        self.assertIn(before, part)
        mutated = lean[:start] + part.replace(before, after, 1) + lean[end:]
        with self.assertRaises(ValueError):
            check((ROOT / row['python']).read_text(), mutated,
                  row['python_function'], row['lean_definition'],
                  signature_definition=row['signature_definition'],
                  expected_signature=row['expected_signature'],
                  expected_call_bindings=row['expected_call_bindings'])

    def test_all_bodies(self):
        result = check_manifest(ROOT, MANIFEST)
        self.assertEqual(len(result), 11)
        self.assertTrue(all(r['status'] == 'PASS' for r in result))

    def test_program_id_cast_cannot_disappear(self):
        self.assert_rejected('regularRuntimeSource', 'tl.program_id(0).to(tl.int64)',
                             'tl.program_id(0)')

    def test_value_dtype_cast_cannot_disappear(self):
        self.assert_rejected('regularRuntimeSource', 'normed = normed.to(W_row.dtype)',
                             'normed = normed')

    def test_output_cast_cannot_be_invented(self):
        self.assert_rejected('gemmaRuntimeSource', 'tl.store(Y + col_offsets, output, mask=mask)',
                             'tl.store(Y + col_offsets, output.to(tl.float32), mask=mask)')

    def test_weight_counter_cast_cannot_disappear(self):
        self.assert_rejected('weightSource', 'range(tl.cast(n_rows, tl.int64))',
                             'range(n_rows)')

    def test_weight_dtype_roundtrip_cannot_disappear(self):
        self.assert_rejected('weightSource', 'normed.to(dW.dtype.element_ty).to(tl.float32)',
                             'normed.to(tl.float32)')

    def test_gemma_branch_cannot_be_erased(self):
        self.assert_rejected('inputGradientBody', 'if GEMMA: dY_W = dY_row * (W_row + 1.0)',
                             'if GEMMA: dY_W = dY_row * W_row')

    def test_full_candidate_module(self):
        original = ast.parse((BASE / 'fast_rms_layernorm_repaired.py').read_text())
        expected = copy.deepcopy(original)
        names = {n: n + '_index_commuted' for n in
                 ['_rms_layernorm_forward', '_gemma_rms_layernorm_forward',
                  '_rms_layernorm_backward']}
        for fn in expected.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in names:
                updates = [n for n in fn.body if isinstance(n, ast.AugAssign)
                           and isinstance(n.target, ast.Name) and n.target.id == 'X']
                self.assertEqual(len(updates), 1)
                product = updates[0].value
                self.assertIsInstance(product, ast.BinOp)
                self.assertIsInstance(product.op, ast.Mult)
                self.assertEqual((product.left.id, product.right.id),
                                 ('row_idx', 'X_row_stride'))
                product.left, product.right = product.right, product.left

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in names:
                    node.id = names[node.id]
                return node

            def visit_FunctionDef(self, node):
                if node.name in names:
                    node.name = names[node.name]
                return self.generic_visit(node)

        expected = Rename().visit(expected)
        actual = ast.parse((BASE / 'fast_rms_layernorm_index_commuted.py').read_text())
        self.assertEqual(ast.dump(expected), ast.dump(actual))


if __name__ == '__main__':
    unittest.main()
