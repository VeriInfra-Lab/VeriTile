"""Keep the exact signed three-pass source and its existing scheduling candidate."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/layernorm_fwd_triton'

class ThreePassLayerNormBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads((BASE / 'jit-correspondence.json').read_text())
        cls.source = (BASE / 'Kernels.lean').read_text()

    def reject(self, before, after, row=0):
        changed = self.source.replace(before, after)
        self.assertNotEqual(self.source, changed)
        record = self.rows[row]
        with self.assertRaises(ValueError):
            check((ROOT / record['python']).read_text(), changed,
                  **{k: v for k, v in record.items() if k not in ('python', 'lean', 'profile_variant')})

    def test_complete_bodies(self):
        result = check_manifest(ROOT, BASE / 'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in result], [25, 25])
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in result))

    def test_program_id_casts(self):
        for dim in [0, 1]:
            self.reject(f'tl.program_id({dim}).to(tl.int64)', f'tl.program_id({dim})')

    def test_scalar_casts(self):
        for name in ['stride_x_N', 'stride_x_hn', 'stride_x_hd', 'stride_y_N',
                     'stride_y_hn', 'stride_y_hd', 'stride_w_hn', 'stride_w_hd', 'N']:
            self.reject(f'tl.cast({name}, tl.int64)', name)

    def test_signed_lanes_and_loop_bounds(self):
        self.reject('tl.arange(0, BLOCK_SIZE).to(tl.int64)', 'tl.arange(0, BLOCK_SIZE)')
        self.reject('range(zero, N, step)', 'range(0, N, BLOCK_SIZE)')
        self.reject('step = tl.full((), BLOCK_SIZE, tl.int64)', 'step = BLOCK_SIZE')

    def test_actual_inner_strides(self):
        for name in ['stride_x_hd', 'stride_y_hd', 'stride_w_hd']:
            self.reject(f'cols * {name}', 'cols')

    def test_masked_variance_zero(self):
        self.reject('tl.where(cols < N, x - mean, 0.0)', 'x - mean')

    def test_explicit_output_cast(self):
        self.reject('y.to(X.dtype.element_ty)', 'y')

    def test_weight_load_order(self):
        before = '        w = tl.load(W + cols * stride_w_hd, mask=mask).to(tl.float32)\n'
        # Removing the load cannot be hidden by retaining the mathematical product.
        self.reject(before, '')

    def test_runtime_epsilon_is_not_a_literal(self):
        self.reject('var + eps', 'var + 0.00001')

    def test_full_candidate_only_reschedules_weight_load_and_renames_entries(self):
        original = ast.parse((BASE / 'layernorm_fwd_triton_repaired.py').read_text())
        candidate = ast.parse((BASE / 'layernorm_fwd_triton_scheduled.py').read_text())
        expected = copy.deepcopy(original)
        fn = next(n for n in expected.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_layer_norm_fwd_kernel')
        body = fn.body[-1].body
        self.assertEqual(body[2].targets[0].id, 'w')
        weight = body.pop(2)
        self.assertEqual(body[3].targets[0].id, 'x_hat')
        body.insert(4, weight)
        names = {'_layer_norm_fwd_kernel': '_layer_norm_fwd_kernel_scheduled',
                 'layernorm_forward': 'layernorm_forward_scheduled'}
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
        self.assertEqual(ast.dump(expected), ast.dump(candidate))
