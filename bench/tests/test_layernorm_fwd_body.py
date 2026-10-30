"""Keep complete LayerNorm bodies, source conversions and caller specialization."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check, check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/layer_norm_fwd_nonzero_tile'

class LayerNormFwdBodyTests(unittest.TestCase):
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

    def test_complete_runtime_and_one_bodies(self):
        result = check_manifest(ROOT, BASE / 'jit-correspondence.json')
        self.assertEqual([r['statements'] for r in result], [24] * 4)
        self.assertTrue(all(r['status'] == 'PASS' and r['mode'] == 'pasted_body' for r in result))

    def test_program_id_cast_is_retained(self):
        self.reject('tl.program_id(0).to(tl.int64)', 'tl.program_id(0)')

    def test_seed_storage_does_not_replace_source_cast(self):
        self.reject('("SEEDS",.i64)', '("SEEDS",.u32)')
        self.reject('tl.load(SEEDS + row).to(tl.uint32)', 'tl.load(SEEDS + row)')

    def test_input_cast_is_retained(self):
        self.reject('tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)',
                    'tl.load(X + cols, mask=cols < N, other=0.0)')

    def test_store_does_not_gain_a_cast(self):
        self.reject('tl.store(Y + cols, y, mask=mask)', 'tl.store(Y + cols, y.to(tl.float32), mask=mask)')
        self.reject('tl.store(Rstd + row, rstd)', 'tl.store(Rstd + row, rstd.to(tl.float32))')

    def test_random_rounds_and_stream_address_are_retained(self):
        self.reject('n_rounds=7', 'n_rounds=10')
        self.reject('tl.load(SEEDS + M + row)', 'tl.load(SEEDS + row)')

    def test_mask_keeps_exclusive_upper_bound(self):
        self.reject('mask=cols < N', 'mask=cols <= N')

    def test_optional_secondary_affine_remains(self):
        self.reject('if HAS_W1:', 'if HAS_X1:')

    def test_count_one_requires_the_actual_caller_value(self):
        self.reject('let N : Nat := 1', 'let N : Nat := 2', row=2)

    def test_complete_candidate_module_only_commutes_initial_x_product(self):
        original = ast.parse((BASE / 'layer_norm_fwd_nonzero_tile.py').read_text())
        candidate = ast.parse((BASE / 'layer_norm_fwd_index_commuted.py').read_text())
        expected = copy.deepcopy(original)
        fn = next(n for n in expected.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_layer_norm_fwd_1pass_kernel')
        stmt = next(n for n in fn.body if isinstance(n, ast.AugAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == 'X')
        self.assertIsInstance(stmt.op, ast.Add)
        self.assertIsInstance(stmt.value, ast.BinOp)
        self.assertIsInstance(stmt.value.op, ast.Mult)
        stmt.value.left, stmt.value.right = stmt.value.right, stmt.value.left
        self.assertEqual(ast.dump(expected), ast.dump(candidate))
