"""Raw TMA matrix-product bodies and the corrected conversion comparison."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/matmul_tma_static_checked_v2'

class MatmulTMABodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (BASE / 'matmul_tma_static_checked_v2.py').read_text()
        cls.candidate = (BASE / 'candidate_kernels.py').read_text()
        cls.lean = (BASE / 'KernelOnly/Kernels.lean').read_text()
        cls.rows = json.loads((BASE / 'KernelOnly/jit-correspondence.json').read_text())

    def run_row(self, row, lean=None):
        return check((ROOT / row['python']).read_text(), lean if lean is not None else self.lean,
                     **{k: v for k, v in row.items() if k not in ('python', 'lean')})

    def test_complete_correspondence(self):
        self.assertEqual(len(self.rows), 2)
        for row in self.rows:
            self.assertEqual(self.run_row(row)['status'], 'PASS')

    def test_candidate_only_adds_the_historical_output_cast(self):
        a = next(n for n in ast.parse(self.original).body if isinstance(n, ast.FunctionDef)
                 and n.name == 'matmul_tma_load_store')
        b = next(n for n in ast.parse(self.candidate).body if isinstance(n, ast.FunctionDef)
                 and n.name == 'matmul_tma_explicit_output')
        b = copy.deepcopy(b); b.name = a.name
        self.assertEqual(len(a.body), 8)
        self.assertIsInstance(a.body[6], ast.If)
        self.assertEqual(a.body[6].orelse, [])
        self.assertEqual(len(b.body[6].orelse), 1)
        self.assertEqual(ast.unparse(b.body[6].orelse[0]), 'c = c.to(c_ptr.type.element_ty)')
        b.body[6].orelse = []
        self.assertEqual(ast.dump(a), ast.dump(b))

    def test_original_cannot_gain_a_cast(self):
        text = self.lean.replace('c = tl.dot(a, b)', 'c = tl.dot(a, b).to(tl.float32)', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_half_cast_cannot_disappear(self):
        text = self.lean.replace('c = c.to(tl.float16)', 'c = c', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_condition_cannot_disappear(self):
        text = self.lean.replace('if OUTPUT_F16:', 'if True:', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_block_pointer_order_is_preserved(self):
        text = self.lean.replace('order=(0, 1)', 'order=(1, 0)', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_dot_options_cannot_change(self):
        text = self.lean.replace('tl.dot(a, b)', 'tl.dot(a, b, input_precision="ieee")', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

if __name__ == '__main__':
    unittest.main()
