"""Check complete original token-attention bodies and the existing candidate."""
import ast
import copy
import json
from pathlib import Path
import unittest
from bench.jit_correspondence import check

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'bench/repairs/tritonbench_g/token_attn_llama2_grid_axes_checked'

class TokenAttentionBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lean = (BASE / 'Kernels.lean').read_text()
        cls.rows = json.loads((BASE / 'jit-correspondence.json').read_text())

    def run_row(self, row, lean=None):
        return check((ROOT / row['python']).read_text(), self.lean if lean is None else lean,
                     **{k: v for k, v in row.items() if k not in ('python', 'lean')})

    def test_complete_correspondence(self):
        self.assertEqual(len(self.rows), 2)
        for row in self.rows:
            self.assertEqual(self.run_row(row)['status'], 'PASS')

    def test_existing_candidate_only_reuses_block_origin(self):
        functions = []
        for row in self.rows:
            functions.append(next(n for n in ast.parse((ROOT / row['python']).read_text()).body
                                  if isinstance(n, ast.FunctionDef) and n.name == row['python_function']))
        a, b = functions
        self.assertEqual(len(a.body), 15)
        b = copy.deepcopy(b)
        b.name = a.name
        self.assertEqual(ast.unparse(b.body[10]), 'block_stard_index = start_n * BLOCK_N')
        self.assertEqual(ast.unparse(b.body[11]), 'offs_n = block_stard_index + tl.arange(0, BLOCK_N)')
        b.body[10:12] = copy.deepcopy(a.body[10:12])
        self.assertEqual(ast.dump(a), ast.dump(b))

    def test_input_cast_cannot_be_invented(self):
        text = self.lean.replace('q = tl.load(Q + off_q + start_mark)',
                                 'q = tl.load(Q + off_q + start_mark).to(tl.float32)', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_loop_cannot_lose_its_runtime_bound(self):
        text = self.lean.replace('range(0, block_mask, 1)', 'range(0, 1, 1)', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_reduction_axis_cannot_change(self):
        text = self.lean.replace('tl.sum(q[None, :] * k, 1)', 'tl.sum(q[None, :] * k, 0)', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_gather_mask_cannot_change(self):
        text = self.lean.replace('mask=offs_n_new < cur_batch_end_index',
                                 'mask=offs_n_new <= cur_batch_end_index', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

    def test_scaling_cannot_change(self):
        text = self.lean.replace('att_value *= sm_scale', 'att_value += sm_scale', 1)
        with self.assertRaises(ValueError): self.run_row(self.rows[0], text)

if __name__ == '__main__':
    unittest.main()
