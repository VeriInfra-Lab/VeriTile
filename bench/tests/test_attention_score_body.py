"""Check the complete copied score source and its existing local candidate."""
import ast
import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from bench.jit_correspondence import check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('bench/repairs/tritonbench_g/attention_score_tile_checked')

class AttentionScoreBody(unittest.TestCase):
    def test_two_complete_bodies(self):
        self.assertEqual(len(check_manifest(ROOT, ROOT / BASE / 'jit-correspondence.json')), 2)

    def reject_mutation(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / BASE
            dest.mkdir(parents=True)
            for name in ['Kernels.lean', 'attention_score_tile_checked.py',
                         'attention_score_scalar_left.py', 'jit-correspondence.json']:
                shutil.copy2(ROOT / BASE / name, dest / name)
            path = dest / 'Kernels.lean'
            text = path.read_text()
            start = text.index('def widened\n')
            self.assertIn(before, text[start:])
            path.write_text(text[:start] + text[start:].replace(before, after, 1))
            with self.assertRaises(ValueError):
                check_manifest(root, dest / 'jit-correspondence.json')

    def test_program_cast(self):
        self.reject_mutation('(tl.program_id(0)).to(tl.int64)', 'tl.program_id(0)')

    def test_block_offset_narrowing(self):
        self.reject_mutation('(start_n * BLOCK_N).to(tl.int32)', 'start_n * BLOCK_N')

    def test_query_bound_cast(self):
        self.reject_mutation('hi = tl.cast(N_CTX, tl.int64)', 'hi = N_CTX')

    def test_query_tail_mask(self):
        self.reject_mutation('start_m + tl.arange(0, BLOCK_M) < N_CTX',
                             'start_m + tl.arange(0, BLOCK_M) < NKV_CTX')

    def test_default_dot_options(self):
        self.reject_mutation('qk += tl.dot(q, k)', 'qk += tl.dot(q, k, allow_tf32=False)')

    def test_signed_window(self):
        self.reject_mutation('tl.cast(sliding_window_offset, tl.int64)', 'sliding_window_offset')

    def test_store_conversion(self):
        self.reject_mutation('o.to(Out.type.element_ty)', 'o')

    def test_source_pointer_advance(self):
        self.reject_mutation('offsets=(BLOCK_M, 0)', 'offsets=(BLOCK_N, 0)')

    def test_only_existing_candidate_operand_swap(self):
        def function(file, name):
            return next(n for n in ast.parse((ROOT / BASE / file).read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        original = function('attention_score_tile_checked.py', '_score_kernel')
        candidate = copy.deepcopy(function('attention_score_scalar_left.py', '_score_scalar_left_kernel'))
        edits = 0
        for node in ast.walk(candidate):
            if isinstance(node, ast.Assign) and ast.dump(node.value) == ast.dump(ast.parse('qk_scale * qk', mode='eval').body):
                node.value.left, node.value.right = node.value.right, node.value.left
                edits += 1
        self.assertEqual(edits, 1)
        self.assertEqual(ast.dump(original.args), ast.dump(candidate.args))
        self.assertEqual([ast.dump(n) for n in original.body], [ast.dump(n) for n in candidate.body])

if __name__ == '__main__':
    unittest.main()
