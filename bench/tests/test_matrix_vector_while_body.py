"""Protect the actual unsigned while bodies, including the explicit FMA pair."""
import ast
import copy
import shutil
import tempfile
import unittest
from pathlib import Path

from bench.jit_correspondence import check_manifest

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('bench/repairs/tritonbench_g/matrix_vector_multip')


class MatrixVectorWhileBody(unittest.TestCase):
    def test_two_complete_bodies(self):
        self.assertEqual(len(check_manifest(ROOT, ROOT / BASE / 'jit-correspondence.json')), 2)

    def reject_mutation(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / BASE
            dest.mkdir(parents=True)
            for name in ['Kernels.lean', 'matrix_vector_multip_repaired.py',
                         'jit-correspondence.json']:
                shutil.copy2(ROOT / BASE / name, dest / name)
            path = dest / 'Kernels.lean'
            text = path.read_text()
            self.assertIn(before, text)
            path.write_text(text.replace(before, after, 1))
            with self.assertRaises(ValueError):
                check_manifest(root, dest / 'jit-correspondence.json')

    def test_all_six_unsigned_casts(self):
        for name in ['N', 'M', 'stride_an', 'stride_am', 'stride_bm', 'stride_cn']:
            with self.subTest(name=name):
                self.reject_mutation(f'{name} = tl.cast({name}, tl.uint64)', f'{name} = {name}')

    def test_program_cast(self):
        self.reject_mutation('tl.program_id(0).to(tl.uint64)', 'tl.program_id(0)')

    def test_while_guard(self):
        self.reject_mutation('while m < M:', 'while m <= M:')

    def test_unsigned_initial_counter(self):
        self.reject_mutation('m = tl.full((), 0, tl.uint64)', 'm = 0')

    def test_counter_advance(self):
        self.reject_mutation('m += BLOCK_M', 'm += 1')

    def test_tail_mask(self):
        self.reject_mutation('mask=n_mask & m_mask', 'mask=n_mask')

    def test_matrix_pointer_advance(self):
        self.reject_mutation('A_ptrs += BLOCK_M * stride_am', 'A_ptrs += BLOCK_M')

    def test_float_cast(self):
        self.reject_mutation('other=0.0).to(tl.float32)', 'other=0.0)')

    def test_existing_candidate_changes_only_accumulation(self):
        module = ast.parse((ROOT / BASE / 'matrix_vector_multip_repaired.py').read_text())
        functions = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
        original = copy.deepcopy(functions['mv_contiguous_kernel'])
        candidate = copy.deepcopy(functions['mv_contiguous_fma_kernel'])
        class Fuse(ast.NodeTransformer):
            changes = 0
            def visit_AugAssign(self, node):
                if ast.dump(node) == ast.dump(ast.parse('acc += a * b').body[0]):
                    self.changes += 1
                    return ast.parse('acc = tl.fma(a, b, acc)').body[0]
                return self.generic_visit(node)
        transform = Fuse()
        transform.visit(original)
        self.assertEqual(transform.changes, 1)
        self.assertEqual(ast.dump(original.args), ast.dump(candidate.args))
        self.assertEqual([ast.dump(n) for n in original.body], [ast.dump(n) for n in candidate.body])


if __name__ == '__main__':
    unittest.main()
